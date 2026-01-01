from doctest import OutputChecker
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset
import json
import os
from transformers.trainer_utils import EvalLoopOutput
import time


MODEL_NAME = "/model_file/Qwen3-Embedding-4B"
DATA_FILE = '/BIRD_Data/data/embedding_training_data.json'  
#DATA_FILE = '/home/yfwang/wyy/schema_routing/data/spider/spider_data/toy_train.json'
OUTPUT_DIR = "/model_file/Qwen3-Embedding-4B-lora"
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 16      
MAX_SEQ_LENGTH = 512 
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
TEMPERATURE = 0.05
# ##################################################


def process_data_generator(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                data = json.loads(line)
                query = f"Instruction: {data['instruction']}\nQuery: {data['input']}".strip()
                for pos_passage in data["outputs"]:
                    yield {"query": query, "pos": pos_passage}

print("🚀 Processing...")
train_dataset = Dataset.from_generator(process_data_generator, gen_kwargs={"file_path": DATA_FILE})
print(f"✅ Dataset loaded with a total of {len(train_dataset)} training samples.")
print("Sample:", train_dataset[0])

# --- Shuffle dataset ---
print("🔀 Shuffling dataset...")
train_dataset = train_dataset.shuffle(seed=42)
print("✅ Dataset shuffled.")

# --- Split dataset into training and validation sets ---
print("🔪 Splitting dataset...")
split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']
print(f"✅ Dataset split completed: {len(train_dataset)} training samples, {len(eval_dataset)} validation samples.")
print("-" * 50)


# --- 3. Load Model & Tokenizer ---
print("🚀 Loading Model and Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  
)


if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
print("✅ Model and Tokenizer loaded.")
print("-" * 50)


# --- 4. LoRA Configuration ---
print("🚀 Configuring LoRA...")
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, peft_config)
print("✅ LoRA configuration completed. Trainable parameters are as follows:")
model.print_trainable_parameters()
print("-" * 50)


# --- 5. Custom Collator & Trainer ---
class PairCollator:
    """Data collator that tokenizes (query, pos) pairs"""
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, features):
        all_texts = [f['query'] for f in features] + [f['pos'] for f in features]
        tokenized_inputs = self.tokenizer(
            all_texts, padding=True, truncation=True,
            max_length=self.max_length, return_tensors="pt"
        )
        return tokenized_inputs

data_collator = PairCollator(tokenizer, max_length=MAX_SEQ_LENGTH)

def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

class ContrastiveLossTrainer(Trainer):
    def __init__(self, *args, temperature=0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        batch_size = input_ids.size(0) // 2

        outputs = model(input_ids, attention_mask=attention_mask)
        
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_token_pool(last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        q_embeds, p_embeds = embeddings[:batch_size], embeddings[batch_size:]

        scores = torch.matmul(q_embeds, p_embeds.T) / self.temperature
        
        # InfoNCE Loss
        labels = torch.arange(batch_size, device=scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                
                loss = self.compute_loss(model, inputs)
                total_loss += loss.item()
                
                input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
                batch_size = input_ids.size(0) // 2
                
                #outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                outputs = model(input_ids, attention_mask=attention_mask)
                #last_hidden_state = outputs.hidden_states[-1]
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_token_pool(last_hidden_state, attention_mask)
                embeddings = F.normalize(embeddings, p=2, dim=1)
                
                q_embeds, p_embeds = embeddings[:batch_size], embeddings[batch_size:]
                scores = torch.matmul(q_embeds, p_embeds.T) / self.temperature
                
                predicted = torch.argmax(scores, dim=1)
                labels = torch.arange(batch_size, device=scores.device)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += batch_size
                total_samples += batch_size

        eval_runtime = time.time() - start_time
        avg_loss = total_loss / len(dataloader)
        recall_at_1 = correct_predictions / total_predictions if total_predictions > 0 else 0.0
        
        metrics = {
            f"{metric_key_prefix}_loss": avg_loss,
            f"{metric_key_prefix}_recall_at_1": recall_at_1,
            f"{metric_key_prefix}_samples": total_samples,
            f"{metric_key_prefix}_runtime": eval_runtime,
            f"{metric_key_prefix}_samples_per_second": total_samples / eval_runtime if eval_runtime > 0 else 0.0,
            f"{metric_key_prefix}_steps_per_second": len(dataloader) / eval_runtime if eval_runtime > 0 else 0.0,
        }
        
        
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=total_samples
        )
# --- 6. Training Arguments & Trainer ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    save_total_limit=1,
    bf16=True, 
    remove_unused_columns=False,
    label_names=[],
    report_to="tensorboard",
    gradient_accumulation_steps=8,
    dataloader_pin_memory=False,  
    
    eval_strategy="epoch",            # Evaluate every N steps (original evaluation_strategy)
    #eval_steps=100,                   # Evaluate every 100 steps
    save_strategy="epoch",            # Keep save strategy consistent with evaluation strategy
    #save_steps=100,                   # Save checkpoint every 100 steps
    load_best_model_at_end=True,      # Load the best model at the end of training
    metric_for_best_model="eval_recall_at_1",  # Use recall@1 as the best metric
    greater_is_better=True,           # Higher recall@1 is better
)

trainer = ContrastiveLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,        # Pass in validation set
    data_collator=data_collator,
    temperature=TEMPERATURE,
)

print("🚀🚀 Start training! 🚀🚀🚀")
trainer.train()

# --- 7. Save Model ---
print("✅ Training completed, saving the final LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 Model successfully saved to {OUTPUT_DIR}")