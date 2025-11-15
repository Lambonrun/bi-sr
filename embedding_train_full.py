import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer
from transformers.trainer_utils import EvalLoopOutput
from datasets import Dataset
import json
import time
import os
import requests

# --- 1. 参数配置 (Parameters) ---
# ##################################################
# ##       本脚本为全参数微调，不使用 LoRA/PEFT       ##
# ##################################################
MODEL_NAME = "/root/autodl-tmp/model_file/Qwen3-Embedding-4B"
DATA_FILE = '/root/autodl-tmp/schlink/BIRD_Data/data/embedding_training_data.json'
OUTPUT_DIR = "/root/autodl-tmp/model_file/Qwen3-Embedding-4B-finetuned"
NUM_EPOCHS = 1
LEARNING_RATE = 1e-4
# 全参数微调显存占用更高，可根据显存情况适当调大/调小
BATCH_SIZE = 64
MAX_SEQ_LENGTH = 512
TEMPERATURE = 0.05


# --- 2. 数据准备 (Data Preparation) ---
def process_data_generator(file_path):
    """读取并处理 JSONL 数据，生成 (query, positive_passage) 对"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                query = f"Instruction: {data['instruction']}\nQuery: {data['input']}".strip()
                for pos_passage in data["outputs"]:
                    yield {"query": query, "pos": pos_passage}

print("🚀 正在加载和处理数据集...")
train_dataset = Dataset.from_generator(process_data_generator, gen_kwargs={"file_path": DATA_FILE})
print(f"✅ 数据集加载完毕，总共 {len(train_dataset)} 条训练样本。")
print("样本示例:", train_dataset[0])

print("🔀 正在打乱数据集...")
train_dataset = train_dataset.shuffle(seed=42)
print("✅ 数据集已打乱。")

print("🔪 正在分割数据集...")
split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']
print(f"✅ 数据集分割完成: {len(train_dataset)}条训练样本, {len(eval_dataset)}条验证样本。")
print("-" * 50)


# --- 3. 加载模型和 Tokenizer (Load Model & Tokenizer) ---
print("🚀 正在加载模型和 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# 使用 bfloat16 以节省显存并提升性能；如不支持可改为 torch.float16
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 适当启用梯度检查点以进一步节省显存（可按需关闭）
if hasattr(model, "gradient_checkpointing_enable"):
    model.gradient_checkpointing_enable()

# 确保所有参数可训练（全参数微调）
for param in model.parameters():
    param.requires_grad = True

print("✅ 模型和 Tokenizer 加载完毕。")
print("-" * 50)


# --- 4. 数据整理器与 Trainer (Collator & Trainer) ---
class PairCollator:
    """数据整理器，将 (query, pos) 对进行 tokenize"""
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

        outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        embeddings = last_token_pool(last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        q_embeds, p_embeds = embeddings[:batch_size], embeddings[batch_size:]
        scores = torch.matmul(q_embeds, p_embeds.T) / self.temperature

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

                outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1]
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
            num_samples=total_samples,
        )


# --- 5. 训练参数与启动 (Training Arguments & Start) ---
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
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_recall_at_1",
    greater_is_better=True,
)

trainer = ContrastiveLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    temperature=TEMPERATURE,
)

print("🚀🚀 开始全参数微调训练！🚀🚀🚀")
trainer.train()

# --- 6. 保存模型 (Save Model) ---
print("✅ 训练完成，正在保存最终模型...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 模型已成功保存到 {OUTPUT_DIR}")

# For autodl

headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjU1NzUwMywidXVpZCI6ImEyOGI0MWMyLWNhN2YtNGNjOS05MzljLTk1NGVmODIxZTA5MiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.qfhcZ2nJJPOGp_NGi7f9dlHrGjucywo1c5JFM23IYO4jqVx5M7Mof6zvoQHufO-5ht8jljtUSqeDhJ3KItqMfQ"}
resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                     json={
                         "title": "eg. 来自我的程序",
                         "name": "eg. 我的Embedding训练实验",
                         "content": "训练已完成，即将自动关机"
                     }, headers = headers)
print(resp.content.decode())

#自动关机
os.system("/usr/bin/shutdown")
