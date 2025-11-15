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

# --- 1. 参数配置 (Parameters) ---
# ##################################################
# ##           请根据你的环境和需求修改以下参数         ##
# ##################################################
MODEL_NAME = "/root/autodl-tmp/model_file/Qwen3-Embedding-4B"
DATA_FILE = '/root/autodl-tmp/schlink/BIRD_Data/data/embedding_training_data.json'  # 你的数据集文件
#DATA_FILE = '/home/yfwang/wyy/schema_routing/data/spider/spider_data/toy_train.json'
OUTPUT_DIR = "/root/autodl-tmp/model_file/Qwen3-Embedding-4B-lora"
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
BATCH_SIZE = 16      # 4B 模型和更长的序列需要更多显存，从一个较小的值开始尝试
MAX_SEQ_LENGTH = 512 # 根据你的数据长度和显存调整
LORA_RANK = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0.1
TEMPERATURE = 0.05
# ##################################################

# --- 2. 数据准备 (Data Preparation) ---
def process_data_generator(file_path):
    """读取并处理 JSONL 数据，生成 (query, positive_passage) 对"""
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # 跳过空行
                data = json.loads(line)
                query = f"Instruction: {data['instruction']}\nQuery: {data['input']}".strip()
                for pos_passage in data["outputs"]:
                    yield {"query": query, "pos": pos_passage}

print("🚀 正在加载和处理数据集...")
train_dataset = Dataset.from_generator(process_data_generator, gen_kwargs={"file_path": DATA_FILE})
print(f"✅ 数据集加载完毕，总共 {len(train_dataset)} 条训练样本。")
print("样本示例:", train_dataset[0])

# --- 打乱数据 ---
print("🔀 正在打乱数据集...")
train_dataset = train_dataset.shuffle(seed=42)
print("✅ 数据集已打乱。")

# --- 分割数据集为训练集和验证集 ---
print("🔪 正在分割数据集...")
split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']
print(f"✅ 数据集分割完成: {len(train_dataset)}条训练样本, {len(eval_dataset)}条验证样本。")

print("-" * 50)


# --- 3. 加载模型和 Tokenizer (Load Model & Tokenizer) ---
print("🚀 正在加载模型和 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# ‼️ GPU 运行关键步骤:
# 使用 bfloat16 加载模型以节省显存。需要 NVIDIA Ampere 或更新的 GPU。
# 如果 GPU 不支持 bfloat16，可以将其改为 torch.float16。
model = AutoModel.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16  # <-- 优化显存和速度
)

# 设置 pad_token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
print("✅ 模型和 Tokenizer 加载完毕。")
print("-" * 50)


# --- 4. LoRA 配置 (LoRA Configuration) ---
print("🚀 正在配置 LoRA...")
# ‼️ 根据你提供的模型结构，更新 target_modules
peft_config = LoraConfig(
    task_type=TaskType.FEATURE_EXTRACTION,
    inference_mode=False,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    # 🎯 更新：将 LoRA 应用于 Attention 和 MLP 的所有线性层
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, peft_config)
print("✅ LoRA 配置完成。可训练参数如下:")
model.print_trainable_parameters()
print("-" * 50)


# --- 5. 自定义数据整理器和训练器 (Custom Collator & Trainer) ---
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
        # inputs 来自 PairCollator，`Trainer` 会自动将其移动到正确的设备 (GPU)
        input_ids, attention_mask = inputs['input_ids'], inputs['attention_mask']
        batch_size = input_ids.size(0) // 2

        # 模型前向传播
        #outputs = model(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        outputs = model(input_ids, attention_mask=attention_mask)
        
        # 使用正确的pooling策略
        #last_hidden_state = outputs.hidden_states[-1]
        last_hidden_state = outputs.last_hidden_state
        embeddings = last_token_pool(last_hidden_state, attention_mask)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        # 分离 query 和 positive embeddings
        q_embeds, p_embeds = embeddings[:batch_size], embeddings[batch_size:]

        # 计算余弦相似度
        scores = torch.matmul(q_embeds, p_embeds.T) / self.temperature
        
        # InfoNCE Loss
        labels = torch.arange(batch_size, device=scores.device)
        loss = F.cross_entropy(scores, labels)
        
        return (loss, outputs) if return_outputs else loss

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        自定义评估循环，添加更有意义的指标
        """
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        total_predictions = 0
        
        start_time = time.time()
        
        for step, inputs in enumerate(dataloader):
            with torch.no_grad():
                # 计算损失
                loss = self.compute_loss(model, inputs)
                total_loss += loss.item()
                
                # 计算准确率（对角线元素应该是最大的）
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
                
                # 计算recall@1 (对角线元素是否是每行最大值)
                predicted = torch.argmax(scores, dim=1)
                labels = torch.arange(batch_size, device=scores.device)
                correct_predictions += (predicted == labels).sum().item()
                total_predictions += batch_size
                total_samples += batch_size

        # 计算平均指标
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
        
        # 创建一个具有 .metrics 属性的对象
        
        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=total_samples
        )


# --- 6. 训练参数和启动 (Training Arguments & Start) ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=20,
    save_total_limit=1,
    # ‼️ GPU 运行关键步骤: 开启 bf16 混合精度训练
    bf16=True, # 如果 GPU 不支持 bf16, 改为 fp16=True
    remove_unused_columns=False,
    label_names=[],
    report_to="tensorboard",
    gradient_accumulation_steps=8,
    dataloader_pin_memory=False,  # 节省显存
    # --- 新增评估配置 ---
    eval_strategy="epoch",            # 每N步评估一次 (原 evaluation_strategy)
    #eval_steps=100,                   # 每100步评估一次
    save_strategy="epoch",            # 保存策略与评估策略保持一致
    #save_steps=100,                   # 每100步保存一次检查点
    load_best_model_at_end=True,      # 训练结束后加载最优模型
    metric_for_best_model="eval_recall_at_1",  # 使用 recall@1 作为最优标准
    greater_is_better=True,           # recall@1 越大越好
)

trainer = ContrastiveLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,        # 传入验证集
    data_collator=data_collator,
    temperature=TEMPERATURE,
)

print("🚀🚀 开始训练！🚀🚀🚀")
trainer.train()

# --- 7. 保存模型 (Save Model) ---
print("✅ 训练完成，正在保存最终的 LoRA 适配器...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"🎉 模型已成功保存到 {OUTPUT_DIR}")