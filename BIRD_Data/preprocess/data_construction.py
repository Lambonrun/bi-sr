import openai
import json
import os
from sql_metadata import Parser
import re
from tqdm import tqdm
from typing import Optional, List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = "sk-9fe9b714a9ad4b6ab83bf7a13ead42ec"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

def get_completion(prompt: str, system_prompt: str = "You are a helpful assistant."):
    message = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=8192,
        temperature=0.7,
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
        ]
    )
    #print("token usage: ", message.usage.total_tokens)
    return message.choices[0].message.content

def generate_question(question: str, tables: str):
    sys_prompt = """
    You are a helpful assistant in database domain.
    """
    prompt = f"""
    Your job is to align the question with the tables given. You need to describe tables needed to answer the question, and align the table description with the table name, with other potential tables names.

    <Example>
    Question: What is the name of the document with the most number of sections?
    Tables: ['documents', 'document_sections']
    Answer: 
    - documents: Stores document metadata and access information, potential names:[Document, Documents, Document_Metadata, Document_Access_Information]
    - document_sections: Document sections with sequence and titles, potential names:[Section, Sections, Section_Titles, Section_Sequences]
    </Example>

    Question: {question}
    Tables: {tables}
    
    You can think step by step, and output the answer in the following format:
    <Reasoning>
    - ...
    </Reasoning>
    <Answer>
    - table_name: table_description, potential names:[table_name_1, table_name_2, ...]
    - table_name: table_description, potential names:[table_name_1, table_name_2, ...]
    - ...
    </Answer>
    """
    answer = get_completion(prompt, system_prompt=sys_prompt)
    # 提取<Answer>标签中的内容
    
    # 使用正则表达式匹配<Answer>和</Answer>之间的内容
    answer_pattern = r'<Answer>(.*?)</Answer>'
    answer_match = re.search(answer_pattern, answer, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
        
    else:
        print(f"Warning: Could not find <Answer> tags in response: {answer}")
        answer_content = answer
    final_question = question+"\n potential tables:\n"+answer_content
    return final_question

def test_single_sample(spider_train_path: str, descriptions_path: str, sample_index: int = 0):
    """
    测试单个样本的完整数据构建流程。
    """
    print("--- Starting Single Sample Test ---")

    # 1. 加载描述和训练数据
    print("\n[Step 1] Loading data files...")
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    if sample_index >= len(train_data):
        print(f"Error: Sample index {sample_index} is out of bounds.")
        return

    # 创建查找字典
    desc_lookup = {}
    for item in descriptions:
        db_id = item['database']
        table_name = item['table']
        full_text = item['full_text']
        if db_id not in desc_lookup:
            desc_lookup[db_id] = {}
        desc_lookup[db_id][table_name.lower()] = full_text

    # 2. 选择单个样本
    item = train_data[sample_index]
    print(f"\n[Step 2] Selected sample #{sample_index}:")
    print(json.dumps(item, indent=2))
    
    db_id = item['db_id']
    #query = item['query']
    query = item['SQL'] #SQL corresponds to answer in BIRD
    question = item['question']

    # 3. 解析SQL获取表名
    print(f"\n[Step 3] Parsing SQL query to get table names...")
    used_tables = Parser(query).tables
    print(f" -> Parsed tables: {used_tables}")
    if not used_tables:
        print(" -> Test failed: Could not parse tables from query.")
        return

    # 4. 获取表描述 (outputs)
    print("\n[Step 4] Fetching table descriptions (outputs)...")
    outputs = []
    table_lookup_for_db = desc_lookup.get(db_id, {})
    for table in used_tables:
        desc = table_lookup_for_db.get(table.lower())
        if desc:
            outputs.append(desc)
    print(" -> Found descriptions:")
    for out in outputs:
        print(f"    - {out}")

    if not outputs:
        print(" -> Test failed: Could not find descriptions for parsed tables.")
        return
        
    # 5. 生成合成问题 (input)
    print("\n[Step 5] Generating synthetic question (input)...")
    try:
        synthetic_q = generate_question(question, used_tables)
        print(f" -> Generated Input (Synthetic Question):\n{synthetic_q}")
    except Exception as e:
        print(f" -> Test failed: Error during question generation: {e}")
        return

    # 6. 组合最终格式
    print("\n[Step 6] Assembling final data entry...")
    instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
    data_entry = {
        "instruction": instruction,
        "input": synthetic_q,
        "outputs": outputs
    }
    print(" -> Final JSON object:")
    print(json.dumps(data_entry, indent=4, ensure_ascii=False))
    print("\n--- Single Sample Test Finished ---")

def construct_training_data(spider_train_path: str, descriptions_path: str, output_path: str):   
    """ 
    构建用于嵌入模型指令微调的训练数据 (修复版)。
    """
    # 1. 加载表描述文件
    print("Loading table descriptions...")
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    
    desc_lookup = {}
    for item in descriptions:
        db_id = item['database']
        table_name = item['table']
        full_text = item['full_text']
        if db_id not in desc_lookup:
            desc_lookup[db_id] = {}
        desc_lookup[db_id][table_name.lower()] = full_text

    # 2. 加载Spider训练数据
    print("Loading Spider training data...")
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # ★★★【修改点 1】★★★
    # 在循环开始前，先清空一次文件，确保每次运行都是从头开始。
    # 这样可以防止重复运行函数时，数据不断累积。
    if os.path.exists(output_path):
        open(output_path, 'w').close()

    batch_data = [] # 用于暂存一个批次的数据，替换原来的 final_dataset
    instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
    total_generated_count = 0

    print(f"Starting data construction for {len(train_data)} items...")
    # 使用 enumerate 来跟踪索引，方便调试
    for i, item in enumerate(tqdm(train_data, desc="Processing data")):
        db_id = item['db_id']
        #query = item['query']
        query = item['SQL'] #SQL corresponds to answer in BIRD
        question = item['question']

        # 3. 解析SQL，获取表名
        used_tables = Parser(query).tables
        if not used_tables:
            continue
        
        # 4. 获取表描述
        outputs = []
        table_lookup_for_db = desc_lookup.get(db_id, {})
        for table in used_tables:
            desc = table_lookup_for_db.get(table.lower())
            if desc:
                outputs.append(desc)

        if not outputs:
            continue

        # 5. 生成合成问题
        try:
            synthetic_q = generate_question(question, used_tables)
        except Exception as e:
            print(f"\nError generating synthetic question for '{question}': {e}")
            continue

        # 6. 组合成最终格式
        data_entry = {
            "instruction": instruction,
            "input": synthetic_q,
            "outputs": outputs
        }
        batch_data.append(data_entry)
        total_generated_count += 1
        
        # ★★★【修改点 2】★★★
        # 每当批次数据达到500条，就以“追加模式”写入文件
        if len(batch_data) >= 50:
            with open(output_path, 'a', encoding='utf-8') as f: # 模式从 'w' 改为 'a'
                for entry in batch_data:
                    # 逐行写入独立的 JSON 对象
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"\nProcessed item {i+1}, saved a batch of {len(batch_data)}. Total generated: {total_generated_count}")
            batch_data = [] # 清空批次列表

    # ★★★【修改点 3】★★★
    # 7. 循环结束后，保存剩余的样本（如果还有的话）
    if batch_data:
        print(f"\nFinished loop. Saving remaining {len(batch_data)} entries.")
        with open(output_path, 'a', encoding='utf-8') as f: # 同样使用 'a' 模式
            for entry in batch_data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nConstruction complete. Total generated entries: {total_generated_count}")
    print(f"Training data saved as JSON Lines to {output_path}")


def generate_question_with_retry(
    question: str,
    tables,
    max_retries: int = 5,
    initial_delay: float = 1.0,
    log_prefix: str = "",
):
    """
    包装 generate_question，增加指数回退重试，提升并发稳定性。
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return generate_question(question, tables)
        except Exception as e:
            last_error = e
            if log_prefix:
                print(f"{log_prefix} generate_question failed on attempt {attempt+1}/{max_retries}: {e}")
            # 指数回退等待
            time.sleep(initial_delay * (2 ** attempt))
    # 达到最大重试次数仍失败，抛出最后一个异常
    raise last_error


def _build_entry_for_item(
    item,
    desc_lookup,
    instruction: str,
    fill_placeholders: bool = True,
    debug: bool = False,
    log_prefix: str = "",
):
    """
    工作线程函数：为单个样本构建数据条目；
    - 若 fill_placeholders=True，则失败/条件不满足时返回占位符条目（保持与输入一一对应）。
    - 若为 False，则返回 None（调用方可自行跳过）。
    """
    db_id = item.get('db_id')
    query = item.get('SQL')
    question = item.get('question', '')

    def _placeholder():
        if not fill_placeholders:
            return None
        return {
            "instruction": instruction,
            "input": f"[PLACEHOLDER] {question}",
            "outputs": [],
        }

    try:
        # 解析SQL获取表名
        used_tables = Parser(query).tables if query else []
        if debug and log_prefix:
            print(f"{log_prefix} parsed tables: {used_tables}")
        if not used_tables:
            return _placeholder()

        # 获取表描述
        outputs = []
        table_lookup_for_db = desc_lookup.get(db_id, {})
        for table in used_tables:
            desc = table_lookup_for_db.get(table.lower())
            if desc:
                outputs.append(desc)
        if debug and log_prefix:
            print(f"{log_prefix} matched descriptions: {len(outputs)}")
        if not outputs:
            return _placeholder()

        # 生成合成问题（带重试）
        try:
            synthetic_q = generate_question_with_retry(
                question, used_tables, log_prefix=log_prefix
            )
        except Exception as e:
            if debug and log_prefix:
                print(f"{log_prefix} give up after retries: {e}")
            return _placeholder()

        return {
            "instruction": instruction,
            "input": synthetic_q,
            "outputs": outputs,
        }
    except Exception:
        return _placeholder()


def construct_training_data_concurrent(
    spider_train_path: str,
    descriptions_path: str,
    output_path: str,
    max_workers: int = 8,
    flush_every: int = 100,
    fill_placeholders: bool = True,
):
    """
    并发版本：使用线程池同时调用模型生成，显著加速整体构建。
    - max_workers: 并发线程数
    - flush_every: 每累计多少条写入一次文件
    """
    # 1. 加载表描述文件
    print("Loading table descriptions...")
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    desc_lookup = {}
    for item in descriptions:
        db_id = item['database']
        table_name = item['table']
        full_text = item['full_text']
        if db_id not in desc_lookup:
            desc_lookup[db_id] = {}
        desc_lookup[db_id][table_name.lower()] = full_text

    # 2. 加载训练数据
    print("Loading training data...")
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 3. 清空输出文件
    if os.path.exists(output_path):
        open(output_path, 'w').close()

    instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
    buffer = []
    total_generated_count = 0
    success_count = 0

    print(f"Starting concurrent construction over {len(train_data)} items with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 使用 map 按输入顺序产出结果，从而保证输出顺序与输入一致
        results_iter = executor.map(
            lambda it: _build_entry_for_item(it, desc_lookup, instruction, fill_placeholders),
            train_data,
        )

        for result in tqdm(results_iter, total=len(train_data), desc="Concurrent processing"):
            # 若 fill_placeholders=False，可能返回 None；True 时总返回占位符或真实结果
            if result is None:
                continue
            buffer.append(result)
            total_generated_count += 1
            if result.get("outputs"):
                success_count += 1

            # 批量写入（保持顺序，因为迭代顺序与输入一致）
            if len(buffer) >= flush_every:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for entry in buffer:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                buffer = []

    # 写入剩余
    if buffer:
        with open(output_path, 'a', encoding='utf-8') as f:
            for entry in buffer:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\nConcurrent construction complete. Total generated entries (incl. placeholders): {total_generated_count}")
    print(f"Successful entries (non-placeholders): {success_count}")
    print(f"Training data saved as JSON Lines to {output_path}")


def retry_placeholders_in_output(
    spider_train_path: str,
    descriptions_path: str,
    output_path: str,
    max_to_retry: Optional[int] = None,
):
    """
    单线程：读取已保存的 JSON Lines 文件，定位占位符条目并尝试重新生成，保持顺序不变。
    直接覆盖原文件（写入到临时文件后原子替换）。
    """
    # 1) 加载表描述
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)

    desc_lookup = {}
    for item in descriptions:
        db_id = item['database']
        table_name = item['table']
        full_text = item['full_text']
        if db_id not in desc_lookup:
            desc_lookup[db_id] = {}
        desc_lookup[db_id][table_name.lower()] = full_text

    # 2) 加载训练数据（用于按索引取回原始样本）
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 3) 读取现有输出（JSON Lines）
    lines: List[Dict] = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                # 跳过无法解析的行
                continue

    instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."

    def is_placeholder(entry: dict) -> bool:
        if not isinstance(entry, dict):
            return False
        inp = entry.get('input', '')
        outs = entry.get('outputs', None)
        if isinstance(inp, str) and inp.startswith('[PLACEHOLDER]'):
            return True
        if isinstance(outs, list) and len(outs) == 0:
            return True
        return False

    placeholders_found = 0
    regenerated = 0
    still_failed = 0

    # 4) 遍历并尝试重建占位符
    for idx, entry in enumerate(lines):
        if not is_placeholder(entry):
            continue
        placeholders_found += 1
        if max_to_retry is not None and regenerated + still_failed >= max_to_retry:
            break

        if idx >= len(train_data):
            still_failed += 1
            continue

        item = train_data[idx]
        db_id = item.get('db_id')
        log_prefix = f"[retry idx={idx} db={db_id}]"
        print(f"{log_prefix} trying to regenerate placeholder...")
        rebuilt = _build_entry_for_item(
            item,
            desc_lookup,
            instruction,
            fill_placeholders=False,
            debug=True,
            log_prefix=log_prefix,
        )
        if rebuilt is None:
            print(f"{log_prefix} failed to regenerate (still placeholder).")
            still_failed += 1
        else:
            lines[idx] = rebuilt
            regenerated += 1
            print(f"{log_prefix} regenerated successfully.")

    # 5) 覆盖写回（原子替换）
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    os.replace(tmp_path, output_path)

    print(f"Placeholders found: {placeholders_found}")
    print(f"Regenerated: {regenerated}")
    print(f"Still failed: {still_failed}")

if __name__ == '__main__':
    # 定义文件路径
    # 假设此脚本位于 preprocess/ 目录下
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #spider_data_dir = os.path.join(base_dir, '..', 'spider_data')
    bird_data_dir = os.path.join(base_dir, '..', 'data')
    bird_train_path = os.path.join(bird_data_dir, "train/train.json")
    #spider_train_path = os.path.join(spider_data_dir, "train_spider.json")
    descriptions_path = os.path.join(bird_data_dir, "merged_table_descriptions.json")
    output_path = os.path.join(bird_data_dir, "embedding_training_data.json")

    # 检查输入文件是否存在
    if not os.path.exists(bird_train_path) or not os.path.exists(descriptions_path):
        print(f"Error: Make sure '{bird_train_path}' and '{descriptions_path}' exist.")
    else:
        mode = input("Run in (t)est, (f)ull, (c)oncurrent, or (r)etry-placeholders mode? [t/f/c/r]: ")
        if mode.lower() == 't':
            sample_idx_str = input("Enter sample index to test (default: 0): ")
            try:
                sample_idx = int(sample_idx_str)
            except ValueError:
                sample_idx = 0
            test_single_sample(bird_train_path, descriptions_path, sample_index=sample_idx)
        elif mode.lower() == 'f':
            construct_training_data(bird_train_path, descriptions_path, output_path)
        elif mode.lower() == 'c':
            workers_str = input("Max workers? (default: 8): ")
            try:
                workers = int(workers_str)
            except ValueError:
                workers = 8
            construct_training_data_concurrent(bird_train_path, descriptions_path, output_path, max_workers=workers)
        elif mode.lower() == 'r':
            max_retry_str = input("Max placeholders to retry? (empty for all): ")
            max_to_retry = None
            if max_retry_str.strip() != '':
                try:
                    max_to_retry = int(max_retry_str)
                except ValueError:
                    max_to_retry = None
            retry_placeholders_in_output(bird_train_path, descriptions_path, output_path, max_to_retry=max_to_retry)
        else:
            print("Invalid mode selected. Exiting.")