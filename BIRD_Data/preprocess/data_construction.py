import openai
import json
import os
from sql_metadata import Parser
import re
from tqdm import tqdm
from typing import Optional, List, Dict
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

API_KEY = ""
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
    # Extract content from <Answer> tags
    
    # Use regex to match content between <Answer> and </Answer>
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
    Test the complete data construction process for a single sample.
    """
    print("--- Starting Single Sample Test ---")

    # 1. Load description and training data
    print("\n[Step 1] Loading data files...")
    with open(descriptions_path, 'r', encoding='utf-8') as f:
        descriptions = json.load(f)
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    if sample_index >= len(train_data):
        print(f"Error: Sample index {sample_index} is out of bounds.")
        return

    # Create lookup dictionary
    desc_lookup = {}
    for item in descriptions:
        db_id = item['database']
        table_name = item['table']
        full_text = item['full_text']
        if db_id not in desc_lookup:
            desc_lookup[db_id] = {}
        desc_lookup[db_id][table_name.lower()] = full_text

    # 2. Select a single sample
    item = train_data[sample_index]
    print(f"\n[Step 2] Selected sample #{sample_index}:")
    print(json.dumps(item, indent=2))
    
    db_id = item['db_id']
    #query = item['query']
    query = item['SQL'] #SQL corresponds to answer in BIRD
    question = item['question']

    # 3. Parse SQL to get table names
    print(f"\n[Step 3] Parsing SQL query to get table names...")
    used_tables = Parser(query).tables
    print(f" -> Parsed tables: {used_tables}")
    if not used_tables:
        print(" -> Test failed: Could not parse tables from query.")
        return

    # 4. Get table descriptions (outputs)
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
        
    # 5. Generate synthetic questions (input)
    print("\n[Step 5] Generating synthetic question (input)...")
    try:
        synthetic_q = generate_question(question, used_tables)
        print(f" -> Generated Input (Synthetic Question):\n{synthetic_q}")
    except Exception as e:
        print(f" -> Test failed: Error during question generation: {e}")
        return

    # 6. Combine final format
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
    Build training data for embedding model instruction fine-tuning (revised version).
    """
    # 1. Load table descriptions
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

    # 2. Load Spider training data
    print("Loading Spider training data...")
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # ★★★【Modification Point 1】★★★
    # Clear the file once before the loop starts to ensure each run starts fresh.
    # This prevents data accumulation when running the function multiple times.
    if os.path.exists(output_path):
        open(output_path, 'w').close()

    batch_data = [] # Temporarily store batch data, replacing the original final_dataset
    instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
    total_generated_count = 0

    print(f"Starting data construction for {len(train_data)} items...")
    # Use enumerate to track index for easy debugging
    for i, item in enumerate(tqdm(train_data, desc="Processing data")):
        db_id = item['db_id']
        #query = item['query']
        query = item['SQL'] #SQL corresponds to answer in BIRD
        question = item['question']

        # 3. Parse SQL to get table names
        used_tables = Parser(query).tables
        if not used_tables:
            continue
        
        # 4. Get table descriptions
        outputs = []
        table_lookup_for_db = desc_lookup.get(db_id, {})
        for table in used_tables:
            desc = table_lookup_for_db.get(table.lower())
            if desc:
                outputs.append(desc)

        if not outputs:
            continue

        # 5. Generate synthetic questions
        try:
            synthetic_q = generate_question(question, used_tables)
        except Exception as e:
            print(f"\nError generating synthetic question for '{question}': {e}")
            continue

        # 6. Combine into final format
        data_entry = {
            "instruction": instruction,
            "input": synthetic_q,
            "outputs": outputs
        }
        batch_data.append(data_entry)
        total_generated_count += 1
        
        # ★★★【Modification Point 2】★★★
        # Whenever batch data reaches 500 items, write to file in "append mode"
        if len(batch_data) >= 50:
            with open(output_path, 'a', encoding='utf-8') as f: # Mode changed from 'w' to 'a'
                for entry in batch_data:
                    # Write independent JSON objects line by line
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
            print(f"\nProcessed item {i+1}, saved a batch of {len(batch_data)}. Total generated: {total_generated_count}")
            batch_data = [] # Clear batch list

    # ★★★【Modification Point 3】★★★
    # 7. After loop ends, save remaining samples (if any)
    if batch_data:
        print(f"\nFinished loop. Saving remaining {len(batch_data)} entries.")
        with open(output_path, 'a', encoding='utf-8') as f: # Also use 'a' mode
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
    Wrap generate_question with exponential backoff retry to improve concurrent stability.
    """
    last_error = None
    for attempt in range(max_retries):
        try:
            return generate_question(question, tables)
        except Exception as e:
            last_error = e
            if log_prefix:
                print(f"{log_prefix} generate_question failed on attempt {attempt+1}/{max_retries}: {e}")
            # Exponential backoff wait
            time.sleep(initial_delay * (2 ** attempt))
    # Reached maximum retries and still failed, raise the last exception
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
    Worker thread function: Build data entry for a single sample;
    - If fill_placeholders=True, return placeholder entry on failure/unsatisfied conditions (maintain one-to-one correspondence with input).
    - If False, return None (caller can skip as needed).
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
        # Parse SQL to get table names
        used_tables = Parser(query).tables if query else []
        if debug and log_prefix:
            print(f"{log_prefix} parsed tables: {used_tables}")
        if not used_tables:
            return _placeholder()

        # Get table descriptions
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

        # Generate synthetic questions (with retry)
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
    Concurrent version: Use thread pool to call model generation simultaneously, significantly accelerating overall construction.
    - max_workers: Number of concurrent threads
    - flush_every: How many entries to accumulate before writing to file
    """
    # 1. Load table description file
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

    # 2. Load training data
    print("Loading training data...")
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 3. Clear output file
    if os.path.exists(output_path):
        open(output_path, 'w').close()

    instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
    buffer = []
    total_generated_count = 0
    success_count = 0

    print(f"Starting concurrent construction over {len(train_data)} items with {max_workers} workers...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Use map to produce results in input order, ensuring output order matches input order
        results_iter = executor.map(
            lambda it: _build_entry_for_item(it, desc_lookup, instruction, fill_placeholders),
            train_data,
        )

        for result in tqdm(results_iter, total=len(train_data), desc="Concurrent processing"):
            # If fill_placeholders=False, may return None; if True, always returns placeholder or real result
            if result is None:
                continue
            buffer.append(result)
            total_generated_count += 1
            if result.get("outputs"):
                success_count += 1

            # Batch write (maintain order, since iteration order matches input order)
            if len(buffer) >= flush_every:
                with open(output_path, 'a', encoding='utf-8') as f:
                    for entry in buffer:
                        f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                buffer = []

    # Write remaining
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
    Single-threaded: Read saved JSON Lines file, locate placeholder entries and attempt to regenerate, keeping order unchanged.
    Directly overwrite original file (write to temporary file then atomically replace).
    """
    # 1) Load table descriptions
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

    # 2) Load training data (for retrieving original samples by index)
    with open(spider_train_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)

    # 3) Read existing output (JSON Lines)
    lines: List[Dict] = []
    with open(output_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                lines.append(json.loads(line))
            except Exception:
                # Skip lines that cannot be parsed
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

    # 4) Iterate and attempt to rebuild placeholders
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

    # 5) Overwrite back (atomic replacement)
    tmp_path = output_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        for obj in lines:
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')
    os.replace(tmp_path, output_path)

    print(f"Placeholders found: {placeholders_found}")
    print(f"Regenerated: {regenerated}")
    print(f"Still failed: {still_failed}")

if __name__ == '__main__':
    # Define file paths
    # Assume this script is located in the preprocess/ directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    #spider_data_dir = os.path.join(base_dir, '..', 'spider_data')
    bird_data_dir = os.path.join(base_dir, '..', 'data')
    bird_train_path = os.path.join(bird_data_dir, "train/train.json")
    #spider_train_path = os.path.join(spider_data_dir, "train_spider.json")
    descriptions_path = os.path.join(bird_data_dir, "merged_table_descriptions.json")
    output_path = os.path.join(bird_data_dir, "embedding_training_data.json")

    # Check if input files exist
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