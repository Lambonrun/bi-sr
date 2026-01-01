import os
import re
import json
import torch
import json
import re
import asyncio
import aiohttp
import time
import openai
from typing import List, Dict, Tuple, Set

API_KEY = "API"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# Concurrency control
CONCURRENT_REQUESTS = 20  # You can modify this value to control concurrency (recommended 5-15)
# --- Configuration ---
# Root directory of the database where schema.sql files are located
# NO USE HERE
BASE_DB_PATH = "/data/spider/spider_data/database"
# tables.json file path
TABLES_JSON_PATH = "\BIRD_Data\dev_20240627\dev_tables.json"

BASE_INFO_PATH = "\BIRD_Data\data_cache\dev\dev_predicted_tables_finetuned.json"

# --- Data loading and caching (this part has not changed) ---
JSON_SCHEMA_CACHE = None
def load_json_data(json_path: str) -> dict:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {db['db_id']: db for db in data}
    except Exception as e:
        print(f"Warning: Error loading or processing JSON file '{json_path}': {e}")
        return {}

def get_json_schema_data():
    global JSON_SCHEMA_CACHE
    if JSON_SCHEMA_CACHE is None:
        JSON_SCHEMA_CACHE = load_json_data(TABLES_JSON_PATH)
    return JSON_SCHEMA_CACHE

# --- Helper functions ---

def _parse_input_string(input_str: str) -> tuple[str | None, str | None]:
    """Parse input string."""
    try:
        db_match = re.search(r'database:\s*([^,]+)', input_str)
        table_match = re.search(r'table:\s*([^,]+)', input_str)
        if not db_match or not table_match: return None, None
        return db_match.group(1).strip(), table_match.group(1).strip()
    except Exception:
        return None, None

def _extract_raw_create_table_from_sql(database_name: str, table_name: str) -> str:
    """[Strategy 1] Try to extract CREATE TABLE statements from .sql files."""
    schema_file_path = os.path.join(BASE_DB_PATH, database_name, "schema.sql")
    if not os.path.exists(schema_file_path):
        return f"Error: SQL file not found -> {schema_file_path}"
    
    try:
        with open(schema_file_path, 'r', encoding='utf-8') as f: content = f.read()
        pattern = re.compile(f'CREATE TABLE\s+["`]?{table_name}["`]?\s*\(.*?\);', re.DOTALL | re.IGNORECASE)
        match = pattern.search(content)
        if match: return match.group(0)
        else: return f"Error: CREATE TABLE statement for table '{table_name}' not found in {schema_file_path}."
    except Exception as e:
        return f"Error reading or processing SQL file {schema_file_path}: {e}"

def _reconstruct_create_table_from_json(db_info: dict, table_name: str) -> str:
    """[Strategy 2] Reconstruct CREATE TABLE statement based only on JSON data."""
    try:
        table_index = db_info['table_names_original'].index(table_name)
    except (ValueError, KeyError):
        return f"Error: Table '{table_name}' not found in JSON information for database '{db_info.get('db_id')}'."

    original_table_name = db_info['table_names_original'][table_index]
    
    column_definitions = []
    table_column_indices = [i for i, col in enumerate(db_info['column_names']) if col[0] == table_index]
    pk_column_indices_in_table = [idx for idx in db_info.get('primary_keys', []) if idx in table_column_indices]

    for col_idx in table_column_indices:
        clean_col_name = db_info['column_names'][col_idx][1]
        original_col_name = db_info['column_names_original'][col_idx][1]
        col_type = db_info['column_types'][col_idx].upper()
        
        col_def = f'"{original_col_name}" {col_type} -- {clean_col_name}'
        if len(pk_column_indices_in_table) == 1 and col_idx in pk_column_indices_in_table:
            col_def += " PRIMARY KEY"
        column_definitions.append(col_def)

    if len(pk_column_indices_in_table) > 1:
        pk_col_names = [f'"{db_info["column_names_original"][i][1]}"' for i in pk_column_indices_in_table]
        column_definitions.append(f'PRIMARY KEY ({", ".join(pk_col_names)})')

    for fk_pair in db_info.get('foreign_keys', []):
        col1_idx, col2_idx = fk_pair
        source_col_idx, ref_col_idx = (None, None)
        if col1_idx in table_column_indices: source_col_idx, ref_col_idx = col1_idx, col2_idx
        elif col2_idx in table_column_indices: source_col_idx, ref_col_idx = col2_idx, col1_idx
        
        if source_col_idx:
            source_col_name = db_info['column_names_original'][source_col_idx][1]
            ref_col_name = db_info['column_names_original'][ref_col_idx][1]
            ref_table_index = db_info['column_names_original'][ref_col_idx][0]
            ref_table_name = db_info['table_names_original'][ref_table_index]
            column_definitions.append(f'FOREIGN KEY ("{source_col_name}") REFERENCES "{ref_table_name}"("{ref_col_name}")')
    
    # 1. First store the result of the join operation in a variable
    formatted_columns = ",\n    ".join(column_definitions)
    
    # 2. Then use this variable without backslashes in the f-string
    return (f'CREATE TABLE "{original_table_name}" (\n'
            f'    {formatted_columns}\n'
            f');')


def get_formatted_schema_robust(item_string: str) -> str:
    """
    Robust schema extraction function that prioritizes SQL files and falls back to JSON reconstruction on failure.
    """
    db_id, table_name = _parse_input_string(item_string)
    if not db_id or not table_name:
        return f"Error: Unable to parse input '{item_string}'."

    # --- Strategy 1: Try to extract and enhance from SQL file ---
    raw_create_table = _extract_raw_create_table_from_sql(db_id, table_name)
    
    if not raw_create_table.startswith("Error:"):
        json_data = get_json_schema_data()
        name_mapping = {}
        if json_data and db_id in json_data:
            db_info = json_data[db_id]
            try:
                table_index = db_info['table_names'].index(table_name)
                for i, col_orig in enumerate(db_info['column_names_original']):
                    if col_orig[0] == table_index:
                        name_mapping[col_orig[1]] = db_info['column_names'][i][1]
            except (ValueError, KeyError, IndexError): pass

        enhanced_create_table = raw_create_table
        if name_mapping:
            modified_lines = []
            pattern = re.compile(r'^\s*["`]?(\w+)["`]?')
            for line in raw_create_table.split('\n'):
                match = pattern.match(line)
                if match and line.strip().upper().split()[0] not in ('PRIMARY', 'FOREIGN', 'CREATE'):
                    original_name = match.group(1)
                    if original_name in name_mapping:
                        clean_line = line.rstrip().rstrip(',')
                        has_comma = line.rstrip().endswith(',')
                        line = f"{clean_line} -- {name_mapping[original_name]}{',' if has_comma else ''}"
                modified_lines.append(line)
            enhanced_create_table = "\n".join(modified_lines)
        
        return (f"Database: {db_id} Table: {table_name}\n"
                f"Table information:\n"
                f"{enhanced_create_table}")

    # --- Strategy 2: If Strategy 1 fails, fall back to reconstruction from JSON ---
    else:
        # print(f"Info: '{table_name}' not found in SQL file, trying to reconstruct from JSON...") # Debug prompt
        json_data = get_json_schema_data()
        if not json_data or db_id not in json_data:
            return f"Error: Database '{db_id}' not found in JSON data either. Both strategies failed."
        
        db_info = json_data[db_id]
        reconstructed_sql = _reconstruct_create_table_from_json(db_info, table_name)
        
        if reconstructed_sql.startswith("Error:"):
            return reconstructed_sql # If JSON reconstruction also fails, return its error message
            
        return (f"Database: {db_id} Table: {table_name}\n"
                f"Table information:\n"
                f"{reconstructed_sql}")

def get_formatted_schema_json(item_string: str) -> str:
    db_id, table_name = _parse_input_string(item_string)
    if not db_id or not table_name:
        return f"Error: Unable to parse input '{item_string}'."
    
    json_data = get_json_schema_data()
    if not json_data or db_id not in json_data:
        return f"Error: Database '{db_id}' not found in JSON data either. Both strategies failed."
    
    db_info = json_data[db_id]
    reconstructed_sql = _reconstruct_create_table_from_json(db_info, table_name)
    
    if reconstructed_sql.startswith("Error:"):
        return reconstructed_sql # If JSON reconstruction also fails, return its error message
        
    return (f"Database: {db_id} Table: {table_name}\n"
            f"Table information:\n"
            f"{reconstructed_sql}")


def load_schema_construction_data(file_path: str):
    questions = []
    answers = []
    tables = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions_raw = data['questions']
            for item in questions_raw:
                pattern = re.compile(r"Query: (.*?)\n")
                match = pattern.search(item)
                if match:
                    query_content = match.group(1)
                    questions.append(query_content)
                else:
                    print(f"Warning: Could not parse the question: {item}")
                    questions.append(item)
            print("questions length: ", len(questions))
            answers = data['answers']
            print("answers length: ", len(answers))
            tables_raw = data['predicted_tables']
            print("tables_raw length: ", len(tables_raw))
            for i, item in enumerate(tables_raw):
                schemas = []
                lst = item[:5]
                for tble in lst:
                    tble_schema = get_formatted_schema_robust(tble)
                    schemas.append(tble_schema)
                tables.append(schemas)
                if i % 200 == 0:
                    print(f"Processed {i} tables")
            print("tables length: ", len(tables))

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
    return questions, answers, tables

def _parse_highlight_instructions(instructions_string: str) -> Dict[Tuple[str, str], Set[str]]:
    """
    (Internal helper function) Parse a single string containing multiple highlight instructions.
    Returns a dictionary with keys as (database, table) tuples and values as sets of column names to highlight.
    """
    highlights = {}
    pattern = re.compile(r"Database:\s*(.*?)\s*Table:\s*(.*?)\s*Column:\s*(.*)", re.IGNORECASE)
    
    # Split the incoming single string into multiple lines by newline
    lines = instructions_string.split('\n')
    
    for instruction in lines:
        # Remove possible leading/trailing spaces from each line (especially spaces before -)
        clean_instruction = instruction.strip()
        if not clean_instruction:
            continue # Skip empty lines

        match = pattern.search(clean_instruction)
        if match:
            db_name = match.group(1).strip()
            table_name = match.group(2).strip()
            # Split strings like "col1, col2" into sets {'col1', 'col2'}
            columns_to_highlight = {col.strip() for col in match.group(3).split(',')}
            
            highlights[(db_name, table_name)] = columns_to_highlight
            
    return highlights

def add_important_markers(
    statement_list: List[str], 
    instructions_string: str 
) -> List[str]:
    """
    Add "IMPORTANT" markers to column comments in CREATE TABLE statements according to instructions.

    :param statement_list: List of strings containing CREATE TABLE statements generated by previous functions.
    :param instructions_string: Single string containing all highlight instructions.
    :return: New list containing modified statements.
    """
    # 1. Parse highlight instruction string
    highlights_map = _parse_highlight_instructions(instructions_string)
    if not highlights_map:
        print("Warning: No valid highlight instructions parsed.")
        return statement_list

    modified_statements = []
    
    header_pattern = re.compile(r"Database:\s*(.*?)\s*Table:\s*(.*)", re.IGNORECASE)
    col_pattern = re.compile(r'^\s*["`]?(\w+)["`]?')

    # 2. Iterate through each statement to be processed
    for statement in statement_list:
        header_line = statement.split('\n', 1)[0]
        header_match = header_pattern.search(header_line)
        
        if not header_match:
            modified_statements.append(statement)
            continue
            
        db_name = header_match.group(1).strip()
        table_name = header_match.group(2).strip()
        
        columns_to_highlight = highlights_map.get((db_name, table_name))
        if not columns_to_highlight:
            modified_statements.append(statement)
            continue

        # 4. If needed, modify line by line
        original_lines = statement.split('\n')
        new_lines = []
        for line in original_lines:
            col_match = col_pattern.match(line)
            if col_match and col_match.group(1) in columns_to_highlight:
                modified_line = re.sub(r'--', r'-- IMPORTANT', line, count=1)
                new_lines.append(modified_line)
            else:
                new_lines.append(line)
        
        modified_statements.append("\n".join(new_lines))
        
    return modified_statements

async def add_important_markers_async(
    statement_list: List[str], 
    instructions_string: str 
) -> List[str]:
    """
    Asynchronous version: Add "IMPORTANT" markers to column comments in CREATE TABLE statements according to instructions.
    
    :param statement_list: List of strings containing CREATE TABLE statements generated by previous functions.
    :param instructions_string: Single string containing all highlight instructions.
    :return: New list containing modified statements.
    """
    # 1. Parse highlight instruction string
    highlights_map = _parse_highlight_instructions(instructions_string)
    if not highlights_map:
        print("Warning: No valid highlight instructions parsed.")
        return statement_list

    modified_statements = []
    
    header_pattern = re.compile(r"Database:\s*(.*?)\s*Table:\s*(.*)", re.IGNORECASE)
    col_pattern = re.compile(r'^\s*["`]?(\w+)["`]?')

    # 2. Iterate through each statement to be processed
    for statement in statement_list:
        header_line = statement.split('\n', 1)[0]
        header_match = header_pattern.search(header_line)
        
        if not header_match:
            modified_statements.append(statement)
            continue
            
        db_name = header_match.group(1).strip()
        table_name = header_match.group(2).strip()
        
        columns_to_highlight = highlights_map.get((db_name, table_name))
        if not columns_to_highlight:
            modified_statements.append(statement)
            continue

        # 4. If needed, modify line by line
        original_lines = statement.split('\n')
        new_lines = []
        for line in original_lines:
            col_match = col_pattern.match(line)
            if col_match and col_match.group(1) in columns_to_highlight:
                modified_line = re.sub(r'--', r'-- IMPORTANT', line, count=1)
                new_lines.append(modified_line)
            else:
                new_lines.append(line)
        
        modified_statements.append("\n".join(new_lines))
        
        # Yield control periodically to avoid blocking the event loop
        if len(modified_statements) % 10 == 0:
            await asyncio.sleep(0)
        
    return modified_statements

def _get_completion(prompt: str, system_prompt: str = "You are a helpful assistant."):
    message = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=8192,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    return message.choices[0].message.content

async def get_completion_async(session, prompt: str, system_prompt: str = "You are a helpful assistant."):
    """Asynchronous version of API call"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "max_tokens": 8192,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    }
    
    async with session.post(f"{BASE_URL}/chat/completions", json=data, headers=headers) as response:
        if response.status == 200:
            result = await response.json()
            return result['choices'][0]['message']['content']
        else:
            error_text = await response.text()
            print(f"API request failed: {response.status}, {error_text}")
            return None

async def column_filtering_async(session, table: str, question: str):
    """Asynchronous version of column filtering"""
    sys_prompt = """
    You are a helpful assistant in database domain.
    """
    prompt = f"""
    You are given a question and several table schemas. You need to filter out the columns that are especially important to answer the question.
    each table schema includes database name, table name, and a CREATE TABLE statement with column names, types and additional column comments. You need to filter out the columns important to answer the question, and output the table schema with the important columns.

    <Example>
    Question: How many singers do we have?
    Table schema:
    ['Database: singer Table: singer\nTable information:\nCREATE TABLE "singer" (\n"Singer_ID" int -- singer id,\n"Name" text -- name,\n"Birth_Year" real -- birth year,\n"Net_Worth_Millions" real -- net worth millions,\n"Citizenship" text -- citizenship,\nPRIMARY KEY ("Singer_ID")\n);', 'Database: concert_singer Table: singer\nTable information:\nCREATE TABLE "singer" (\n"Singer_ID" int -- singer id,\n"Name" text -- name,\n"Country" text -- country,\n"Song_Name" text -- song name,\n"Song_release_year" text -- song release year,\n"Age" int -- age,\n"Is_male" bool -- is male,\nPRIMARY KEY ("Singer_ID")\n);]
    Answer:
    - Database: singer Table: singer Column: Singer_ID
    - Database: concert_singer Table: singer Column: Singer_ID, Name
    </Example>

    Question: {question}
    Table schema: {table}
    You can think step by step, and output the answer in the following format:
    <Reasoning>
    
    </Reasoning>
    <Answer>
    
    </Answer>
    """
    answer = await get_completion_async(session, prompt, system_prompt=sys_prompt)
    
    if answer is None:
        return table  # If API call fails, return original table
    
    filtered_answer = answer.split("<Answer>")[1].split("</Answer>")[0].strip() if "<Answer>" in answer else answer
    return filtered_answer


async def process_batch_schema(session, batch_data, batch_index):
    """Process a batch of data"""
    print(f"Starting to process batch {batch_index + 1}, containing {len(batch_data)} questions")
    
    # Create all tasks for this batch
    tasks = []
    for item in batch_data:
        question, answer, table = item
        task = column_filtering_async(session, table, question)
        tasks.append(task)
    
    # Execute all tasks in this batch concurrently
    results = await asyncio.gather(*tasks)
    
    # Process results, maintain order
    processed_items = []
    for i, (item, schema) in enumerate(zip(batch_data, results)):
        question, answer, table = item
        processed_items.append({"question": question, "answer": answer, "original_schema": table, "chosen_schema": schema})
    
    print(f"Completed batch {batch_index + 1}")
    return processed_items

async def load_schema_construction_data_async(file_path: str):
    """Asynchronous version of data loading and processing"""
    questions = []
    answers = []
    tables = []
    final_query = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions_raw = data['questions']
            for item in questions_raw:
                pattern = re.compile(r"Query: (.*?)\n")
                match = pattern.search(item)
                if match:
                    query_content = match.group(1)
                    questions.append(query_content)
                else:
                    print(f"Warning: Could not parse the question: {item}")
                    questions.append(item)
            
            print("questions length: ", len(questions))
            answers = data['answers']
            print("answers length: ", len(answers))
            tables_raw = data['predicted_tables']
            print("tables_raw length: ", len(tables_raw))
            
            # Build table structure data
            for i, item in enumerate(tables_raw):
                schemas = []
                lst = item[:5]
                for tble in lst:
                    tble_schema = get_formatted_schema_json(tble)
                    #if "Error" in tble_schema:
                        #print(f"Error in {i} table schema: {tble_schema}")
                    schemas.append(tble_schema)
                tables.append(schemas)
                if i % 200 == 0:
                    print(f"Processed {i} tables")
            print("tables length: ", len(tables))
            
            # Prepare data for concurrent processing
            batch_data = []
            for question, answer, table in zip(questions, answers, tables):
                batch_data.append((question, answer, table))
            
            print(f"Total {len(batch_data)} questions to process")
            
            # Divide data into batches, each with CONCURRENT_REQUESTS items
            batch_size = CONCURRENT_REQUESTS
            batches = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
            print(f"Using {CONCURRENT_REQUESTS} concurrent requests, divided into {len(batches)} batches")
            
            # Create aiohttp session
            connector = aiohttp.TCPConnector(limit=50)  # Limit connection count
            timeout = aiohttp.ClientTimeout(total=600)  # Set timeout
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                start_time = time.time()
                
                # Process batch by batch to maintain order
                for batch_index, batch in enumerate(batches):
                    batch_results = await process_batch_schema(session, batch, batch_index)
                    
                    # Add batch results to final results in order
                    final_query.extend(batch_results)
                    
                    # Display progress
                    processed_count = len(final_query)
                    total_count = len(batch_data)
                    elapsed_time = time.time() - start_time
                    avg_time_per_item = elapsed_time / processed_count if processed_count > 0 else 0
                    estimated_total_time = avg_time_per_item * total_count
                    remaining_time = estimated_total_time - elapsed_time
                    
                    print(f"Progress: {processed_count}/{total_count} "
                          f"({processed_count/total_count*100:.1f}%) "
                          f"Elapsed: {elapsed_time:.1f}s "
                          f"Estimated remaining: {remaining_time:.1f}s")
                    
                    # Take a short break between batches to avoid too frequent requests
                    if batch_index < len(batches) - 1:  # Not the last batch
                        await asyncio.sleep(0.5)
                
                total_time = time.time() - start_time
                print(f"All processing completed, total time: {total_time:.1f}s")

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
    return questions, answers, tables, final_query

# Keep the original synchronous function as backup
def load_schema_construction_data(file_path: str):
    """Original synchronous data loading function (backup)"""
    questions = []
    answers = []
    tables = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions_raw = data['questions']
            for item in questions_raw:
                pattern = re.compile(r"Query: (.*?)\n")
                match = pattern.search(item)
                if match:
                    query_content = match.group(1)
                    questions.append(query_content)
                else:
                    print(f"Warning: Could not parse the question: {item}")
                    questions.append(item)
            print("questions length: ", len(questions))
            answers = data['answers']
            print("answers length: ", len(answers))
            tables_raw = data['predicted_tables']
            print("tables_raw length: ", len(tables_raw))
            for i, item in enumerate(tables_raw):
                schemas = []
                lst = item[:5]
                for tble in lst:
                    tble_schema = get_formatted_schema_robust(tble)
                    schemas.append(tble_schema)
                tables.append(schemas)
                if i % 200 == 0:
                    print(f"Processed {i} tables")
            print("tables length: ", len(tables))

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
    return questions, answers, tables

def column_filtering_sync(table: str, question: str):
    """Original synchronous column filtering function (backup)"""
    sys_prompt = """
    You are a helpful assistant in database domain.
    """
    prompt = f"""
    You are given a question and several table schemas. You need to filter out the columns that are especially important to answer the question.
    each table schema includes database name, table name, and a CREATE TABLE statement with column names, types and additional column comments. You need to filter out the columns important to answer the question, and add 'IMPORTANT' before the column comment.

    <Example>
    Question: How many singers do we have?
    Table schema:
    ['Database: singer Table: singer\nTable information:\nCREATE TABLE "singer" (\n"Singer_ID" int -- singer id,\n"Name" text -- name,\n"Birth_Year" real -- birth year,\n"Net_Worth_Millions" real -- net worth millions,\n"Citizenship" text -- citizenship,\nPRIMARY KEY ("Singer_ID")\n);', 'Database: concert_singer Table: singer\nTable information:\nCREATE TABLE "singer" (\n"Singer_ID" int -- singer id,\n"Name" text -- name,\n"Country" text -- country,\n"Song_Name" text -- song name,\n"Song_release_year" text -- song release year,\n"Age" int -- age,\n"Is_male" bool -- is male,\nPRIMARY KEY ("Singer_ID")\n);'T
    )]
    Answer:
    ['Database: singer Table: singer\nTable information:\nCREATE TABLE "singer" (\n"Singer_ID" int --IMPORTANT singer id,\n"Name" text -- name,\n"Birth_Year" real -- birth year,\n"Net_Worth_Millions" real -- net worth millions,\n"Citizenship" text -- citizenship,\nPRIMARY KEY ("Singer_ID")\n);', 'Database: concert_singer Table: singer\nTable information:\nCREATE TABLE "singer" (\n"Singer_ID" int --IMPORTANT singer id,\n"Name" text --IMPORTANT name,\n"Country" text -- country,\n"Song_Name" text -- song name,\n"Song_release_year" text -- song release year,\n"Age" int -- age,\n"Is_male" bool -- is male,\nPRIMARY KEY ("Singer_ID")\n);']
    </Example>

    Question: {question}
    Table schema: {table}
    You can think step by step, and output the answer in the following format:
    <Reasoning>
    
    </Reasoning>
    <Answer>
    
    </Answer>
    """
    answer = _get_completion(prompt, system_prompt=sys_prompt)
    filtered_answer = answer.split("<Answer>")[1].split("</Answer>")[0].strip() if "<Answer>" in answer else answer
    return filtered_answer

if __name__ == '__main__':
    # Use asynchronous version
    print("Processing data using asynchronous concurrent version...")
    questions, answers, tables, final_query = asyncio.run(load_schema_construction_data_async(BASE_INFO_PATH))
    
    print(f"Processing completed:")
    print(f"questions length: {len(questions)}")
    print(f"answers length: {len(answers)}")
    print(f"tables length: {len(tables)}")
    print(f"final_query length: {len(final_query)}")
    
    # Save results
    with open("\BIRD_Data\data_cache\dev\dev_final_query_column_filtering_BIRD.json", "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers, "tables": tables, "final_query": final_query}, f, ensure_ascii=False, indent=4)
    
    print("Data saving completed!")
    