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

API_KEY = "sk-9fe9b714a9ad4b6ab83bf7a13ead42ec"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 并发数控制
CONCURRENT_REQUESTS = 50  # 可以修改这个值来控制并发数（建议5-15之间）
# --- 配置 ---
# schema.sql 文件所在的数据库根目录
BASE_DB_PATH = "/home/yfwang/wyy/schema_routing/data/spider/spider_data/database"
# tables.json 文件路径
TABLES_JSON_PATH = "/home/yfwang/wyy/schema_routing/data/spider/spider_data/tables.json"

BASE_INFO_PATH = "/home/yfwang/wyy/schema_routing/data/spider/spider_data/cache/dev_predicted_tables_baseline_Qwen3.json"

# --- 数据加载与缓存 (这部分没有变化) ---
JSON_SCHEMA_CACHE = None
def load_json_data(json_path: str) -> dict:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return {db['db_id']: db for db in data}
    except Exception as e:
        print(f"警告: 加载或处理JSON文件 '{json_path}' 时出错: {e}")
        return {}

def get_json_schema_data():
    global JSON_SCHEMA_CACHE
    if JSON_SCHEMA_CACHE is None:
        JSON_SCHEMA_CACHE = load_json_data(TABLES_JSON_PATH)
    return JSON_SCHEMA_CACHE

# --- 辅助函数 ---

def _parse_input_string(input_str: str) -> tuple[str | None, str | None]:
    """解析输入字符串。"""
    try:
        db_match = re.search(r'database:\s*([^,]+)', input_str)
        table_match = re.search(r'table:\s*([^,]+)', input_str)
        if not db_match or not table_match: return None, None
        return db_match.group(1).strip(), table_match.group(1).strip()
    except Exception:
        return None, None

def _extract_raw_create_table_from_sql(database_name: str, table_name: str) -> str:
    """[策略1] 尝试从.sql文件提取CREATE TABLE语句。"""
    schema_file_path = os.path.join(BASE_DB_PATH, database_name, "schema.sql")
    if not os.path.exists(schema_file_path):
        return f"错误: SQL文件未找到 -> {schema_file_path}"
    
    try:
        with open(schema_file_path, 'r', encoding='utf-8') as f: content = f.read()
        pattern = re.compile(f'CREATE TABLE\s+["`]?{table_name}["`]?\s*\(.*?\);', re.DOTALL | re.IGNORECASE)
        match = pattern.search(content)
        if match: return match.group(0)
        else: return f"错误: 在 {schema_file_path} 中未找到表 '{table_name}' 的 CREATE TABLE 语句。"
    except Exception as e:
        return f"读取或处理SQL文件 {schema_file_path} 时出错: {e}"

def _reconstruct_create_table_from_json(db_info: dict, table_name: str) -> str:
    """[策略2] 仅根据JSON数据重建CREATE TABLE语句。"""
    try:
        table_index = db_info['table_names_original'].index(table_name)
    except (ValueError, KeyError):
        return f"错误: 在数据库 '{db_info.get('db_id')}' 的JSON信息中未找到表 '{table_name}'。"

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
    
    # 1. 先将 join 操作的结果存入一个变量
    formatted_columns = ",\n    ".join(column_definitions)
    
    # 2. 然后在 f-string 中使用这个不含反斜杠的变量
    return (f'CREATE TABLE "{original_table_name}" (\n'
            f'    {formatted_columns}\n'
            f');')


def get_formatted_schema_robust(item_string: str) -> str:
    """
    健壮的模式提取函数，优先使用SQL文件，失败则回退到JSON重建。
    """
    db_id, table_name = _parse_input_string(item_string)
    if not db_id or not table_name:
        return f"错误: 无法解析输入 '{item_string}'。"

    # --- 策略1：尝试从SQL文件提取和增强 ---
    raw_create_table = _extract_raw_create_table_from_sql(db_id, table_name)
    
    if not raw_create_table.startswith("错误:"):
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

    # --- 策略2：如果策略1失败，回退到从JSON重建 ---
    else:
        # print(f"信息: 未在SQL文件中找到 '{table_name}'，尝试从JSON重建...") # 用于调试的提示信息
        json_data = get_json_schema_data()
        if not json_data or db_id not in json_data:
            return f"错误: 在JSON数据中也未找到数据库 '{db_id}'。两种策略均失败。"
        
        db_info = json_data[db_id]
        reconstructed_sql = _reconstruct_create_table_from_json(db_info, table_name)
        
        if reconstructed_sql.startswith("错误:"):
            return reconstructed_sql # 如果JSON重建也失败，返回其错误信息
            
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
        print(f"错误: 文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        
    return questions, answers, tables

def _parse_highlight_instructions(instructions_string: str) -> Dict[Tuple[str, str], Set[str]]:
    """
    (内部辅助函数) 解析包含多个高亮指令的单一字符串。
    返回一个字典，键为(数据库, 表)，值为需要高亮的列名集合。
    """
    highlights = {}
    pattern = re.compile(r"Database:\s*(.*?)\s*Table:\s*(.*?)\s*Column:\s*(.*)", re.IGNORECASE)
    
    # 将传入的单个字符串按换行符分割成多行
    lines = instructions_string.split('\n')
    
    for instruction in lines:
        # 去除每行可能存在的前后空格（特别是- 前面的空格）
        clean_instruction = instruction.strip()
        if not clean_instruction:
            continue # 跳过空行

        match = pattern.search(clean_instruction)
        if match:
            db_name = match.group(1).strip()
            table_name = match.group(2).strip()
            # 将"col1, col2"这样的字符串分割成集合{'col1', 'col2'}
            columns_to_highlight = {col.strip() for col in match.group(3).split(',')}
            
            highlights[(db_name, table_name)] = columns_to_highlight
            
    return highlights

def add_important_markers(
    statement_list: List[str], 
    instructions_string: str 
) -> List[str]:
    """
    根据指令，在CREATE TABLE语句的列注释中添加"IMPORTANT"标记。

    :param statement_list: 由之前函数生成的、包含CREATE TABLE语句的字符串列表。
    :param instructions_string: 包含所有高亮指令的单一字符串。
    :return: 一个包含修改后语句的新列表。
    """
    # 1. 解析高亮指令字符串
    highlights_map = _parse_highlight_instructions(instructions_string)
    if not highlights_map:
        print("警告: 未解析到任何有效的高亮指令。")
        return statement_list

    modified_statements = []
    
    header_pattern = re.compile(r"Database:\s*(.*?)\s*Table:\s*(.*)", re.IGNORECASE)
    col_pattern = re.compile(r'^\s*["`]?(\w+)["`]?')

    # 2. 遍历每一个待处理的语句
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

        # 4. 如果需要，则逐行进行修改
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
    异步版本：根据指令，在CREATE TABLE语句的列注释中添加"IMPORTANT"标记。
    
    :param statement_list: 由之前函数生成的、包含CREATE TABLE语句的字符串列表。
    :param instructions_string: 包含所有高亮指令的单一字符串。
    :return: 一个包含修改后语句的新列表。
    """
    # 1. 解析高亮指令字符串
    highlights_map = _parse_highlight_instructions(instructions_string)
    if not highlights_map:
        print("警告: 未解析到任何有效的高亮指令。")
        return statement_list

    modified_statements = []
    
    header_pattern = re.compile(r"Database:\s*(.*?)\s*Table:\s*(.*)", re.IGNORECASE)
    col_pattern = re.compile(r'^\s*["`]?(\w+)["`]?')

    # 2. 遍历每一个待处理的语句
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

        # 4. 如果需要，则逐行进行修改
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
        
        # 定期让出控制权，避免阻塞事件循环
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
    """异步版本的API调用"""
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
            print(f"API请求失败: {response.status}, {error_text}")
            return None

async def column_filtering_async(session, table: str, question: str):
    """异步版本的列过滤"""
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
        return table  # 如果API调用失败，返回原始table
    
    filtered_answer = answer.split("<Answer>")[1].split("</Answer>")[0].strip() if "<Answer>" in answer else answer
    return filtered_answer


async def process_batch_schema(session, batch_data, batch_index):
    """处理一个批次的数据"""
    print(f"开始处理批次 {batch_index + 1}，包含 {len(batch_data)} 个问题")
    
    # 为这个批次创建所有任务
    tasks = []
    for item in batch_data:
        question, answer, table = item
        task = column_filtering_async(session, table, question)
        tasks.append(task)
    
    # 并发执行这个批次的所有任务
    results = await asyncio.gather(*tasks)
    
    # 处理结果，保持顺序
    processed_items = []
    for i, (item, schema) in enumerate(zip(batch_data, results)):
        question, answer, table = item
        processed_items.append({"question": question, "answer": answer, "original_schema": table, "chosen_schema": schema})
    
    print(f"完成批次 {batch_index + 1}")
    return processed_items

async def load_schema_construction_data_async(file_path: str):
    """异步版本的数据加载和处理"""
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
            
            # 构建表结构数据
            for i, item in enumerate(tables_raw):
                schemas = []
                lst = item[:5]
                for tble in lst:
                    tble_schema = get_formatted_schema_robust(tble)
                    if "错误" in tble_schema:
                        print(f"Error in {i} table schema: {tble_schema}")
                    schemas.append(tble_schema)
                tables.append(schemas)
                if i % 200 == 0:
                    print(f"Processed {i} tables")
            print("tables length: ", len(tables))
            
            # 准备并发处理的数据
            batch_data = []
            for question, answer, table in zip(questions, answers, tables):
                batch_data.append((question, answer, table))
            
            print(f"总共需要处理 {len(batch_data)} 个问题")
            
            # 将数据分成批次，每批CONCURRENT_REQUESTS个
            batch_size = CONCURRENT_REQUESTS
            batches = [batch_data[i:i + batch_size] for i in range(0, len(batch_data), batch_size)]
            print(f"使用 {CONCURRENT_REQUESTS} 个并发请求，共分为 {len(batches)} 个批次")
            
            # 创建aiohttp会话
            connector = aiohttp.TCPConnector(limit=50)  # 限制连接数
            timeout = aiohttp.ClientTimeout(total=600)  # 设置超时时间
            
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
                start_time = time.time()
                
                # 逐批处理以保持顺序
                for batch_index, batch in enumerate(batches):
                    batch_results = await process_batch_schema(session, batch, batch_index)
                    
                    # 将批次结果按顺序添加到最终结果中
                    final_query.extend(batch_results)
                    
                    # 显示进度
                    processed_count = len(final_query)
                    total_count = len(batch_data)
                    elapsed_time = time.time() - start_time
                    avg_time_per_item = elapsed_time / processed_count if processed_count > 0 else 0
                    estimated_total_time = avg_time_per_item * total_count
                    remaining_time = estimated_total_time - elapsed_time
                    
                    print(f"进度: {processed_count}/{total_count} "
                          f"({processed_count/total_count*100:.1f}%) "
                          f"已用时: {elapsed_time:.1f}s "
                          f"预计剩余: {remaining_time:.1f}s")
                    
                    # 在批次之间稍作休息，避免请求过于频繁
                    if batch_index < len(batches) - 1:  # 不是最后一个批次
                        await asyncio.sleep(0.5)
                
                total_time = time.time() - start_time
                print(f"所有处理完成，总用时: {total_time:.1f}s")

    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        
    return questions, answers, tables, final_query

# 保留原有的同步函数作为备用
def load_schema_construction_data(file_path: str):
    """原有的同步数据加载函数（备用）"""
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
        print(f"错误: 文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        
    return questions, answers, tables

def column_filtering_sync(table: str, question: str):
    """原有的同步列过滤函数（备用）"""
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
    # 使用异步版本
    print("使用异步并发版本处理数据...")
    questions, answers, tables, final_query = asyncio.run(load_schema_construction_data_async(BASE_INFO_PATH))
    
    print(f"处理完成:")
    print(f"questions length: {len(questions)}")
    print(f"answers length: {len(answers)}")
    print(f"tables length: {len(tables)}")
    print(f"final_query length: {len(final_query)}")
    
    # 保存结果
    with open("/home/yfwang/wyy/schema_routing/data/spider/spider_data/cache/dev_final_query_column_filtering_baseline_Qwen3.json", "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers, "tables": tables, "final_query": final_query}, f, ensure_ascii=False, indent=4)
    
    print("数据保存完成！")
    