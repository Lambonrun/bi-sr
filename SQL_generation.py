import os
import re
import json
import asyncio
import aiohttp
import time
import openai

API_KEY = "sk-9fe9b714a9ad4b6ab83bf7a13ead42ec"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"
BASE_DB_PATH = "/home/yfwang/wyy/schema_routing/data/spider/spider_data/database"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)

# 并发数控制
CONCURRENT_REQUESTS = 50  # 可以修改这个值来控制并发数

def get_completion(prompt: str, system_prompt: str = "You are a helpful assistant."):
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

def load_final_queries(file_path: str):
    """
    加载最终查询数据，兼容不同格式
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查文件格式
        if isinstance(data, list):
            print(f"加载了 {len(data)} 个查询项目")
            return data
        elif isinstance(data, dict) and 'final_query' in data:
            final_query = data['final_query']
            print(f"从字典格式中加载了 {len(final_query)} 个查询项目")
            return final_query
        else:
            print(f"错误: 不支持的文件格式。期望列表或包含'final_query'键的字典。")
            return []
            
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
        return []
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        return []

def generate_sql(question: str, schema: str, database: str):
    """原有的同步SQL生成函数（备用）"""
    sys_prompt = """
    You are an expert in database domain.
    """
    prompt = f"""
    You are given a question, a target database and several table schemas. You need to generate a SQL query to answer the question.
    Each table schema includes database name, table name, and a CREATE TABLE statement with column names, types and additional column comments. Some columns are marked as IMPORTANT, which means they may be especially important to answer the question. While some schemas are not related to the target database, you should ignore them.
    You need to generate a SQL query to answer the question and the targeted database.

    <Example>
    Question: How many singers do we have?
    Schema:
    [['Database: singer Table: singer\\nTable information:\\nCREATE TABLE \"singer\" (\\n\"Singer_ID\" int --IMPORTANT singer id,\\n\"Name\" text -- name,\\n\"Birth_Year\" real -- birth year,\\n\"Net_Worth_Millions\" real -- net worth millions,\\n\"Citizenship\" text -- citizenship,\\nPRIMARY KEY (\"Singer_ID\")\\n);', 'Database: concert_singer Table: singer\\nTable information:\\nCREATE TABLE \"singer\" (\\n\"Singer_ID\" int --IMPORTANT singer id,\\n\"Name\" text --IMPORTANT name,\\n\"Country\" text -- country,\\n\"Song_Name\" text -- song name,\\n\"Song_release_year\" text -- song release year,\\n\"Age\" int -- age,\\n\"Is_male\" bool -- is male,\\nPRIMARY KEY (\"Singer_ID\")\\n);', 'Database: singer Table: song\\nTable information\\nCREATE TABLE \"song\" (\\n\"Song_ID\" int -- song id,\\n\"Title\" text -- title,\\n\"Singer_ID\" int -- singer id,\\n\"Sales\" real -- sales,\\n\"Highest_Position\" real -- highest position,\\nPRIMARY KEY (\"Song_ID\"),\\nFOREIGN KEY (\"Singer_ID\") REFERENCES `singer`(\"Singer_ID\")\\n);', 'Database: music_2 Table: Vocals\\nTable information:\\nCREATE TABLE \"Vocals\" ( \\n\\t\"SongId\" INTEGER, \\n\\t\"Bandmate\" INTEGER, \\n\\t\"Type\" TEXT,\\n\\tPRIMARY KEY(SongId, Bandmate),\\n\\tFOREIGN KEY (SongId) REFERENCES Songs(SongId),\\n\\tFOREIGN KEY (Bandmate) REFERENCES Band(Id)\\n);', 'Database: concert_singer Table: singer_in_concert\\nTable information:\\nCREATE TABLE \"singer_in_concert\" (\\n\"concert_ID\" int,\\n\"Singer_ID\" text,\\nPRIMARY KEY (\"concert_ID\",\"Singer_ID\"),\\nFOREIGN KEY (\"concert_ID\") REFERENCES \"concert\"(\"concert_ID\"),\\nFOREIGN KEY (\"Singer_ID\") REFERENCES \"singer\"(\"Singer_ID\")\\n);']]
    Target database: concert_singer
    
    Answer:
    -SQL: SELECT count(*) FROM singer
    -Database: concert_singer
    </Example>

    Question: {question}
    Schema: {schema}
    Target database: {database}
    Make sure that your SQL answer is in one line!!!!
    You can think step by step, and output the answer in the following format:
    <Reasoning>
    
    </Reasoning>
    <Answer>
    -SQL: ...
    -Database: ...
    </Answer>
    """
    answer = get_completion(prompt, system_prompt=sys_prompt)
    # 使用正则表达式提取SQL和数据库
    match = re.search(r"<Answer>\s*-SQL: (.*?)\s*-Database: (.*?)\s*</Answer>", answer, re.DOTALL)
    if match:
        ans_sql = match.group(1).strip()
        ans_database = match.group(2).strip()
    else:
        print("No SQL or database found in the answer")
        ans_sql = "NULL"
        ans_database = database  # 使用传入的数据库作为默认值
    
    final_answer = f"{ans_sql}\t{ans_database}"
    return final_answer

async def generate_sql_async(session, question: str, schema: str, database: str, chosen_schema: str):
    """异步版本的SQL生成"""
    sys_prompt = """
    You are an expert in database domain.
    """
    prompt = f"""
    You are given a question, a target database and several table schemas. You need to generate a SQL query to answer the question.
    Each table schema includes database name, table name, and a CREATE TABLE statement with column names, types and additional column comments. Also, you are given some columns worth noting, which are the columns that may be important to answer the question.
    You need to generate a SQLite3 SQL query to answer the question and the targeted database.

    <Example>
    Question: How many singers do we have?
    Schema:
    [
        "Database: singer Table: singer\nTable information:\nCREATE TABLE \"singer\" (\n\"Singer_ID\" int -- singer id,\n\"Name\" text -- name,\n\"Birth_Year\" real -- birth year,\n\"Net_Worth_Millions\" real -- net worth millions,\n\"Citizenship\" text -- citizenship,\nPRIMARY KEY (\"Singer_ID\")\n);",
        "Database: concert_singer Table: singer\nTable information:\nCREATE TABLE \"singer\" (\n\"Singer_ID\" int -- singer id,\n\"Name\" text -- name,\n\"Country\" text -- country,\n\"Song_Name\" text -- song name,\n\"Song_release_year\" text -- song release year,\n\"Age\" int -- age,\n\"Is_male\" bool -- is male,\nPRIMARY KEY (\"Singer_ID\")\n);",
        "Database: singer Table: song\nTable information:\nCREATE TABLE \"song\" (\n\"Song_ID\" int -- song id,\n\"Title\" text -- title,\n\"Singer_ID\" int -- singer id,\n\"Sales\" real -- sales,\n\"Highest_Position\" real -- highest position,\nPRIMARY KEY (\"Song_ID\"),\nFOREIGN KEY (\"Singer_ID\") REFERENCES `singer`(\"Singer_ID\")\n);",
        "Database: music_2 Table: Vocals\nTable information:\nCREATE TABLE \"Vocals\" ( \n\t\"SongId\" INTEGER, \n\t\"Bandmate\" INTEGER, \n\t\"Type\" TEXT,\n\tPRIMARY KEY(SongId, Bandmate),\n\tFOREIGN KEY (SongId) REFERENCES Songs(SongId),\n\tFOREIGN KEY (Bandmate) REFERENCES Band(Id)\n);",
        "Database: music_4 Table: artist\nTable information:\nCREATE TABLE \"artist\" (\n    \"Artist_ID\" int -- artist id,\n    \"Artist\" text -- artist,\n    \"Age\" int -- age,\n    \"Famous_Title\" text -- famous title,\n    \"Famous_Release_date\" text -- famous release date,\n    PRIMARY KEY (\"Artist_ID\")\n);"
            ]
    Columns worth noting: "- Database: singer Table: singer Column: Singer_ID\n- Database: concert_singer Table: singer Column: Singer_ID, Name"
    Target database: concert_singer
    
    Answer:
    -SQL: SELECT count(*) FROM singer
    -Database: concert_singer
    </Example>

    Question: {question}
    Schema: {schema}
    Columns worth noting: {chosen_schema}
    Target database: {database}
    You need to generate a SQLite3 SQL query to answer the question and the targeted database. Generate the SQL query answer anyway.
    Make sure that your SQLite3 SQL answer is in one line, try not use \n in your SQL query!!!!

    You can think step by step, and output the answer in the following format:
    <Reasoning>
    
    </Reasoning>
    <Answer>
    -SQL: ...
    -Database: ...
    </Answer>
    """
    
    answer = await get_completion_async(session, prompt, system_prompt=sys_prompt)
    
    if answer is None:
        return f"NULL\t{database}"  # 如果API调用失败，返回默认值
    
    # 使用正则表达式提取SQL和数据库
    match = re.search(r"<Answer>\s*-SQL: (.*?)\s*-Database: (.*?)\s*</Answer>", answer, re.DOTALL)
    if match:
        ans_sql = match.group(1).strip()
        ans_database = match.group(2).strip()
    else:
        print("No SQL or database found in the answer")
        ans_sql = "NULL"
        ans_database = database  # 使用传入的数据库作为默认值

    final_answer = f"{ans_sql}"
    return final_answer

async def process_batch_sql(session, batch_data, batch_index):
    """处理一个批次的SQL生成任务"""
    print(f"开始处理批次 {batch_index + 1}，包含 {len(batch_data)} 个问题")
    
    # 为这个批次创建所有任务
    tasks = []
    for item in batch_data:
        question, schema, database, chosen_schema = item
        task = generate_sql_async(session, question, schema, database, chosen_schema)
        tasks.append(task)
    
    # 并发执行这个批次的所有任务
    results = await asyncio.gather(*tasks)
    
    print(f"完成批次 {batch_index + 1}")
    return results

async def generate_sql_batch_async(final_queries):
    """异步批量生成SQL的主函数"""
    final_answers = []
    
    # 准备并发处理的数据
    batch_data = []
    for query in final_queries:
        question = query['question']
        schema = query['original_schema']
        database = query['answer']['db_id']
        chosen_schema = query['chosen_schema']
        batch_data.append((question, schema, database, chosen_schema))
    
    print(f"总共需要处理 {len(batch_data)} 个SQL生成任务")
    
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
            batch_results = await process_batch_sql(session, batch, batch_index)
            
            # 将批次结果按顺序添加到最终结果中
            final_answers.extend(batch_results)
            
            # 显示进度
            processed_count = len(final_answers)
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
        print(f"所有SQL生成完成，总用时: {total_time:.1f}s")
    
    return final_answers

if __name__ == '__main__':
    print("使用异步并发版本生成SQL...")
    final_queries = load_final_queries("/home/yfwang/wyy/schema_routing/data/spider/spider_data/cache/dev_final_query_column_filtering_baseline_Qwen3.json")
    
    # 使用异步版本
    final_answers = asyncio.run(generate_sql_batch_async(final_queries))
    
    print(f"生成完成:")
    print(f"SQL生成数量: {len(final_answers)}")
    
    # 保存结果为sql文件
    with open("/home/yfwang/wyy/schema_routing/data/spider/spider_data/cache/dev_final_query_column_filtering_sql_baseline_Qwen3.sql", "w", encoding="utf-8") as f:
        for final_answer in final_answers:
            f.write(final_answer + "\n")
    print("数据保存完成！")