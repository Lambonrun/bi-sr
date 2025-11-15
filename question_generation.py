import json
from sql_metadata import Parser
import re
import asyncio
import aiohttp
import time

API_KEY = "sk-9fe9b714a9ad4b6ab83bf7a13ead42ec"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

# 并发数控制
CONCURRENT_REQUESTS = 10  # 可以修改这个值来控制并发数

async def get_completion_async(session, prompt: str, system_prompt: str = "You are a helpful assistant."):
    """异步版本的API调用"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL_NAME,
        "max_tokens": 8192,
        "temperature": 0.7,
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

async def generate_question_async(session, question: str):
    """异步版本的问题生成"""
    sys_prompt = """
    You are a helpful assistant in database domain.
    """
    prompt = f"""
    Your job is to generate the tables needed to answer the question. You need to describe tables needed to answer the question, and generate the table description with the table name, with other potential tables names.

    <Example>
    Question: What is the name of the document with the most number of sections?
    Answer: 
    - documents: Stores document metadata and access information, potential names:[Document, Documents, Document_Metadata, Document_Access_Information]
    - document_sections: Document sections with sequence and titles, potential names:[Section, Sections, Section_Titles, Section_Sequences]
    </Example>

    <Example>
    Question: Find distinct cities of addresses of people?
    Answer: 
    - addresses: Stores address information including city names, potential names:[Address, Addresses, Location, Address_Details, Address_Info]
    - people_addresses: Links people to their addresses (junction table), potential names:[Person_Address, People_Addresses, People_Locations, Address_Assignments, Person_Location]
    </Example>

    Question: {question}
    
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
    answer = await get_completion_async(session, prompt, system_prompt=sys_prompt)
    
    if answer is None:
        return question  # 如果API调用失败，返回原问题
    
    # 提取<Answer>标签中的内容
    answer_pattern = r'<Answer>(.*?)</Answer>'
    answer_match = re.search(answer_pattern, answer, re.DOTALL)
    
    if answer_match:
        answer_content = answer_match.group(1).strip()
    else:
        print(f"Warning: Could not find <Answer> tags in response: {answer}")
        answer_content = answer
        
    final_question = question + "\n potential tables:\n" + answer_content
    return final_question

async def process_batch(session, batch_data, batch_index):
    """处理一个批次的数据"""
    print(f"开始处理批次 {batch_index + 1}，包含 {len(batch_data)} 个问题")
    
    # 为这个批次创建所有任务
    tasks = []
    for item in batch_data:
        task = generate_question_async(session, item['question'])
        tasks.append(task)
    
    # 并发执行这个批次的所有任务
    results = await asyncio.gather(*tasks)
    
    # 处理结果，保持顺序
    processed_items = []
    for i, (item, generated_question) in enumerate(zip(batch_data, results)):
        query = item['SQL']
        db_id = item['db_id']
        tables = Parser(query).tables
        answer = {"db_id": db_id, "tables": tables}
        
        instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
        question = f"Instruction: {instruction} \n Query: {generated_question}"
        
        processed_items.append((question, answer))
    
    print(f"完成批次 {batch_index + 1}")
    return processed_items

async def load_queries_async(file_path: str):
    """异步版本的数据加载和处理"""
    questions = []
    answers = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"总共需要处理 {len(data)} 个问题")
        
        # 将数据分成批次，每批CONCURRENT_REQUESTS个
        batch_size = CONCURRENT_REQUESTS
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        print(f"使用 {CONCURRENT_REQUESTS} 个并发请求，共分为 {len(batches)} 个批次")
        
        # 创建aiohttp会话
        connector = aiohttp.TCPConnector(limit=10)  # 限制连接数
        timeout = aiohttp.ClientTimeout(total=60)  # 设置超时时间
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            start_time = time.time()
            
            # 逐批处理以保持顺序
            for batch_index, batch in enumerate(batches):
                batch_results = await process_batch(session, batch, batch_index)
                
                # 将批次结果按顺序添加到最终结果中
                for question, answer in batch_results:
                    questions.append(question)
                    answers.append(answer)
                
                # 显示进度
                processed_count = len(questions)
                total_count = len(data)
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
        
    return questions, answers

# 保留原有的同步函数作为备用
def get_completion(prompt: str, system_prompt: str = "You are a helpful assistant."):
    """原有的同步API调用函数（备用）"""
    import openai
    client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
    message = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=8192,
        temperature=0.7,
        messages=[
          {"role": "system", "content": system_prompt},
          {"role": "user", "content": prompt}
        ]
    )
    return message.choices[0].message.content

def generate_question(question: str):
    """原有的同步问题生成函数（备用）"""
    sys_prompt = """
    You are a helpful assistant in database domain.
    """
    prompt = f"""
    Your job is to generate the tables needed to answer the question. You need to describe tables needed to answer the question, and generate the table description with the table name, with other potential tables names.

    <Example>
    Question: What is the name of the document with the most number of sections?
    Answer: 
    - documents: Stores document metadata and access information, potential names:[Document, Documents, Document_Metadata, Document_Access_Information]
    - document_sections: Document sections with sequence and titles, potential names:[Section, Sections, Section_Titles, Section_Sequences]
    </Example>

    <Example>
    Question: Find distinct cities of addresses of people?
    Answer: 
    - addresses: Stores address information including city names, potential names:[Address, Addresses, Location, Address_Details, Address_Info]
    - people_addresses: Links people to their addresses (junction table), potential names:[Person_Address, People_Addresses, People_Locations, Address_Assignments, Person_Location]
    </Example>

    Question: {question}
    
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

def load_queries(file_path: str):
    """原有的同步数据加载函数（备用）"""
    questions = []
    answers = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                question= generate_question(item['question'])
                query = item['query']
                db_id = item['db_id']
                tables = Parser(query).tables
                answer = {"db_id": db_id, "tables": tables}
                
                instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
                questions.append(f"Instruction: {instruction} \n Query: {question}")
                answers.append(answer)
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        
    return questions, answers

if __name__ == "__main__":
    # 使用异步版本
    print("使用异步并发版本处理数据...")
    questions, answers = asyncio.run(load_queries_async("/root/autodl-tmp/schlink/BIRD_Data/data/dev_20240627/dev.json"))
    
    print("question length: ", len(questions))
    print("answer length: ", len(answers))
    if questions:
        print("question example: ", questions[0])
    if answers:
        print("answer example: ", answers[0])
    
    # save the questions and answers
    with open("/root/autodl-tmp/schlink/BIRD_Data/data/dev_20240627/dev_question_generation.json", "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers}, f, ensure_ascii=False, indent=4)
    
    print("数据保存完成！")
    
