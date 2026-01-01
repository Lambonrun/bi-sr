import json
from sql_metadata import Parser
import re
import asyncio
import aiohttp
import time

API_KEY = "API"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"

# Concurrency control
CONCURRENT_REQUESTS = 10  # You can modify this value to control concurrency

async def get_completion_async(session, prompt: str, system_prompt: str = "You are a helpful assistant."):
    """Asynchronous version of API call"""
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
            print(f"API request failed: {response.status}, {error_text}")
            return None

async def generate_question_async(session, question: str):
    """Asynchronous version of question generation"""
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
        return question  # If API call fails, return original question
    
    # Extract content from <Answer> tags
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
    """Process a batch of data"""
    print(f"Starting to process batch {batch_index + 1}, containing {len(batch_data)} questions")
    
    # Create all tasks for this batch
    tasks = []
    for item in batch_data:
        task = generate_question_async(session, item['question'])
        tasks.append(task)
    
    # Execute all tasks in this batch concurrently
    results = await asyncio.gather(*tasks)
    
    # Process results, maintain order
    processed_items = []
    for i, (item, generated_question) in enumerate(zip(batch_data, results)):
        query = item['SQL']
        db_id = item['db_id']
        tables = Parser(query).tables
        answer = {"db_id": db_id, "tables": tables}
        
        instruction = "Given a user's question, retrieve the most relevant table descriptions from the database."
        question = f"Instruction: {instruction} \n Query: {generated_question}"
        
        processed_items.append((question, answer))
    
    print(f"Completed batch {batch_index + 1}")
    return processed_items

async def load_queries_async(file_path: str):
    """Asynchronous version of data loading and processing"""
    questions = []
    answers = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Total {len(data)} questions to process")
        
        # Split data into batches, each with CONCURRENT_REQUESTS items
        batch_size = CONCURRENT_REQUESTS
        batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        print(f"Using {CONCURRENT_REQUESTS} concurrent requests, divided into {len(batches)} batches")
        
        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=10)  # Limit connection count
        timeout = aiohttp.ClientTimeout(total=60)  # Set timeout
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            start_time = time.time()
            
            # Process batch by batch to maintain order
            for batch_index, batch in enumerate(batches):
                batch_results = await process_batch(session, batch, batch_index)
                
                # Add batch results to final results in order
                for question, answer in batch_results:
                    questions.append(question)
                    answers.append(answer)
                
                # Show progress
                processed_count = len(questions)
                total_count = len(data)
                elapsed_time = time.time() - start_time
                avg_time_per_item = elapsed_time / processed_count if processed_count > 0 else 0
                estimated_total_time = avg_time_per_item * total_count
                remaining_time = estimated_total_time - elapsed_time
                
                print(f"Progress: {processed_count}/{total_count} "
                      f"({processed_count/total_count*100:.1f}%) "
                      f"Elapsed: {elapsed_time:.1f}s "
                      f"Remaining: {remaining_time:.1f}s")
                
                # Take a short break between batches to avoid too frequent requests
                if batch_index < len(batches) - 1:  # Not the last batch
                    await asyncio.sleep(0.5)
            
            total_time = time.time() - start_time
            print(f"All processing completed, total time: {total_time:.1f}s")
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
    return questions, answers

# Keep the original synchronous function as backup
def get_completion(prompt: str, system_prompt: str = "You are a helpful assistant."):
    """Original synchronous API call function (backup)"""
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
    """Original synchronous question generation function (backup)"""
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

def load_queries(file_path: str):
    """Original synchronous data loading function (backup)"""
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
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
    return questions, answers

if __name__ == "__main__":
    # Use asynchronous version
    print("Processing data using asynchronous concurrent version...")
    questions, answers = asyncio.run(load_queries_async("/BIRD_Data/data/dev_20240627/dev.json"))
    
    print("question length: ", len(questions))
    print("answer length: ", len(answers))
    if questions:
        print("question example: ", questions[0])
    if answers:
        print("answer example: ", answers[0])
    
    # save the questions and answers
    with open("\BIRD_Data\data\dev_20240627\dev_question_generation.json", "w", encoding="utf-8") as f:
        json.dump({"questions": questions, "answers": answers}, f, ensure_ascii=False, indent=4)
    
    print("Data saving completed!")
    
