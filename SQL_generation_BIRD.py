import os
import re
import json
import asyncio
import aiohttp
import time
import openai

API_KEY = "API"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"
BASE_DB_PATH = "/data/spider/spider_data/database"

client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


CONCURRENT_REQUESTS = 50  

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

def load_final_queries(file_path: str):
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Check file format
        if isinstance(data, list):
            print(f"Loaded {len(data)} query items")
            return data
        elif isinstance(data, dict) and 'final_query' in data:
            final_query = data['final_query']
            print(f"Loaded {len(final_query)} query items from dictionary format")
            return final_query
        else:
            print(f"Error: Unsupported file format. Expected a list or a dictionary containing the 'final_query' key.")
            return []
            
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        return []

def generate_sql(question: str, schema: str, database: str):
    
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
    # Use regex to extract SQL and database
    match = re.search(r"<Answer>\s*-SQL: (.*?)\s*-Database: (.*?)\s*</Answer>", answer, re.DOTALL)
    if match:
        ans_sql = match.group(1).strip()
        ans_database = match.group(2).strip()
    else:
        print("No SQL or database found in the answer")
        ans_sql = "NULL"
        ans_database = database  # Use the passed database as default value
    
    final_answer = f"{ans_sql}\t{ans_database}"
    return final_answer

async def generate_sql_async(session, question: str, schema: str, database: str, chosen_schema: str):
    """Asynchronous version of SQL generation"""
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
    -SQL: SELECT count(*) FROM singer;
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
        return f"NULL\t----- bird -----\t{database}"  # If API call fails, return default value
    
    # Use regex to extract SQL and database
    match = re.search(r"<Answer>\s*-SQL: (.*?)\s*-Database: (.*?)\s*</Answer>", answer, re.DOTALL)
    if match:
        ans_sql = match.group(1).strip()
        ans_database = match.group(2).strip()
    else:
        print("No SQL or database found in the answer")
        ans_sql = "NULL"
        ans_database = database  # Use the passed database as default value

    final_answer = f"{ans_sql}\t----- bird -----\t{ans_database}"
    return final_answer

async def process_batch_sql(session, batch_data, batch_index):
    """Process a batch of SQL generation tasks"""
    print(f"Start processing batch {batch_index + 1}, containing {len(batch_data)} questions")
    
    # Create all tasks for this batch
    tasks = []
    for item in batch_data:
        question, schema, database, chosen_schema = item
        task = generate_sql_async(session, question, schema, database, chosen_schema)
        tasks.append(task)
    
    # Execute all tasks in this batch concurrently
    results = await asyncio.gather(*tasks)
    
    print(f"Completed batch {batch_index + 1}")
    return results

async def generate_sql_batch_async(final_queries):
    """Main function for asynchronous batch SQL generation"""
    final_answers = []
    
    # Prepare data for concurrent processing
    batch_data = []
    for query in final_queries:
        question = query['question']
        schema = query['original_schema']
        for table in schema:
            if 'error' in table:
                schema.remove(table)
        database = query['answer']['db_id']
        chosen_schema = query['chosen_schema']
        batch_data.append((question, schema, database, chosen_schema))
    
    print(f"Total {len(batch_data)} SQL generation tasks to process")
    
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
            batch_results = await process_batch_sql(session, batch, batch_index)
            
            # Add batch results to final results in order
            final_answers.extend(batch_results)
            
            # Display progress
            processed_count = len(final_answers)
            total_count = len(batch_data)
            elapsed_time = time.time() - start_time
            avg_time_per_item = elapsed_time / processed_count if processed_count > 0 else 0
            estimated_total_time = avg_time_per_item * total_count
            remaining_time = estimated_total_time - elapsed_time
            
            print(f"Progress: {processed_count}/{total_count} "
                  f"({processed_count/total_count*100:.1f}%) "
                  f"Elapsed: {elapsed_time:.1f}s "
                  f"ETA: {remaining_time:.1f}s")
            
            # Take a short break between batches to avoid too frequent requests
            if batch_index < len(batches) - 1:  # Not the last batch
                await asyncio.sleep(0.5)
        
        total_time = time.time() - start_time
        print(f"All SQL generation completed, total time: {total_time:.1f}s")
    
    return final_answers

if __name__ == '__main__':
    print("Generating SQL using asynchronous concurrent version...")
    final_queries = load_final_queries('\BIRD_Data\data_cache\dev\dev_final_query_column_filtering_BIRD.json')
    
    # Use asynchronous version
    final_answers = asyncio.run(generate_sql_batch_async(final_queries))
    
    print(f"Generation completed:")
    print(f"Number of SQL generated: {len(final_answers)}")
    
    # Save results as json file
    with open('\BIRD_Data\data_cache\dev\dev_final_query_column_filtering_sql_BIRD.json', "w", encoding="utf-8") as f:
        for final_answer in final_answers:
            f.write(final_answer + "\n")
    print("Data saving completed!")