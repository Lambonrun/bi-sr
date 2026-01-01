import openai
import json
import os
import re
import time

API_KEY = "API"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"


client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
total_consumption = 0

# How many items to process before auto-saving (can be overridden by environment variable SAVE_EVERY)
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "100"))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _atomic_write_json(path: str, data) -> None:
    """Write JSON atomically to avoid file corruption from mid-way termination."""
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)

def get_completion(prompt: str, system_prompt: str = "You are a helpful assistant."):
    global total_consumption
    message = client.chat.completions.create(
        model=MODEL_NAME,
        max_tokens=8192,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )
    #print("token usage: ", message.usage.total_tokens)
    if message.usage:
        total_consumption += message.usage.total_tokens
    return message.choices[0].message.content


def generate_database_description(table_names):
    """
    Generate database description based on table names
    """
    table_list = ", ".join(table_names)
    prompt = f"""
    Given a list of database table names, provide a concise database description (3-5 words) that summarizes its main function or domain.
    Table names: {table_list}
    Please return only the description, no other content.
    You can think step by step, and output the answer in the following format:
    <Reasoning>
    - ...
    </Reasoning>
    <Answer>
    - description
    </Answer>
    NOTICE: MAKE SURE YOUR ANSWER CONTAINS <Answer> and </Answer> TAGS.
    """

    system_prompt = "You are a database expert who can infer database functionality and purpose from table names."
    description = get_completion(prompt, system_prompt)
    description_pattern = r'<Answer>(.*?)</Answer>'
    description_match = re.search(description_pattern, description, re.DOTALL)
    if description_match:
        description_content = description_match.group(1).strip()
    else:
        print(f"Warning: Could not find <Answer> tags in response: {description}")
        description_content = description
    return description_content


def generate_table_description(table_name, table_name_original, column_names):
    """
    Generate detailed table description based on table name and column names
    """
    columns_list = ", ".join([col for col in column_names if col != "*"])
    prompt = f"""
    Given a database table name and its column names, provide a concise description (5-8 words) of what this table represents or stores.
    
    Table name: {table_name_original}
    Normalized table name: {table_name}
    Column names: {columns_list}
    
    Please return only a brief description that explains the table's purpose or what kind of data it stores.
    You can think step by step, and output the answer in the following format:
    <Reasoning>
    - ...
    </Reasoning>
    <Answer>
    - description
    </Answer>
    NOTICE: MAKE SURE YOUR ANSWER CONTAINS <Answer> and </Answer> TAGS.
    """

    system_prompt = (
        "You are a database expert who can understand table purpose from its structure."
    )
    description= get_completion(prompt, system_prompt)
    description_pattern = r'<Answer>(.*?)</Answer>'
    description_match = re.search(description_pattern, description, re.DOTALL)
    if description_match:
        description_content = description_match.group(1).strip()
    else:
        print(f"Warning: Could not find <Answer> tags in response: {description}")
        description_content = description
    return description_content


def process_tables_json(input_file_path, output_file_path):
    """
    Process tables.json file, generate new file containing database and table descriptions
    """
    # Read tables.json
    with open(input_file_path, "r", encoding="utf-8") as f:
        databases = json.load(f)

    result_data = []

    # Estimate total count (number of all tables)
    try:
        total_count = sum(len(db.get("table_names", [])) for db in databases)
    except Exception:
        total_count = 0
    processed = 0
    start_time = time.time()

    # Process each database
    for db in databases:
        db_id = db["db_id"]
        table_names = db["table_names"]
        table_names_original = db["table_names_original"]
        column_names = db["column_names"]

        print(f"Processing database: {db_id}")

        # Generate database description
        try:
            db_description= generate_database_description(table_names)
            print(f"Generated database description for {db_id}: {db_description}")
        except Exception as e:
            print(f"Error generating database description for {db_id}: {e}")
            db_description = "database"  # Default description

        # Generate detailed description for each table
        for i, (table_name, table_name_original) in enumerate(
            zip(table_names, table_names_original)
        ):
            # Get column names for this table
            table_columns = [col[1] for col in column_names if col[0] == i]

            # Generate table description
            try:
                table_description = generate_table_description(
                    table_name, table_name_original, table_columns
                )
                print(f"  Generated table description: {table_description}")
            except Exception as e:
                print(
                    f"  Error generating table description for {table_name_original}: {e}"
                )
                table_description = f"{table_name} table"  # Default description

            entry = {
                "database": db_id,
                "table": table_name_original,
                "full_text": f"Database: {db_id}, Table: {table_name_original}, Description: {table_description} in {db_description}",
            }
            result_data.append(entry)
            processed += 1

            # Auto-save every SAVE_EVERY items and print ETA
            if SAVE_EVERY > 0 and processed % SAVE_EVERY == 0:
                _atomic_write_json(output_file_path, result_data)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = max(0, (total_count - processed))
                eta = remaining / rate if rate > 0 else 0
                percent = (processed / total_count * 100) if total_count else 0
                print(
                    f"Auto-save: Processed {processed}/{total_count} items ({percent:.1f}%), "
                    f"Elapsed { _format_duration(elapsed) }, speed {rate:.2f} items/sec, ETA { _format_duration(eta) } -> {output_file_path}"
                )

    # Write output file
    _atomic_write_json(output_file_path, result_data)

    print(f"Data processing completed! Processed {len(result_data)} records in total")
    print(f"Results saved to: {output_file_path}")


def process_tables_json_raw(input_file_path, output_file_path):
    """
    Process tables.json file, generate new file containing database and table descriptions
    """
    # Read tables.json
    with open(input_file_path, "r", encoding="utf-8") as f:
        databases = json.load(f)

    result_data = []
    total_count = sum(len(db.get("table_names_original", [])) for db in databases)
    processed = 0
    start_time = time.time()

    # Process each database
    for db in databases:
        db_id = db["db_id"]
        table_names_original = db["table_names_original"]

        print(f"Processing database: {db_id}")
        for i, table_name_original in enumerate(table_names_original):
            entry = {
                "database": db_id,
                "table": table_name_original,
            }
            result_data.append(entry)
            processed += 1

            if SAVE_EVERY > 0 and processed % SAVE_EVERY == 0:
                _atomic_write_json(output_file_path, result_data)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = max(0, (total_count - processed))
                eta = remaining / rate if rate > 0 else 0
                percent = (processed / total_count * 100) if total_count else 0
                print(
                    f"Auto-save: Processed {processed}/{total_count} items ({percent:.1f}%), "
                    f"Elapsed { _format_duration(elapsed) }, speed {rate:.2f} items/sec, ETA { _format_duration(eta) } -> {output_file_path}"
                )

    # Write output file
    _atomic_write_json(output_file_path, result_data)

    print(f"Data processing completed! Processed {len(result_data)} records in total")
    print(f"Results saved to: {output_file_path}")


def test_single_database(input_file_path, db_index=0):
    """
    Test function: Process only one database to verify functionality
    """
    # Read tables.json
    with open(input_file_path, "r", encoding="utf-8") as f:
        databases = json.load(f)

    if db_index >= len(databases):
        print(f"Error: Database index {db_index} out of range, total {len(databases)} databases")
        return

    # Process only the specified database
    db = databases[db_index]
    db_id = db["db_id"]
    table_names = db["table_names"]
    table_names_original = db["table_names_original"]
    column_names = db["column_names"]

    print(f"=== Testing database: {db_id} ===")
    print(f"Number of table names: {len(table_names)}")
    print(f"Table names: {table_names}")

    # Generate database description
    try:
        print("Generating database description...")
        db_description = generate_database_description(table_names)
        print(f"Generated database description: {db_description}")
    except Exception as e:
        print(f"Error generating database description: {e}")
        db_description = "database"

    # Generate result data
    result_data = []
    for i, (table_name, table_name_original) in enumerate(
        zip(table_names, table_names_original)
    ):
        # Get column names for this table
        table_columns = [col[1] for col in column_names if col[0] == i]

        print(f"\n--- Processing table: {table_name_original} ---")
        print(f"Column names: {table_columns}")

        # Generate table description
        try:
            print("Generating table description...")
            table_description= generate_table_description(
                table_name, table_name_original, table_columns
            )
            print(f"Generated table description: {table_description}")
        except Exception as e:
            print(f"Error generating table description: {e}")
            table_description = f"{table_name} table"

        entry = {
            "database": db_id,
            "table": table_name_original,
            "full_text": f"Database: {db_id}, Table: {table_name_original}, Description: {table_description} in {db_description}",
        }
        result_data.append(entry)

    # Display results
    print(f"\n=== Generated data ===")
    for i, entry in enumerate(result_data):
        print(f"{i+1}. {entry['full_text']}")

    # Save test results
    test_output_file = f"test_result_{db_id}.json"
    with open(test_output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\nTest results saved to: {test_output_file}")
    print(f"Generated {len(result_data)} records in total")


def process_and_merge_tables_json(input_files, output_files, merged_output_file):
    """
    Process multiple tables.json files separately, then merge the results
    
    Args:
        input_files: Input file path list
        output_files: Output file path list
        merged_output_file: Merged output file path
    """
    if len(input_files) != len(output_files):
        print("Error: Number of input files and output files do not match")
        return
    
    all_result_data = []

    # Pre-count total table count for ETA (across all files)
    total_count = 0
    loaded_files = []  # [(input_file, output_file, databases)]
    for input_file, output_file in zip(input_files, output_files):
        with open(input_file, "r", encoding="utf-8") as f:
            current_databases = json.load(f)
        loaded_files.append((input_file, output_file, current_databases))
        total_count += sum(len(db.get("table_names", [])) for db in current_databases)
    processed = 0
    start_time = time.time()
    
    # Process each file separately
    for i, (input_file, output_file, current_databases) in enumerate(loaded_files):
        print(f"\n=== Processing file {i+1}: {input_file} ===")
        
        result_data = []
        processed_in_file = 0

        # Process each database in the current file
        for db in current_databases:
            db_id = db["db_id"]
            table_names = db["table_names"]
            table_names_original = db["table_names_original"]
            column_names = db["column_names"]
            
            #print(f"Processing database: {db_id}")
            
            # Generate database description
            try:
                db_description = generate_database_description(table_names)
                #print(f"Generated database description: {db_description}")
            except Exception as e:
                print(f"Error generating database description: {e}")
                db_description = "database"  # Default description
            
            # Generate detailed description for each table
            for j, (table_name, table_name_original) in enumerate(
                zip(table_names, table_names_original)
            ):
                # Get column names for this table
                table_columns = [col[1] for col in column_names if col[0] == j]
                
                # Generate table description
                try:
                    table_description = generate_table_description(
                        table_name, table_name_original, table_columns
                    )
                    #print(f"  Generated table description: {table_description}")
                except Exception as e:
                    print(f"  Error generating table description: {e}")
                    table_description = f"{table_name} table"  # Default description
                
                entry = {
                    "database": db_id,
                    "table": table_name_original,
                    "full_text": f"Database: {db_id}, Table: {table_name_original}, Description: {table_description} in {db_description}",
                }
                result_data.append(entry)
                all_result_data.append(entry)
                processed += 1
                processed_in_file += 1

                if SAVE_EVERY > 0 and processed % SAVE_EVERY == 0:
                    # Perform atomic write for both single file and merged file
                    _atomic_write_json(output_file, result_data)
                    _atomic_write_json(merged_output_file, all_result_data)

                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = max(0, (total_count - processed))
                    eta = remaining / rate if rate > 0 else 0
                    percent = (processed / total_count * 100) if total_count else 0
                    print(
                        f"Auto-save: Overall progress {processed}/{total_count} items ({percent:.1f}%), "
                        f"Elapsed { _format_duration(elapsed) }, speed {rate:.2f} items/sec, ETA { _format_duration(eta) }"
                    )
                    print(
                        f"  -> Current file written: {output_file}; Merged file written: {merged_output_file}"
                    )
        
        # Write individual output file
        _atomic_write_json(output_file, result_data)
        
        print(f"File processing completed! Processed {len(result_data)} records in total")
        print(f"Results saved to: {output_file}")
    
    # Write merged file
    _atomic_write_json(merged_output_file, all_result_data)
    
    print(f"\n=== Merged Results ===")
    print(f"Processed {len(all_result_data)} records in total")
    print(f"Merged results saved to: {merged_output_file}")


def process_and_merge_tables_json_raw(input_files, output_files, merged_output_file):
    """
    Process multiple tables.json files separately (baseline mode), then merge the results
    
    Args:
        input_files: Input file path list
        output_files: Output file path list
        merged_output_file: Merged output file path
    """
    if len(input_files) != len(output_files):
        print("Error: Number of input files and output files do not match")
        return
    
    all_result_data = []

    # Pre-count total table count (across all files)
    total_count = 0
    loaded_files = []
    for input_file, output_file in zip(input_files, output_files):
        with open(input_file, "r", encoding="utf-8") as f:
            current_databases = json.load(f)
        loaded_files.append((input_file, output_file, current_databases))
        total_count += sum(len(db.get("table_names_original", [])) for db in current_databases)
    processed = 0
    start_time = time.time()
    
    for i, (input_file, output_file, current_databases) in enumerate(loaded_files):
        print(f"\n=== Processing file {i+1}: {input_file} ===")
        
        result_data = []
        
        # Process each database in the current file
        for db in current_databases:
            db_id = db["db_id"]
            table_names_original = db["table_names_original"]
            
            for table_name_original in table_names_original:
                entry = {
                    "database": db_id,
                    "table": table_name_original,
                }
                result_data.append(entry)
                all_result_data.append(entry)
                processed += 1

                if SAVE_EVERY > 0 and processed % SAVE_EVERY == 0:
                    _atomic_write_json(output_file, result_data)
                    _atomic_write_json(merged_output_file, all_result_data)

                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = max(0, (total_count - processed))
                    eta = remaining / rate if rate > 0 else 0
                    percent = (processed / total_count * 100) if total_count else 0
                    print(
                        f"Auto-save: Overall progress {processed}/{total_count} items ({percent:.1f}%), "
                        f"Elapsed { _format_duration(elapsed) }, speed {rate:.2f} items/sec, ETA { _format_duration(eta) }"
                    )
                    print(
                        f"  -> Current file written: {output_file}; Merged file written: {merged_output_file}"
                    )
        
        # Write individual output file
        _atomic_write_json(output_file, result_data)
        
        print(f"File processing completed! Processed {len(result_data)} records in total")
        print(f"Results saved to: {output_file}")
    
    # Write merged file
    _atomic_write_json(merged_output_file, all_result_data)
    
    print(f"\n=== Merged Results ===")
    print(f"Processed {len(all_result_data)} records in total")
    print(f"Merged results saved to: {merged_output_file}")


if __name__ == "__main__":
    # Set file paths
    dev_input_file = "/BIRD_Data/data/dev_20240627/dev_tables.json"
    train_input_file = "/BIRD_Data/data/train/train_tables.json"
    
    dev_output_file = "/BIRD_Data/data/dev_20240627/dev_table_descriptions.json"
    train_output_file = "/BIRD_Data/data/train/train_table_descriptions.json"
    merged_output_file = "/BIRD_Data/data/merged_table_descriptions.json"
    
    dev_baseline_output = "/BIRD_Data/data/dev_20240627/dev_table_baseline.json"
    train_baseline_output = "/BIRD_Data/data/train/train_table_baseline.json"
    merged_baseline_output = "/BIRD_Data/data/merged_table_baseline.json"
    
    # Check if input files exist
    if not os.path.exists(dev_input_file):
        print(f"Error: File not found {dev_input_file}")
    if not os.path.exists(train_input_file):
        print(f"Error: File not found {train_input_file}")
    
    if not os.path.exists(dev_input_file) or not os.path.exists(train_input_file):
        print("Please ensure tables.json files exist in the correct paths")
    else:
        print("=== Start processing all data ===")
        print("Is it baseline mode? (y/n): ")
        user_input = input()
        
        if user_input.lower() == "y":
            # Baseline mode: Process multiple files and merge
            input_files = [dev_input_file, train_input_file]
            output_files = [dev_baseline_output, train_baseline_output]
            process_and_merge_tables_json_raw(input_files, output_files, merged_baseline_output)
        else:
            # Run test function first
            print("=== Start test mode ===")
            print("Please select the file to test:")
            print("1. dev_tables.json")
            print("2. train_tables.json")
            file_choice = input("Please enter choice (1 or 2): ")
            
            if file_choice == "1":
                test_single_database(dev_input_file, db_index=0)
            elif file_choice == "2":
                test_single_database(train_input_file, db_index=0)
            else:
                print("Invalid choice, defaulting to test dev file")
                test_single_database(dev_input_file, db_index=0)
            
            # Ask if continue processing all data
            print("\n=== Test completed ===")
            user_input = input("Continue processing all databases? (y/n): ")
            if user_input.lower() == "y":
                # Process multiple files and merge
                input_files = [dev_input_file, train_input_file]
                output_files = [dev_output_file, train_output_file]
                process_and_merge_tables_json(input_files, output_files, merged_output_file)
            else:
                print("Program ended")
    print(f"Total tokens consumed: {total_consumption}")
