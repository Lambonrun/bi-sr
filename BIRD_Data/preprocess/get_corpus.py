import openai
import json
import os
import re
import time

API_KEY = "sk-9fe9b714a9ad4b6ab83bf7a13ead42ec"
MODEL_NAME = "deepseek-chat"
BASE_URL = "https://api.deepseek.com/v1"


client = openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)
total_consumption = 0

# 每处理多少条自动保存一次（可用环境变量 SAVE_EVERY 覆盖）
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "100"))


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _atomic_write_json(path: str, data) -> None:
    """以原子方式写入 JSON，避免中途终止导致文件损坏。"""
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
    根据数据库的表名生成数据库描述
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
    根据表名和列名生成表的详细描述
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
    处理 tables.json 文件，生成包含数据库和表描述的新文件
    """
    # 读取 tables.json
    with open(input_file_path, "r", encoding="utf-8") as f:
        databases = json.load(f)

    result_data = []

    # 预估总条数（所有表的数量）
    try:
        total_count = sum(len(db.get("table_names", [])) for db in databases)
    except Exception:
        total_count = 0
    processed = 0
    start_time = time.time()

    # 处理每个数据库
    for db in databases:
        db_id = db["db_id"]
        table_names = db["table_names"]
        table_names_original = db["table_names_original"]
        column_names = db["column_names"]

        print(f"Processing database: {db_id}")

        # 生成数据库描述
        try:
            db_description= generate_database_description(table_names)
            print(f"Generated database description for {db_id}: {db_description}")
        except Exception as e:
            print(f"Error generating database description for {db_id}: {e}")
            db_description = "database"  # 默认描述

        # 为每个表生成详细描述
        for i, (table_name, table_name_original) in enumerate(
            zip(table_names, table_names_original)
        ):
            # 获取该表的列名
            table_columns = [col[1] for col in column_names if col[0] == i]

            # 生成表描述
            try:
                table_description = generate_table_description(
                    table_name, table_name_original, table_columns
                )
                print(f"  Generated table description: {table_description}")
            except Exception as e:
                print(
                    f"  Error generating table description for {table_name_original}: {e}"
                )
                table_description = f"{table_name} table"  # 默认描述

            entry = {
                "database": db_id,
                "table": table_name_original,
                "full_text": f"Database: {db_id}, Table: {table_name_original}, Description: {table_description} in {db_description}",
            }
            result_data.append(entry)
            processed += 1

            # 每 SAVE_EVERY 条自动保存并打印 ETA
            if SAVE_EVERY > 0 and processed % SAVE_EVERY == 0:
                _atomic_write_json(output_file_path, result_data)
                elapsed = time.time() - start_time
                rate = processed / elapsed if elapsed > 0 else 0
                remaining = max(0, (total_count - processed))
                eta = remaining / rate if rate > 0 else 0
                percent = (processed / total_count * 100) if total_count else 0
                print(
                    f"自动保存: 已处理 {processed}/{total_count} 条 ({percent:.1f}%), "
                    f"耗时 { _format_duration(elapsed) }, 速度 {rate:.2f} 条/秒, 预计剩余 { _format_duration(eta) } -> {output_file_path}"
                )

    # 写入输出文件
    _atomic_write_json(output_file_path, result_data)

    print(f"数据处理完成！共处理了 {len(result_data)} 条记录")
    print(f"结果已保存到: {output_file_path}")


def process_tables_json_raw(input_file_path, output_file_path):
    """
    处理 tables.json 文件，生成包含数据库和表描述的新文件
    """
    # 读取 tables.json
    with open(input_file_path, "r", encoding="utf-8") as f:
        databases = json.load(f)

    result_data = []
    total_count = sum(len(db.get("table_names_original", [])) for db in databases)
    processed = 0
    start_time = time.time()

    # 处理每个数据库
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
                    f"自动保存: 已处理 {processed}/{total_count} 条 ({percent:.1f}%), "
                    f"耗时 { _format_duration(elapsed) }, 速度 {rate:.2f} 条/秒, 预计剩余 { _format_duration(eta) } -> {output_file_path}"
                )

    # 写入输出文件
    _atomic_write_json(output_file_path, result_data)

    print(f"数据处理完成！共处理了 {len(result_data)} 条记录")
    print(f"结果已保存到: {output_file_path}")


def test_single_database(input_file_path, db_index=0):
    """
    测试函数：只处理一个数据库来验证功能
    """
    # 读取 tables.json
    with open(input_file_path, "r", encoding="utf-8") as f:
        databases = json.load(f)

    if db_index >= len(databases):
        print(f"错误：数据库索引 {db_index} 超出范围，总共有 {len(databases)} 个数据库")
        return

    # 只处理指定的数据库
    db = databases[db_index]
    db_id = db["db_id"]
    table_names = db["table_names"]
    table_names_original = db["table_names_original"]
    column_names = db["column_names"]

    print(f"=== 测试数据库: {db_id} ===")
    print(f"表名数量: {len(table_names)}")
    print(f"表名: {table_names}")

    # 生成数据库描述
    try:
        print("正在生成数据库描述...")
        db_description = generate_database_description(table_names)
        print(f"生成的数据库描述: {db_description}")
    except Exception as e:
        print(f"生成数据库描述时出错: {e}")
        db_description = "database"

    # 生成结果数据
    result_data = []
    for i, (table_name, table_name_original) in enumerate(
        zip(table_names, table_names_original)
    ):
        # 获取该表的列名
        table_columns = [col[1] for col in column_names if col[0] == i]

        print(f"\n--- 处理表: {table_name_original} ---")
        print(f"列名: {table_columns}")

        # 生成表描述
        try:
            print("正在生成表描述...")
            table_description= generate_table_description(
                table_name, table_name_original, table_columns
            )
            print(f"生成的表描述: {table_description}")
        except Exception as e:
            print(f"生成表描述时出错: {e}")
            table_description = f"{table_name} table"

        entry = {
            "database": db_id,
            "table": table_name_original,
            "full_text": f"Database: {db_id}, Table: {table_name_original}, Description: {table_description} in {db_description}",
        }
        result_data.append(entry)

    # 显示结果
    print(f"\n=== 生成的数据 ===")
    for i, entry in enumerate(result_data):
        print(f"{i+1}. {entry['full_text']}")

    # 保存测试结果
    test_output_file = f"test_result_{db_id}.json"
    with open(test_output_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

    print(f"\n测试结果已保存到: {test_output_file}")
    print(f"共生成 {len(result_data)} 条记录")


def process_and_merge_tables_json(input_files, output_files, merged_output_file):
    """
    分别处理多个 tables.json 文件，然后合并结果
    
    Args:
        input_files: 输入文件路径列表
        output_files: 输出文件路径列表
        merged_output_file: 合并后的输出文件路径
    """
    if len(input_files) != len(output_files):
        print("错误：输入文件和输出文件数量不匹配")
        return
    
    all_result_data = []

    # 预先统计总表数用于 ETA（跨所有文件）
    total_count = 0
    loaded_files = []  # [(input_file, output_file, databases)]
    for input_file, output_file in zip(input_files, output_files):
        with open(input_file, "r", encoding="utf-8") as f:
            current_databases = json.load(f)
        loaded_files.append((input_file, output_file, current_databases))
        total_count += sum(len(db.get("table_names", [])) for db in current_databases)
    processed = 0
    start_time = time.time()
    
    # 分别处理每个文件
    for i, (input_file, output_file, current_databases) in enumerate(loaded_files):
        print(f"\n=== 处理第 {i+1} 个文件: {input_file} ===")
        
        result_data = []
        processed_in_file = 0

        # 处理当前文件中的每个数据库
        for db in current_databases:
            db_id = db["db_id"]
            table_names = db["table_names"]
            table_names_original = db["table_names_original"]
            column_names = db["column_names"]
            
            #print(f"处理数据库: {db_id}")
            
            # 生成数据库描述
            try:
                db_description = generate_database_description(table_names)
                #print(f"生成的数据库描述: {db_description}")
            except Exception as e:
                print(f"生成数据库描述时出错: {e}")
                db_description = "database"  # 默认描述
            
            # 为每个表生成详细描述
            for j, (table_name, table_name_original) in enumerate(
                zip(table_names, table_names_original)
            ):
                # 获取该表的列名
                table_columns = [col[1] for col in column_names if col[0] == j]
                
                # 生成表描述
                try:
                    table_description = generate_table_description(
                        table_name, table_name_original, table_columns
                    )
                    #print(f"  生成的表描述: {table_description}")
                except Exception as e:
                    print(f"  生成表描述时出错: {e}")
                    table_description = f"{table_name} table"  # 默认描述
                
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
                    # 单文件与合并文件均做一次安全写入
                    _atomic_write_json(output_file, result_data)
                    _atomic_write_json(merged_output_file, all_result_data)

                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = max(0, (total_count - processed))
                    eta = remaining / rate if rate > 0 else 0
                    percent = (processed / total_count * 100) if total_count else 0
                    print(
                        f"自动保存: 全部进度 {processed}/{total_count} 条 ({percent:.1f}%), "
                        f"耗时 { _format_duration(elapsed) }, 速度 {rate:.2f} 条/秒, 预计剩余 { _format_duration(eta) }"
                    )
                    print(
                        f"  -> 当前文件已写入: {output_file}; 合并文件已写入: {merged_output_file}"
                    )
        
        # 写入单个输出文件
        _atomic_write_json(output_file, result_data)
        
        print(f"文件处理完成！共处理了 {len(result_data)} 条记录")
        print(f"结果已保存到: {output_file}")
    
    # 写入合并后的文件
    _atomic_write_json(merged_output_file, all_result_data)
    
    print(f"\n=== 合并结果 ===")
    print(f"总共处理了 {len(all_result_data)} 条记录")
    print(f"合并结果已保存到: {merged_output_file}")


def process_and_merge_tables_json_raw(input_files, output_files, merged_output_file):
    """
    分别处理多个 tables.json 文件（baseline模式），然后合并结果
    
    Args:
        input_files: 输入文件路径列表
        output_files: 输出文件路径列表
        merged_output_file: 合并后的输出文件路径
    """
    if len(input_files) != len(output_files):
        print("错误：输入文件和输出文件数量不匹配")
        return
    
    all_result_data = []

    # 预先统计总表数（跨所有文件）
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
        print(f"\n=== 处理第 {i+1} 个文件: {input_file} ===")
        
        result_data = []
        
        # 处理当前文件中的每个数据库
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
                        f"自动保存: 全部进度 {processed}/{total_count} 条 ({percent:.1f}%), "
                        f"耗时 { _format_duration(elapsed) }, 速度 {rate:.2f} 条/秒, 预计剩余 { _format_duration(eta) }"
                    )
                    print(
                        f"  -> 当前文件已写入: {output_file}; 合并文件已写入: {merged_output_file}"
                    )
        
        # 写入单个输出文件
        _atomic_write_json(output_file, result_data)
        
        print(f"文件处理完成！共处理了 {len(result_data)} 条记录")
        print(f"结果已保存到: {output_file}")
    
    # 写入合并后的文件
    _atomic_write_json(merged_output_file, all_result_data)
    
    print(f"\n=== 合并结果 ===")
    print(f"总共处理了 {len(all_result_data)} 条记录")
    print(f"合并结果已保存到: {merged_output_file}")


if __name__ == "__main__":
    # 设置文件路径
    dev_input_file = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/dev_20240627/dev_tables.json"
    train_input_file = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/train/train_tables.json"
    
    dev_output_file = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/dev_20240627/dev_table_descriptions.json"
    train_output_file = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/train/train_table_descriptions.json"
    merged_output_file = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/merged_table_descriptions.json"
    
    dev_baseline_output = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/dev_20240627/dev_table_baseline.json"
    train_baseline_output = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/train/train_table_baseline.json"
    merged_baseline_output = "/Users/wuyuyang/Code/schlink/BIRD_Data/data/merged_table_baseline.json"
    
    # 检查输入文件是否存在
    if not os.path.exists(dev_input_file):
        print(f"错误：找不到文件 {dev_input_file}")
    if not os.path.exists(train_input_file):
        print(f"错误：找不到文件 {train_input_file}")
    
    if not os.path.exists(dev_input_file) or not os.path.exists(train_input_file):
        print("请确保 tables.json 文件存在于正确的路径")
    else:
        print("=== 开始处理全部数据 ===")
        print("是否为baseline模式？(y/n): ")
        user_input = input()
        
        if user_input.lower() == "y":
            # baseline模式：处理多个文件并合并
            input_files = [dev_input_file, train_input_file]
            output_files = [dev_baseline_output, train_baseline_output]
            process_and_merge_tables_json_raw(input_files, output_files, merged_baseline_output)
        else:
            # 先运行测试函数
            print("=== 开始测试模式 ===")
            print("请选择要测试的文件：")
            print("1. dev_tables.json")
            print("2. train_tables.json")
            file_choice = input("请输入选择 (1 或 2): ")
            
            if file_choice == "1":
                test_single_database(dev_input_file, db_index=0)
            elif file_choice == "2":
                test_single_database(train_input_file, db_index=0)
            else:
                print("无效选择，默认测试 dev 文件")
                test_single_database(dev_input_file, db_index=0)
            
            # 询问是否继续处理全部数据
            print("\n=== 测试完成 ===")
            user_input = input("是否继续处理全部数据库？(y/n): ")
            if user_input.lower() == "y":
                # 处理多个文件并合并
                input_files = [dev_input_file, train_input_file]
                output_files = [dev_output_file, train_output_file]
                process_and_merge_tables_json(input_files, output_files, merged_output_file)
            else:
                print("程序结束")
    print(f"总消耗token: {total_consumption}")
