import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Dict
from sentence_transformers import SentenceTransformer, util, models
import json
from sql_metadata import Parser
import re

def parse_prediction(pred_str: str):
    """
    解析预测字符串，提取 database 和 table。
    例如: "database: perpetrator, table: perpetrator" -> ("perpetrator", "perpetrator")
    """
    # 使用正则表达式来匹配 "database: value, table: value" 格式，忽略大小写和多余的空格
    match = re.search(r"database: (.*?), table: (.*?),", pred_str, re.IGNORECASE)
    if match:
        db_id = match.group(1).strip()
        table = match.group(2).strip()
        return db_id, table
    
    # 如果正则匹配失败，打印一个警告并返回 None
    print(f"警告: 无法解析预测字符串: '{pred_str}'")
    return None, None

def load_corpus(file_path: str) -> list[str]:

    """
    读取包含JSON对象列表的文件，并将每个对象转换为描述性字符串。

    参数:
    - file_path (str): JSON文件的路径。

    返回:
    - list[str]: 由描述性字符串组成的语料库列表。
    """
    corpus = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # data 是一个列表
            
            # 遍历列表中的每一个JSON对象 (字典)
            for item in data:
                # 将键值对转换为 "key is value" 的形式
                parts = [f"{key}: {value}" for key, value in item.items()]
                
                # 用逗号和空格将它们连接成一个完整的字符串
                text_representation = ", ".join(parts)
                
                corpus.append(text_representation)
        print("corpus length: ", len(corpus))
        print("corpus example: ", corpus[0])
                
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        
    return corpus

def load_queries(file_path: str):
    questions = []
    answers = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            questions = data['questions']
            answers = data['answers']
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到。")
    except json.JSONDecodeError:
        print(f"错误: 文件 '{file_path}' 不是有效的JSON格式。")
        
    return questions, answers

def find_most_similar(query: str, corpus_embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 5):
    """
    在一个给定的语料库向量中，为一个查询语句找到最相似的 top_k 个句子。

    参数:
    - query (str): 你要查询的句子。
    - corpus_embeddings (torch.Tensor): 已经提前计算好的语料库向量矩阵。
    - model (SentenceTransformer): 使用的句向量模型。
    - top_k (int): 返回最相似句子的数量。

    返回:
    - a list of tuples: 每个元组包含 (相似度分数, 句子在语料库中的索引)。
    """
    # 1. 将查询语句编码为向量
    query_embedding = model.encode(query, convert_to_tensor=True, device="cuda")

    # 2. 使用 util.cos_sim 计算查询向量与所有语料库向量的余弦相似度
    #    这个函数非常高效，利用了PyTorch的并行计算能力
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 3. 使用 torch.topk 找到分数最高的 top_k 个结果的索引和分数
    #    这比手动排序然后切片更高效
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus_embeddings)))
    
    # top_results 是一个包含 (values, indices) 的元组
    return zip(top_results[0], top_results[1])

def evaluate(all_predictions: list[list[str]], all_answers: list[dict]):
    """
    计算预测结果的召回率 recall@1, recall@5, recall@10, recall@15.
    规则: database正确且table在answer中即为正确，大小写不匹配也视为正确。
    
    参数:
    - all_predictions (list[list[str]]): 所有问题的预测结果列表，每个子列表包含最多15个预测字符串。
    - all_answers (list[dict]): 对应的标准答案列表。

    返回:
    - dict: 包含各项平均召回率的字典。
    """
    recall_scores = {'recall@1': [], 'recall@5': [], 'recall@10': [], 'recall@15': []}
    
    for predicted_list, gold_answer in zip(all_predictions, all_answers):
        gold_db = gold_answer['db_id'].lower()
        gold_tables = {table.lower() for table in gold_answer['tables']}
        num_gold_tables = len(gold_tables)

        if num_gold_tables == 0:
            continue

        hits = set()
        
        for i, pred_str in enumerate(predicted_list):
            pred_db, pred_table = parse_prediction(pred_str)
            
            if pred_db and pred_table:
                if pred_db.lower() == gold_db and pred_table.lower() in gold_tables:
                    hits.add(pred_table.lower())
            
            k = i + 1
            if k == num_gold_tables:
                recall_scores['recall@1'].append(len(hits) / num_gold_tables)
            elif k == 5:
                recall_scores['recall@5'].append(len(hits) / num_gold_tables)
            elif k == 10:
                recall_scores['recall@10'].append(len(hits) / num_gold_tables)
            elif k == 15:
                # 确保在列表末尾也计算 recall@15
                if len(predicted_list) <= 15:
                    recall_scores['recall@15'].append(len(hits) / num_gold_tables)

    # 计算各项召回率的平均值
    final_recalls = {
        key: (sum(scores) / len(scores) if scores else 0)
        for key, scores in recall_scores.items()
    }
    
    return final_recalls

    
if __name__ == "__main__":
    corpus = load_corpus("/root/autodl-tmp/schlink/BIRD_Data/data/merged_table_descriptions.json")
    questions, answers = load_queries("/root/autodl-tmp/schlink/BIRD_Data/data/dev_20240627/cache/dev_question_generation.json")
    print("question length: ", len(questions))
    print("answer length: ", len(answers))
    print("question example: ", questions[0])
    print("answer example: ", answers[0])
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    
    model = SentenceTransformer("/root/autodl-tmp/model_file/Qwen3-Embedding-4B-lora-merged", device=device)

    corpus_embeddings = model.encode(corpus, convert_to_tensor=True, device=device, show_progress_bar=True, batch_size=128)

    predicted = []
    for i, (question, answer) in enumerate(zip(questions, answers)):
        top_results = find_most_similar(question, corpus_embeddings, model, top_k=15)
        predicted_tables = []
        for score, index in top_results:
            score = score.item()
            index = index.item()
            predicted_tables.append(corpus[index])
        predicted.append(predicted_tables)
        if i % 100 == 0:
            print(f"Processed {i} questions")
    
    if len(predicted) != len(answers):
        print("predicted length: ", len(predicted))
        print("answers length: ", len(answers))
        raise ValueError("predicted length != answers length")
    else:
        with open("/root/autodl-tmp/schlink/BIRD_Data/data/dev_20240627/cache/dev_predicted_tables_finetuned.json", "w", encoding="utf-8") as f:
            json.dump({"questions": questions, "predicted_tables": predicted, "answers": answers}, f, ensure_ascii=False, indent=4)
            print("数据保存完成！保存到 /root/autodl-tmp/schlink/BIRD_Data/data/dev_20240627/cache/dev_predicted_tables_finetuned.json")
        recall_results = evaluate(predicted, answers)
        print("评估结果:", recall_results)
