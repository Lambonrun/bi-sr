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
    Parse the prediction string, extract database and table.
    For example: "database: perpetrator, table: perpetrator" -> ("perpetrator", "perpetrator")
    """
    # Use regular expression to match "database: value, table: value" format, ignore case and extra spaces
    match = re.search(r"database: (.*?), table: (.*?),", pred_str, re.IGNORECASE)
    if match:
        db_id = match.group(1).strip()
        table = match.group(2).strip()
        return db_id, table
    
    # If regex matching fails, print a warning and return None
    print(f"Warning: Unable to parse prediction string: '{pred_str}'")
    return None, None

def load_corpus(file_path: str) -> list[str]:

    """
    Read a file containing a list of JSON objects and convert each object to a descriptive string.

    Parameters:
    - file_path (str): Path to the JSON file.

    Returns:
    - list[str]: List of descriptive strings forming the corpus.
    """
    corpus = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f) # data is a list
            
            # Iterate through each JSON object (dictionary) in the list
            for item in data:
                # Convert key-value pairs to "key: value" format
                parts = [f"{key}: {value}" for key, value in item.items()]
                
                # Join them with commas and spaces into a complete string
                text_representation = ", ".join(parts)
                
                corpus.append(text_representation)
        print("corpus length: ", len(corpus))
        print("corpus example: ", corpus[0])
                
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
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
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: File '{file_path}' is not valid JSON format.")
        
    return questions, answers

def find_most_similar(query: str, corpus_embeddings: torch.Tensor, model: SentenceTransformer, top_k: int = 5):
    """
    Find the top_k most similar sentences to a query statement in a given corpus vector.

    Parameters:
    - query (str): The sentence you want to query.
    - corpus_embeddings (torch.Tensor): Pre-computed corpus vector matrix.
    - model (SentenceTransformer): The sentence vector model used.
    - top_k (int): Number of most similar sentences to return.

    Returns:
    - a list of tuples: Each tuple contains (similarity score, index of sentence in corpus).
    """
    # 1. Encode the query statement as a vector
    query_embedding = model.encode(query, convert_to_tensor=True, device="cuda")

    # 2. Use util.cos_sim to calculate cosine similarity between query vector and all corpus vectors
    #    This function is very efficient, utilizing PyTorch's parallel computing capabilities
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]

    # 3. Use torch.topk to find the indices and scores of the top_k highest scoring results
    #    This is more efficient than manual sorting and slicing
    top_results = torch.topk(cos_scores, k=min(top_k, len(corpus_embeddings)))
    
    # top_results is a tuple containing (values, indices)
    return zip(top_results[0], top_results[1])

def evaluate(all_predictions: list[list[str]], all_answers: list[dict]):
    """
    Calculate recall rates for prediction results: recall@1, recall@5, recall@10, recall@15.
    Rule: Correct if database is correct and table is in answer, case mismatch is also considered correct.
    
    Parameters:
    - all_predictions (list[list[str]]): List of prediction results for all questions, each sublist contains up to 15 prediction strings.
    - all_answers (list[dict]): Corresponding list of standard answers.

    Returns:
    - dict: Dictionary containing average recall rates for each item.
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
                # Ensure recall@15 is also calculated at the end of the list
                if len(predicted_list) <= 15:
                    recall_scores['recall@15'].append(len(hits) / num_gold_tables)

    # Calculate the average of each recall rate
    final_recalls = {
        key: (sum(scores) / len(scores) if scores else 0)
        for key, scores in recall_scores.items()
    }
    
    return final_recalls

    
if __name__ == "__main__":
    corpus = load_corpus("/BIRD_Data/data/merged_table_descriptions.json")
    questions, answers = load_queries("/BIRD_Data/data/dev_20240627/cache/dev_question_generation.json")
    print("question length: ", len(questions))
    print("answer length: ", len(answers))
    print("question example: ", questions[0])
    print("answer example: ", answers[0])
    

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    
    model = SentenceTransformer("/model_file/Qwen3-Embedding-4B-lora-merged", device=device)

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
        with open("/BIRD_Data/data/dev_20240627/cache/dev_predicted_tables_finetuned.json", "w", encoding="utf-8") as f:
            json.dump({"questions": questions, "predicted_tables": predicted, "answers": answers}, f, ensure_ascii=False, indent=4)
            print("Data saving completed! Saved to /BIRD_Data/data/dev_20240627/cache/dev_predicted_tables_finetuned.json")
        recall_results = evaluate(predicted, answers)
        print("Evaluation results:", recall_results)
