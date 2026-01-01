Schema-side enhancement:
'''
python BIRD_Data\preprocess\get_corpus.py
'''

Online query enhancement:
'''
python question_generation.py
python table_retrieval.py
'''
training data prepare:
'''
python BIRD_Data\preprocess\data_construction.py
'''
Column pruning:
'''
python schema_construction_bird.py # For BIRD
python schema_construction_spyder.py # For Spider
'''

Generation & evaluation:
'''
python SQL_generation.py # For BIRD
python SQL_generation_spyder.py # For Spider
sh BIRD_Data\llm\run\run_evaluation_ves.sh
'''
You may need to download dataset data under BIRD_Data and configure your API :)