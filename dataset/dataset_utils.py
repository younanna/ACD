import pickle
import json
import os


def load_pkl(path:str):
    
    with open(path, 'rb') as infile:
        pkl_file = pickle.load(infile)
    
    return pkl_file

def save_pkl(data, path):
    
    with open(path, 'wb') as outfile:
        pickle.dump(data, outfile)


def save_json(data, path):
    
    with open(path, 'w') as outfile:
        json.dump(data, outfile)


def load_json(path):

    with open(path) as infile:
        data = json.load(infile)
    
    return data


def get_template():

    return """{}Answer:"""


def get_nq_fewshot_samples(cache_dir:str, split:str, n_shots:int=5):

    # gold_files = ['nq-train_gold_info.json', 'nq-dev_gold_info.json', 'nq-test_gold_info.json']
    if split == 'validation':
        gold_files = ['nq-test_gold_info.json']
    elif split == 'train':
        gold_files = ['nq-train_gold_info.json', 'nq-dev_gold_info.json']
    
    gold_data = []
    for gold_file in gold_files:
        gold_data += load_json(os.path.join(cache_dir, 'NQ/gold', gold_file))['data']
    
    # get fewshot samples
    fewshot_samples = []
    for i, data in enumerate(gold_data):
        if data['context'] != '':
            ctx_len = len(data['context'].split())
            if ctx_len > 20 and ctx_len < 100 and data['short_answers'][0].lower() in data['context'].lower():
                data['hasanswer'] = True
                data['question'] = data['question'] + '?' if data['question'][-1] != '?' else data['question']
                
                fewshot_samples.append(data)

        if i == n_shots:
            break
    
    sequences = []
    for i, sample in enumerate(fewshot_samples):
        sequences.append({
            'id': sample['example_id'],
            'question': sample['question'],
            'answers': sample['short_answers'],
            'context': sample['context'],
            'hasanswer': sample['hasanswer']
        })

    return sequences

