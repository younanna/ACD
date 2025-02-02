import pandas as pd
from typing import List, Dict
from datasets import Dataset
from evaluation.eval_acc import *


def clean_generation(generation:list) -> List[str]:
    clean_candidates = ['\n\n', '\nQuestion:', '\nContext:']

    cleaned_gen = []

    for gen in generation:
        for cc in clean_candidates:
            if cc in gen:
                gen = gen.split(cc)[0]
        cleaned_gen.append(gen)
    
    return cleaned_gen


def get_accuracy(outputs:List[Dict], args) -> Dict:
    
    # PREPROCESS DATA
    outputs = Dataset.from_pandas(pd.DataFrame(data=outputs))

    generation: List[str] = outputs['most_likely_generation']
    answer = outputs['answers']

    # if answer is List of string, change to List[List[str]]
    if isinstance(answer[0], str):
        answer: List[List[str]] = [[ans] for ans in answer]

    # clean generation (most_likely_generation)
    generation = clean_generation(generation)
    cleaned_gen = generation
    
    # EVALUATE
    result = {'id': outputs['id']}
    result['hasanswer'] = sum(outputs['hasanswer']) / len(outputs)
    # exact match
    result['EM'] = get_exact_match(generations=generation if cleaned_gen is None else cleaned_gen, answers=answer)
    
    return result, cleaned_gen


def get_accuracy_result(accuracy:Dict) -> dict:
    
    # print result in pretty format
    result = pd.DataFrame(data=accuracy)
    result = result.drop(columns='id')
    result = result.mean()
    result = result.to_dict()
    result = {k: v*100 for k, v in result.items()}

    return result

    