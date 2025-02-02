from datasets import Dataset
import os
from dataset.base import RetrievedContext
from dataset.dataset_utils import save_json, load_json
from utils import get_fewshot_path
import datasets 
import ast
import random

class PopQA(RetrievedContext):
    def __init__(self,
                name:str,
                cache_dir:str,
                subset_size:float=1.0,
                batch_size:int=16,
                seed:int=1234,
                split:str='validation',
                retriever_name:str=None,
                n_shots:int=5
                ) -> None:
        
        super().__init__(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed, split=split, retriever_name=retriever_name)

        self.n_shots = n_shots

    def _extract_sample_for_each_prop(self, data:Dataset) -> dict:
        # dict_keys(['occupation', 'place of birth', 'genre', 'father', 'country', 'producer', 'director', 'capital of', 'screenwriter', 'composer', 'color', 'religion', 'sport', 'author', 'mother', 'capital'])
        prop2id = {}
        for sample in data:
            prop = sample['prop']
            if prop not in prop2id:
                prop2id[prop] = []
            prop2id[prop].append(sample['id'])

        # randomly select 1 sample from each prop
        # and save its id
        rand_ids = []
        for prop_ids in prop2id.values():
            # randomly select 1 id
            rand_ids.append(random.choice(prop_ids))
        
        return rand_ids


    def _load_dataset(self, split:str=None) -> Dataset:
        
        if split is None:
            split = self.split
        
        data = datasets.load_dataset("akariasai/PopQA", split='test', cache_dir=self.cache_dir)

        if split == 'full_test':
            return data
        
        if self.n_shots > 0:
            fewshot_path = get_fewshot_path(dataname=self.name, n_shots=self.n_shots)
            fewshot_data = load_json(fewshot_path)
            train_id = [x['id'] for x in fewshot_data]
        else:
            train_id = []
        
        if split == 'validation':
            data = data.filter(lambda x: x['id'] not in train_id)
        elif split == 'train':
            # FEW SHOTS (5 shots)
            data = data.filter(lambda x: x['id'] in train_id)
        
        return data

    
    def _add_context(self, data:Dataset, context_type:str=None, split:str=None) -> Dataset:
        # original: {'id', 'question', 'answers'}
        # after reformatting:
        #   dict_keys(['question', 'id', 'answers', 'context'])
        if context_type is None:
            context_type = self.prompt
        # 'context'
        if  context_type == 'noContext':
            data = data.map(lambda x: {'context': ''}, batched=False)
            data = data.map(lambda x: {'hasanswer': False}, batched=False)
        else:
            id2context = self._get_context(data, context_type, split=split)
            data = data.map(lambda x: {'context': id2context[x['id']]['ctx']}, batched=False)
            data = data.map(lambda x: {'hasanswer': id2context[x['id']]['hasanswer']}, batched=False)
        
        return data


    def _reformat_data(self, data) -> Dataset:
        
        data = data.map(lambda x: {'answers': ast.literal_eval(x['possible_answers'])}, batched=False)
        
        item_keys = ['id', 'question', 'answers']
        data = data.remove_columns([k for k in data.column_names if k not in item_keys])
        return data
        

    def _clear_dataset(self, data:Dataset, clean_context:bool=False) -> Dataset:
        
        # remove duplicate samples 
        id_list = []
        
        def _filter_unique_samples(sample):
            sample_id = sample['id']
            if sample_id not in id_list:
                id_list.append(sample_id)
                return True
            else:
                return False
        
        data = data.filter(_filter_unique_samples)
        print(f"[Unique] Num data samples: {len(data)}")

        if clean_context:
            data = data.filter(lambda sample: len(sample['context']) > 0)
        
        # remove empty samples
        data = data.filter(lambda sample: len(sample['answers']) > 0 and len(sample['question']) > 0)
        
        return data


    def get_dataset(self, add_context:bool=True) -> Dataset:
        
        data = self._load_dataset()
        print(f"Num data samples: {len(data)}")
        
        data = self._reformat_data(data)
        
        data = self._clear_dataset(data)
        print(f"After clear: {len(data)}")
        
        if add_context:
            data = self._add_context(data)
            print(f"After add contexts: {len(data)}")
        
        orig_size = len(data)
        if self.subset_size > 1.0:
            data = data.train_test_split(test_size=orig_size-self.subset_size, seed=self.seed, shuffle=False)['train']

        print(f"Final data size: {len(data)} | Full data size: {orig_size}")
        
        return data


    def _check_hasanswer(self, data:Dataset) -> Dataset:

        def _check_answer(sample):
            answers = sample['answers']
            context = sample['context']
            for answer in answers:
                if answer.lower() in context.lower():
                    return True
            
            return False
        
        data = data.map(lambda x: {'hasanswer': _check_answer(x)}, batched=False)
        return data
    

    def get_few_shots(self, task:str, n_shots:int, is_base_model:bool, template:str, model_name:str) -> list:
        
        # if custom fewshot exists
        fewshot_path = get_fewshot_path(dataname=self.name, n_shots=n_shots)

        if os.path.exists(fewshot_path):
            print(f"Fewshot path {fewshot_path} exists")
            sequences:list[dict] = load_json(fewshot_path)
            
        else:
            data = self._load_dataset(split='full_test')
            print(f"[FEWSHOT] # samples: {len(data)}")
            data = self._reformat_data(data)
            data = self._clear_dataset(data)
            print(f"[FEWSHOT] After clear: {len(data)}")
            data = self._add_context(data, context_type=f'{self.retriever_name}-gold', split='train')
            print(f"[FEWSHOT] After reformat: {len(data)}")

            # check hasanswer
            data = self._check_hasanswer(data)
            print(f"[FEWSHOT] HAS ANSWER: {len(data)}")
            
            new_data = data.filter(lambda x: x['hasanswer'] == True)
            if len(new_data) > n_shots:
                data = new_data
            
            orig_size = len(data)
            if orig_size > n_shots:
                data = data.train_test_split(test_size=orig_size-n_shots, seed=self.seed, shuffle=False)['train']
            print(f"[FEWSHOT] Few shots: {len(data)}")
            sequences = []
            for i, sample in enumerate(data):
                sequences.append({
                    'id': sample['id'],
                    'question': sample['question'],
                    'answers': sample['answers'],
                    'context': sample['context'],
                    'hasanswer': sample['hasanswer']
                })
            
        save_json(path=fewshot_path, data=sequences)

        fewshot_data:dict = self._get_few_shots(data=sequences, n_shots=n_shots, task=task, is_base_model=is_base_model, template=template, model_name=model_name)
        
        return fewshot_data