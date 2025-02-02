from datasets import load_dataset, Dataset
import random
from dataset.base import BaseDatasetClass

class NQSwap(BaseDatasetClass):
    def __init__(self,
                name:str,
                cache_dir:str,
                subset_size:float=1.0,
                batch_size:int=16,
                seed:int=1234,
                split:str='validation'
                ) -> None:

        super().__init__(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed, split=split)
    

    def _extract_sample_ids(self, data:Dataset) -> dict:
        
        data_ids = []
        for i, sample in enumerate(data):
            if len(sample['org_context'].split()) <= 100:
                data_ids.append(sample['id'])
        
        rand_ids = random.sample(data_ids, 5)

        return rand_ids


    def _load_dataset(self, split:str=None) -> Dataset:
        
        if split is None:
            split = self.split
        
        data = load_dataset("pminervini/NQ-Swap", split='dev', cache_dir=self.cache_dir)
        # dict_keys(['question', 'org_context', 'org_answer', 'sub_context', 'sub_answer', 'id'])
        
        ##### context in <Table> not used #####
        data = data.filter(lambda x: '<Table>' not in x['org_context'])
        print(f"DROP table context: {len(data)}")
        
        # append 'id' as order in ascending order
        id_list = [i for i in range(len(data))]
        data = data.add_column('id', id_list)
        
        # EXTRACT 5 samples for few-shot
        train_id = self._extract_sample_ids(data)

        if split == 'validation':
            data = data.filter(lambda x: x['id'] not in train_id)
        elif split == 'train':
            # FEW SHOTS (5 shots)
            data = data.filter(lambda x: x['id'] in train_id)
        
        return data

    
    def _add_context(self, data:Dataset, split:str=None, fewshot_context_type:str=None) -> Dataset:
        
        if fewshot_context_type is not None:
            context_type = fewshot_context_type
        else:
            context_type = self.prompt

        if  context_type == 'noContext':
            data = data.map(lambda x: {'context': ''}, batched=False)
            data = data.map(lambda x: {'hasanswer': False}, batched=False)
            data = data.map(lambda x: {'answers': x['org_answer']}, batched=False)
        else:
            data = data.map(lambda x: {'context': x[f'{context_type}_context']}, batched=False)
            data = data.map(lambda x: {'hasanswer': True}, batched=False)
            data = data.map(lambda x: {'answers': x[f'{context_type}_answer']}, batched=False)
        
        return data


    def _reformat_data(self, data) -> Dataset:
        # dict_keys(['question', 'answers', 'context', 'id'])
        # remove <p> from context
        data = data.map(lambda x: {'context': x['context'].replace('<P>', '')}, batched=False)
        # reomve </p> from context
        data = data.map(lambda x: {'context': x['context'].replace('</P>', '')}, batched=False)
        # question append '?'
        data = data.map(lambda x: {'question': x['question'] + '?' if x['question'][-1] != '?' else x['question']}, batched=False)
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

        # drop keys except
        item_keys = ['id', 'question', 'answers', 'context', 'hasanswer']
        data = data.remove_columns([k for k in data.column_names if k not in item_keys])
        
        return data


    def get_dataset(self, add_context:bool=True) -> Dataset:
        
        data = self._load_dataset()
        print(f"Num data samples: {len(data)}")
        
        if add_context:
            data = self._add_context(data)
            print(f"After add contexts: {len(data)}")
        
        data = self._reformat_data(data)
        
        data = self._clear_dataset(data)
        print(f"After clear: {len(data)}")
        
        orig_size = len(data)
        if self.subset_size > 1.0:
            data = data.train_test_split(test_size=orig_size-self.subset_size, seed=self.seed, shuffle=False)['train']

        print(f"Final data size: {len(data)} | Full data size: {orig_size}")
        
        return data


    def get_few_shots(self, task:str, n_shots:int, is_base_model:bool, template:str, model_name:str) -> list:
        
        data = self._load_dataset(split='train')
        print(f"[FEWSHOT] # samples: {len(data)}")
        
        if self.prompt == 'noContext':
            data = self._add_context(data, split='train', fewshot_context_type='org')
        else:
            data = self._add_context(data, split='train')
        
        print(f"[FEWSHOT] After reformat: {len(data)}")
        data = self._reformat_data(data)
        data = self._clear_dataset(data)
        print(f"[FEWSHOT] After clear: {len(data)}")
        orig_size = len(data)
        if orig_size > n_shots:
            data = data.train_test_split(test_size=orig_size-n_shots, seed=self.seed, shuffle=False)['train']
        
        print(f"[FEWSHOT] Few shots: {len(data)}")

        # Dataset to list[dict]
        sequences = []
        for i, sample in enumerate(data):
            sequences.append({
                'id': sample['id'],
                'question': sample['question'],
                'answers': sample['answers'],
                'context': sample['context'],
                'hasanswer': sample['hasanswer']
            })

        fewshot_data:dict = self._get_few_shots(data=sequences, n_shots=n_shots, task=task, is_base_model=is_base_model, template=template, model_name=model_name)
        
        return fewshot_data