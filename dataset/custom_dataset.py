from datasets import Dataset
import os
import pandas as pd
from dataset.base import RetrievedContext
import json
from dataset.dataset_utils import save_json, load_json, get_nq_fewshot_samples
from utils import get_fewshot_path


class CustomDataset(RetrievedContext):
    def __init__(self,
                name:str,
                cache_dir:str,
                subset_size:float=1.0,
                batch_size:int=16,
                seed:int=1234,
                split:str='validation',
                retriever_name:str=None,
                ) -> None:

        super().__init__(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed, split=split, retriever_name=retriever_name)
    
    
    def _load_dataset_path(self, split:str) -> str:
        # PATHS 
        if self.name == 'nq':    
            if split == 'validation':
                f_name = 'NQ/NQ-open.dev.json'
            elif split == 'train':
                f_name = 'NQ/NQ-open.train-1000.json'
            else:
                raise ValueError(f"Split {split} is not supported")
        elif self.name == 'triviaqa':
            if split == 'validation':
                f_name = 'triviaqa/unfiltered-web-dev.json'
            elif split == 'train':
                f_name = 'triviaqa/unfiltered-web-train-1000.json'
            else:
                raise ValueError(f"Split {split} is not supported")
        else:
            raise ValueError(f"Dataset {self.name} is not supported")
        return f_name


    def _load_dataset(self, split:str=None) -> Dataset:
        
        if split is None:
            split = self.split
        
        f_name = self._load_dataset_path(split=split)
        
        with open(os.path.join(self.cache_dir, f_name)) as f:
            data = json.load(f)

        # data:list[dict] -> Dataset
        data = Dataset.from_pandas(pd.DataFrame(data))

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
        
        return data


    def get_dataset(self, add_context:bool=True) -> Dataset:
        
        data = self._load_dataset()
        print(f"Num data samples: {len(data)}")
        
        data = self._clear_dataset(data)
        print(f"After clear: {len(data)}")
        
        data = self._reformat_data(data)
        if add_context:
            data = self._add_context(data)
        print(f"After reformat: {len(data)}")

        orig_size = len(data)
        if self.subset_size > 1.0:
            # TODO: if x < self.subset_size exists, load x samples data and use it
            data = data.train_test_split(test_size=orig_size-self.subset_size, seed=self.seed, shuffle=False)['train']

        print(f"Final data size: {len(data)} | Full data size: {orig_size}")

        return data


    def get_few_shots(self, task:str, n_shots:int, is_base_model:bool, template:str, model_name:str) -> list:
        
        # if custom fewshot exists
        fewshot_path = get_fewshot_path(dataname=self.name, n_shots=n_shots)

        if os.path.exists(fewshot_path):
            print(f"Fewshot path {fewshot_path} exists")
            sequences:list[dict] = load_json(fewshot_path)
            # : list[ {'id', 'question', 'answers', 'context', 'hasanswer'} ]
            
        else:
            
            if self.name == 'nq':
                # GOLD context exists for NQ
                sequences:list[dict] = get_nq_fewshot_samples(cache_dir=self.cache_dir, split='train', n_shots=n_shots)
            
            else:
                # GOLD context does not exist for triviaqa
                data = self._load_dataset(split='train')
                print(f"[FEWSHOT] # samples: {len(data)}")
                
                data = self._clear_dataset(data)
                print(f"[FEWSHOT] After clear: {len(data)}")
                data = self._reformat_data(data)
                data = self._add_context(data, context_type=f'{self.retriever_name}-gold', split='train')
                print(f"[FEWSHOT] After reformat: {len(data)}")

                data = data.filter(lambda x: x['hasanswer'] == True)
                print(f"[FEWSHOT] HAS ANSWER: {len(data)}")

                orig_size = len(data)
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
            
            save_json(path=fewshot_path, data=sequences)

        fewshot_data:dict = self._get_few_shots(data=sequences, n_shots=n_shots, task=task, is_base_model=is_base_model, template=template, model_name=model_name)
        
        return fewshot_data