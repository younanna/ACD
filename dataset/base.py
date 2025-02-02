from datasets import Dataset
import pandas as pd
import os
from utils import get_retriever_path
import json

WORD_LIMIT = 100

class BaseDatasetClass:
    def __init__(self,
                name:str,
                cache_dir:str,
                subset_size:float=1.0,
                batch_size:int=16,
                seed:int=1234,
                split:str='validation',
                retriever_name:str=None
                ) -> None:
        
        self.name = name
        self.cache_dir = cache_dir
        self.split = split
        self.subset_size = subset_size
        self.batch_size = batch_size
        self.seed = seed
        self.prompt = 'none'
        self.retriever_name = retriever_name


    def get_prompt(self, task:str, with_context:bool, is_base_model:bool=True, fewshot:bool=False) -> dict:
        
        prompt = ""
        prefix = ""
    
        if with_context:
            prompt = "Context: {context}\nQuestion: {question}\n"
            
        else:
            prompt = "Question: {question}\n"
        
        if fewshot:
            if is_base_model:
                prompt += "Answer: {answer}\n"
            prefix = 'Answer the following questions:\n\n'
        else:
            prefix = 'Answer the following question:\n\n'
    
        return {'prompt': prompt, 'prefix': prefix}

    
    def get_dataset(self) -> Dataset:
        pass
    
    
    def get_dataset_for_generation(self, task:str, with_context:bool, is_base_model:bool, template:str, model_name:str, fewshots:str=None) -> Dataset:
        
        data = self.get_dataset()
        prompt_dict = self.get_prompt(task=task, with_context=with_context, fewshot=False)
        prompt = prompt_dict['prompt']
        
        if with_context:
            data = data.map(lambda sample: {'prompt': prompt.format(context=sample['context'], question=sample['question'])})
        else:
            data = data.map(lambda sample: {'prompt': prompt.format(question=sample['question'])})

        # FEWSHOTS
        if fewshots is not None:
            newline = '\n' if fewshots[-2:] != '\n\n' else ''
            
            if is_base_model:
                
                data = data.map(lambda sample: {'prompt': fewshots + newline + sample['prompt']})
            else:
                if model_name == 'meta-llama/Meta-Llama-3-8B-Instruct':
                    bos = '<|begin_of_text|>'
                elif model_name == 'meta-llama/Llama-2-7b-chat-hf':
                    bos = '<s>'
                else:
                    raise ValueError(f"Model {model_name} is not supported")
                
                template = template.replace(bos, '')
                
                data = data.map(lambda sample: {'prompt': fewshots + newline + template.format(sample['prompt'])})
                
        else:
            # without fewshots
            data = data.map(lambda sample: {'prompt': prompt_dict['prefix'] + '\n' + sample['prompt']})
        
        return data

    def _get_few_shots(self, data:list, task:str, n_shots:int, is_base_model:bool, template:str, model_name:str) -> dict:
        '''
            data: list[dict(['id', 'question', 'answers', 'context', 'hasanswer'])]
        '''
        # append "...\n"
        for i in range(len(data)):
            data[i]['context'] = data[i]['context'] + "..." if data[i]['context'][-1] != '.' else data[i]['context']
            
        # to Dataset
        data:Dataset = Dataset.from_pandas(pd.DataFrame(data))

        def _data_to_few_shots(with_context:bool) -> str:
            
            prompt_dict = self.get_prompt(task=task, with_context=with_context, fewshot=True, is_base_model=is_base_model)
            prompt = prompt_dict['prompt']

            if with_context:
                fs_data:Dataset = data.map(lambda sample: {'prompt': prompt.format(context=sample['context'], question=sample['question'], answer=sample['answers'][0])})
                fs_list = [prompt_dict['prefix']]
                fs_list += [sample['prompt'] for sample in fs_data]   
            else:
                fs_data:Dataset = data.map(lambda sample: {'prompt': prompt.format(question=sample['question'], answer=sample['answers'][0])})
                fs_list = [prompt_dict['prefix']]
                fs_list += [sample['prompt'] for sample in fs_data]
            
            return "\n".join(fs_list) + '\n' if fs_list[-1][-1] != '\n' else "\n".join(fs_list)
        
        few_shots_dict = {'with_context': '', 'without_context': ''}

        few_shots_dict['with_context'] = _data_to_few_shots(with_context=True)
        few_shots_dict['without_context'] = _data_to_few_shots(with_context=False)
        
        return few_shots_dict


class RetrievedContext(BaseDatasetClass):

    def __init__(self,
                name:str,
                cache_dir:str,
                subset_size:float=1.0,
                batch_size:int=16,
                seed:int=1234,
                split:str='validation',
                retriever_name:str=None
                ) -> None:

        super().__init__(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed, split=split, retriever_name=retriever_name)


    def _get_context(self, data, context_type:str=None, split:str=None) -> dict:
        # id2context = {'id': {'ctx': str, 'hasanswer': bool}, ...}
        if context_type is None:
            context_type = self.prompt
        
        if context_type.startswith('contriever'):
            # '{retriever_name}-{mode}'
            mode = context_type.split("-")[-1]
            
            id2context = self._get_contriever(data=data, mode=mode, split=split)
            
        else:
            raise ValueError(f"Context type {context_type} is not supported")
        
        return id2context


    def _get_contriever(self, data, mode:str='', split:str=None) -> dict:
        id2context = {}
        if split is None:
            contriever_path = get_retriever_path(dataname=self.name, retriever_name=self.retriever_name, split=self.split)
        else:
            contriever_path = get_retriever_path(dataname=self.name, retriever_name=self.retriever_name, split=split)    
        if not os.path.exists(contriever_path):
            raise FileNotFoundError(f"Contriever path {contriever_path} does not exist")

        with open(contriever_path, 'r') as f:
            for line in f:
                c_data = json.loads(line)
                top_i = self._get_top_i(c_data, mode=mode)
                # data['ctxs']:list -> retrieved contexts sorted in descending order of contriever score 
                curr_context = c_data['ctxs'][top_i]['text']
                curr_context = curr_context + " ..." if curr_context[-1] != '.' else curr_context

                curr_data = {'ctx': curr_context, 'hasanswer': c_data['ctxs'][top_i]['hasanswer']}
                id2context[c_data['id']] = curr_data
        
        return id2context


    def _get_top_i(self, c_data:dict, mode:str) -> int:
        
        # pos == top-1 | neg == top-100
        
        if mode == 'pos':
            top_i = 0
        elif mode == 'gold':
            hasanswer = [c['hasanswer'] for c in c_data['ctxs']]
            if sum(hasanswer) == 0:
                top_i = 0
            else:
                # get hasanswer index closest to index 0
                top_i = hasanswer.index(True)
        elif mode == 'noanswer':
            # get hasanswer == 0 index closest to index 0
            hasanswer = [c['hasanswer'] for c in c_data['ctxs']]
            if sum(hasanswer) == len(hasanswer):
                top_i = -1
            else:
                top_i = hasanswer.index(False)
        else:
            raise ValueError(f"Mode {mode} is not supported")
        
        return top_i
    