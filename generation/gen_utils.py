from typing import List, Dict
from omegaconf import DictConfig


def get_data_config(args:DictConfig):
    if args.dataset.name in ['triviaqa', 'nq']:
        data_config = dict(dataset_name=args.dataset.name, prompt=args.context.naive, data_cache_dir=args.dir.custom_data_dir, subset_size=args.dataset.subset, batch_size=args.generation.batch_size)
    else:
        # data_cache_dir = huggingface dir
        data_config = dict(dataset_name=args.dataset.name, prompt=args.context.naive, data_cache_dir=args.dir.hf_data_dir, subset_size=args.dataset.subset, batch_size=args.generation.batch_size)

    return data_config


def dataset_to_list_input_data(data_cls, prompt:str, template:str, task:str, use_fewshots:bool, fewshot_dict:dict, is_base_model:bool, model_name:str):
    
    if prompt == 'none':
        return None, None
    
    if prompt == 'noContext':
        with_context = False
    else:
        with_context = True
    
    data_cls.prompt = prompt

    if is_base_model:
        if use_fewshots:
            input_dataset = data_cls.get_dataset_for_generation(task=task, with_context=with_context, fewshots=fewshot_dict['with_context'] if with_context else fewshot_dict['without_context'], is_base_model=is_base_model, template=template, model_name=model_name)
        else:
            input_dataset = data_cls.get_dataset_for_generation(task=task, with_context=with_context, is_base_model=is_base_model, template=template, model_name=model_name)
        input_data = [template.format(sample['prompt']) for sample in input_dataset]
    else:
        # chat or instruct models
        if use_fewshots:
            input_dataset = data_cls.get_dataset_for_generation(task=task, with_context=with_context, fewshots=fewshot_dict['with_context'] if with_context else fewshot_dict['without_context'], is_base_model=is_base_model, template=template, model_name=model_name)
        else:
            input_dataset = data_cls.get_dataset_for_generation(task=task, with_context=with_context, is_base_model=is_base_model, template=template, model_name=model_name)
        input_data = [sample['prompt'] for sample in input_dataset]
    
    return input_dataset, input_data