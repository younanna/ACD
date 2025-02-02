from typing import List
from model import load_model_and_tokenizer
from time import time
from omegaconf import DictConfig, OmegaConf
from dataset import get_data_class
from dataset.dataset_utils import get_template
from generation.gen_utils import get_data_config, dataset_to_list_input_data
from tqdm import tqdm
from transformers import GenerationConfig
import torch
import os
import torch
import json
from typing import List
import hydra
import pandas as pd
from utils import seed_everything, get_paths, check_valid_model_dataset, get_file_postfix


def get_input_data(data_cls, c_config:DictConfig, task:str, model_name:str, n_shots:int=0) -> List[str]:
    
    template = get_template()

    is_base_model = 'chat' not in model_name and 'Instruct' not in model_name
    use_fewshots = n_shots > 0 # and is_base_model
    
    if use_fewshots:
        # TODO : might change prompt name later
        if data_cls.name in ['triviaqa', 'nq', 'popqa']:
            data_cls.prompt = 'gold'
        else:
            data_cls.prompt = c_config.pos_context
        fewshot_dict = data_cls.get_few_shots(task=task, n_shots=n_shots, is_base_model=is_base_model, template=template, model_name=model_name)
    else:
        fewshot_dict = None
    
    input_dataset, input_data = dataset_to_list_input_data(data_cls=data_cls, prompt=c_config.naive, template=template, task=task, use_fewshots=use_fewshots, fewshot_dict=fewshot_dict, is_base_model=is_base_model, model_name=model_name)

    input_dataset_pos, input_data_pos = dataset_to_list_input_data(data_cls=data_cls, prompt=c_config.pos_context, template=template, task=task, use_fewshots=use_fewshots, fewshot_dict=fewshot_dict, is_base_model=is_base_model, model_name=model_name)

    if input_data_pos is not None:
        return input_dataset_pos, input_data, input_data_pos
    else:
        return input_dataset, input_data, input_data_pos


def get_gen_configs(tokenizer, model_name:str, max_new_tokens:int) -> GenerationConfig:
    generation_config = dict(
        max_new_tokens = max_new_tokens,
        pad_token_id = tokenizer.pad_token_id
    )
    generation_config = GenerationConfig(**generation_config)
    
    return generation_config


def run_generation(args: DictConfig, paths:dict=None, task:str='generation') -> None:

    # LOAD DATASET
    start_time = time()
    data_config = get_data_config(args=args)
    data_cls = get_data_class(name=data_config['dataset_name'], cache_dir=data_config['data_cache_dir'], subset_size=data_config['subset_size'], batch_size=data_config['batch_size'], seed=args.seed, retriever_name=args.model.retriever, n_shots=args.dataset.n_shots)
    c_config = args.context
    input_dataset, input_data, input_data_pos = get_input_data(data_cls=data_cls, c_config=c_config, task=args.dataset.task, model_name=args.model.name, n_shots=args.dataset.n_shots)
    end_time = time()
    print(f'Done loading dataset. Total loading time : {end_time - start_time} sec.')

    # LOAD MODEL
    start_time = time()
    model, tokenizer = load_model_and_tokenizer(model_name=args.model.name, offload_dir=args.dir.offload_dir, cache_dir=args.dir.model_cache)
    model.gen_model.resize_token_embeddings(len(tokenizer))
    model.gen_model.config.pad_token = tokenizer.pad_token
    end_time = time()
    print(f'Done loading HF model. Total loading time : {end_time - start_time} sec.')
    
    # ------------------------------------- #
    # HYPERPARAM

    if args.dataset.n_shots > 0 and (args.model.name == 'meta-llama/Meta-Llama-3-8B' or args.model.name == 'meta-llama/Llama-2-7b-hf' or args.model.name == 'meta-llama/Llama-2-13b-hf' or args.model.name == 'mistralai/Mistral-7B-v0.1'):
        max_new_tokens = 15
    else:
        max_new_tokens = args.generation.max_new_tokens
    
    generation_config = get_gen_configs(tokenizer=tokenizer, model_name=args.model.name, max_new_tokens=max_new_tokens)
    batch_size = args.generation.batch_size
        
    # ------------------------------------- #
    # GENERATION
    greedy_outputs = []

    with torch.no_grad():
        for i in tqdm(range(0, len(input_data), batch_size)):
            
            # batched input
            i_naive = input_data[i:i+batch_size]
            i_pos = input_data_pos[i:i+batch_size] if input_data_pos is not None else None
            
            # tokenized input -> input_ids
            tokenized_inputs = tokenizer(i_naive, return_tensors="pt", padding=True, truncation=False).to(model.device).input_ids
            tokenized_inputs_pos = tokenizer(i_pos, return_tensors="pt", padding=True, truncation=False).to(model.device).input_ids if i_pos is not None else None
            outputs_ids = model.generate_contrast(tokenized_inputs, tokenized_inputs_pos, num_beams=1, do_sample=False, generation_config=generation_config, mode=args.dataset.task)
            
            outputs_ids = [outputs_ids[i][len(tokenized_inputs[i]):] for i in range(tokenized_inputs.shape[0])]
            outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
            
            greedy_outputs += outputs

    # put generated outputs to sequences
    sequences = []

    for i, sample in enumerate(tqdm(input_dataset)):
        curr_seq = {
            'id': sample['id'],
            'prompt': input_data[i],
            'question': sample['question'],
            'answers': sample['answers'],
            'most_likely_generation': greedy_outputs[i],
            'hasanswer': sample['hasanswer']
        }
        sequences.append(curr_seq)
    
    return sequences


@hydra.main(config_path='../configs/', config_name='acd.yaml')
def main(args: DictConfig) -> None:
    
    print(OmegaConf.to_yaml(args))
    
    # fix random seed
    seed_everything(args.seed)
    # check valid model, dataset
    check_valid_model_dataset(args=args, task='generation')
    # get paths
    paths:dict = get_paths(args=args, task='generation')
    # output file
    
    f_postfix = get_file_postfix(args=args)
    output_file = os.path.join(paths['gen_dir'], f'genr_{f_postfix}.pkl')
    
    # ``save`` config
    with open(os.path.join(paths['log_dir'], f'config_{f_postfix}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(args))
    
    sequences = run_generation(args=args, paths=paths)

    try:
        pd.to_pickle(sequences, output_file)
    except:
        print(f'Unable to save results to : {output_file}')

    visualize_len = 50 if args.dataset.subset > 50 or args.dataset.subset == -1 else args.dataset.subset
    filtered_sequences = sequences[:visualize_len]
    output_file = os.path.join(paths['gen_dir'], f'smpl_{f_postfix}.jsonl')
    print(f'Save readable samples to : {output_file}')
    with open(output_file, 'w') as f:
        for seq in filtered_sequences:
            f.write(json.dumps(seq) + '\n')


if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f'Total run time : {end_time - start_time} sec.')
    