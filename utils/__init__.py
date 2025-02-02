import os
import random
import numpy as np
import torch
from typing import Dict


def seed_everything(seed: int) -> None:
    print(f'Initialize random seed to {seed}')
    assert seed is not None, 'Random seed cannot be None.'
   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_file_postfix(args) -> str:

    postfix = f"{args.dataset.task}_{args.context.naive}_{args.context.pos_context}_{args.context.neg_context}_{args.dataset.n_shots}"

    return postfix


def get_base_wo_task(args) -> str:
    
    model_name_for_dir = args.model.name.replace('/', '-')
    
    return os.path.join(args.dir.output_dir, f'{model_name_for_dir}-{args.dataset.name}/{args.dataset.subset}_samples/{str(args.seed)}')


def get_paths(args, task:str, target_task:str=None) -> Dict:
    # task : currently running code (method given in args)
    # target_task : specified in the code then ignore task from args

    path_dict = {}

    if target_task is not None:
        base_dir = f'{get_base_wo_task(args)}/{target_task}'
    else:
        base_dir = f'{get_base_wo_task(args)}/{args.dataset.task}'
    
    path_dict['base_dir'] = base_dir

    path_dict['log_dir'] = os.path.join(base_dir, 'logs')
    path_dict['offload_dir'] = os.path.join(args.dir.offload_dir, args.model.name)

    path_dict['gen_dir'] = os.path.join(base_dir, 'generation')
    path_dict['eval_dir'] = os.path.join(base_dir, 'evaluation')

    if task == 'summarization':
        path_dict['summ_dir'] = get_base_wo_task(args)
    else:
        # for non- summarization, generation, evaluation tasks
        if task != 'generation' or task != 'evaluation':
            path_dict[f'{task}_dir'] = os.path.join(base_dir, task)

    # create directories if task is not summarization    
    if task != 'summarization':
        # create directories
        for key, path in path_dict.items():
            if 'offload' in key:
                continue
            makedirs(path=path)
    
    return path_dict


def retriever_path(dataname:str, retriever_name:str) -> str:
    return f'./results/RAG_results/contriever/{retriever_name}_{dataname}'


def get_retriever_path(dataname:str, retriever_name:str, split:str, postfix:str='') -> str:

    if dataname == 'nq':
        if split == 'train':
            f_name = f'NQ-open.train-1000{postfix}.json'
        else:
            # DEV
            f_name = f'NQ-open.dev{postfix}.json'
    elif dataname == 'triviaqa':
        if split == 'train':
            f_name = f'unfiltered-web-train-1000{postfix}.json'
        else:
            # DEV
            f_name = f'unfiltered-web-dev{postfix}.json'
    elif dataname == 'popqa':
        f_name = f'popqa-dev{postfix}.json'
    elif 'mrqa' in dataname:
        if split == 'train':
            f_name = f'{dataname}-train{postfix}.json'
        else:
            f_name = f"{dataname}-dev{postfix}.json"
    else:
        raise ValueError(f'{dataname} is not supported')
    
    return os.path.join(retriever_path(dataname=dataname, retriever_name=retriever_name), f_name)


def get_fewshot_path(dataname:str, n_shots:int) -> str:
    base_dir = './dataset/fewshots/'

    if dataname.startswith('mrqa'):
        base_dir += f'{dataname}/'
        f_name = f'{dataname}_train-{n_shots}shots.json'
    
    elif dataname == 'nq':
        base_dir += 'NQ/'
        f_name = f'NQ-open.train-{n_shots}shots.json'

    elif dataname == 'triviaqa':
        base_dir += 'triviaqa/'
        f_name = f'unfiltered-web-train-{n_shots}shots.json'
    elif dataname == 'popqa':
        base_dir += 'popqa/'
        f_name = f'popqa-train-{n_shots}shots.json'
    # create dir if not exists
    makedirs(path=base_dir)

    return os.path.join(base_dir, f_name)


def makedirs(path: str) -> None:
    if os.path.isdir(path) is False:
        os.makedirs(path, exist_ok=True)


def check_valid_model_dataset(args, task:str) -> None:
    # check valid model, dataset
    
    assert args.model.name in args.candidates.model_name, f'{args.model.name} not in {args.candidates.model_name}'
    assert args.dataset.name in args.candidates.dataset_name, f'{args.dataset.name} not in {args.candidates.dataset_name}'
    