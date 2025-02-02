from typing import List
import torch
from transformers import AutoTokenizer
from generation.gen_wrapper import LMWrapper


def _get_device_info() -> dict:
    
    max_memory = {0:"45GiB"}
    device_map = 'auto'
    
    return {'device_map': device_map, 'max_memory': max_memory}


def load_model(model_name:str, offload_dir:str, cache_dir:str, dtype:float=torch.float16) -> LMWrapper: 
    device_info = _get_device_info()
    
    model = LMWrapper(
        model_name_or_path=model_name,
        cache_dir=cache_dir,
        device_map=device_info['device_map'],
        max_memory=device_info['max_memory'],
        offload_folder=offload_dir,
        )
    
    return model


def load_tokenizer(model_name:str, cache_dir:str=None) -> AutoTokenizer:
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.padding_side = 'left'
    
    return tokenizer


def load_model_and_tokenizer(model_name:str, offload_dir:str, cache_dir:str) -> List:
    model = load_model(model_name=model_name, offload_dir=offload_dir, cache_dir=cache_dir)
    tokenizer = load_tokenizer(model_name, cache_dir=cache_dir)
    return model, tokenizer

