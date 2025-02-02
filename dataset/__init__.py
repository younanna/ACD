from datasets import Dataset
from dataset.custom_dataset import CustomDataset
from dataset.popqa import PopQA
from dataset.nq_swap import NQSwap


def get_data_class(name:str, cache_dir:str, subset_size:float, batch_size:int, seed:int, retriever_name:str, n_shots:int=5)-> Dataset:

    if name in ['triviaqa', 'nq']:
        data_class = CustomDataset(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed, retriever_name=retriever_name)
    elif name == 'popqa':
        data_class = PopQA(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed, retriever_name=retriever_name, n_shots=n_shots)
    elif name == 'nqswap':
        data_class = NQSwap(name=name, cache_dir=cache_dir, subset_size=subset_size, batch_size=batch_size, seed=seed)
    else:
        raise NotImplementedError(f'{name} is not implemented yet.')
    
    return data_class

