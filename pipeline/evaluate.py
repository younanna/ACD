
import os
from time import time
import hydra
import csv
from omegaconf import DictConfig, OmegaConf
from dataset.dataset_utils import load_pkl, save_pkl
from utils import seed_everything, get_paths, check_valid_model_dataset, get_file_postfix
from evaluation import get_accuracy, get_accuracy_result


@hydra.main(config_path='../configs/', config_name='acd.yaml')
def main(args: DictConfig) -> None:
    
    # fix random seed
    seed_everything(args.seed)
    # check valid model, dataset
    check_valid_model_dataset(args=args, task='evaluation')
    # get paths
    paths:dict = get_paths(args=args, task='generation')
    
    # postfix for output file
    f_postfix = get_file_postfix(args=args)
    
    # save config
    with open(os.path.join(paths['log_dir'], f'config_{f_postfix}.yaml'), 'w') as f:
        f.write(OmegaConf.to_yaml(args))
    
    # output file
    output_file = os.path.join(paths['gen_dir'], f'genr_{f_postfix}.pkl')
    if not os.path.exists(output_file):
        print(f'Generated output does not exist at : {output_file}')
        return
    outputs = load_pkl(output_file)
    
    acc_fname = f'eval_{f_postfix}.pkl'
    print('Start getting accuracy...')
    start_time = time()
    accuracy, cleaned_gen = get_accuracy(outputs=outputs, args=args)
    if cleaned_gen is not None:
        save_pkl(cleaned_gen, os.path.join(paths['gen_dir'], f'clen_{f_postfix}.pkl'))
    
    save_pkl(accuracy, os.path.join(paths['eval_dir'], acc_fname))
    end_time = time()
    print(f'Done getting accuracy. Total time : {end_time - start_time} sec.')
    
    # get accuracy result : aggregated as score
    eval_result:dict = get_accuracy_result(accuracy=accuracy)
    
    # save as csv file
    csv_fname = f'eval_{f_postfix}.csv'
    with open(os.path.join(paths['eval_dir'], csv_fname), 'w') as f:
        w = csv.writer(f)
        # header
        w.writerow(['metric', 'value'])
        for k,v in eval_result.items():
            w.writerow([k, format(round(v, 2), '.2f')])
            print(f"{k:<40} | {format(round(v, 2), '.2f')}")
    print(f"Saved evaluation result : {os.path.join(paths['eval_dir'], csv_fname)}")
    

if __name__ == '__main__':
    start_time = time()
    main()
    end_time = time()
    print(f'Total run time : {end_time - start_time} sec.')
    