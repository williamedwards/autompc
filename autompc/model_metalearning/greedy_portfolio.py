import os
import pickle
import numpy as np
import pandas as pd

from autompc.model_metalearning.meta_utils import load_data, load_cfg, load_matrix
import autompc.model_metalearning.portfolio_util

data_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_data'
cfg_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_cfg'
matrix_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_matrix'
portfolio_path = '/home/baoyu/baoyul2/autompc/autompc/model_metalearning/meta_portfolio'

def build_portfolio(
    matrix: pd.DataFrame,
    portfolio_size: int,
    names: list,
    # rng: np.random.RandomState
):
    portfolio = []
    configurations = []
    for cfg_name in names:
        cfg = load_cfg(cfg_path, cfg_name)
        configurations.append(cfg)
    mean_scores = list(matrix.mean(axis=0))
    
    # Construct Portfolio
    while len(portfolio) < portfolio_size:
        min_index = mean_scores.index(min(mean_scores))
        print(names[min_index])
        portfolio.append(configurations[min_index])
        
        mean_scores.pop(min_index)
        names.pop(min_index)
        configurations.pop(min_index)
    
    # Save portfolio
    output_file_name = os.path.join(portfolio_path, 'portfolio.pkl')
    print("Dumping to ", output_file_name)
    with open(output_file_name, 'wb') as fh:
        pickle.dump(portfolio, fh)
    
    return portfolio
        
if __name__ == "__main__":
    names = ["HalfCheetah-v2", "HalfCheetahSmall-v2", "ReacherSmall-v2", "SwimmerSmall-v2"]
    matrix = load_matrix(matrix_path)
    portfolio = build_portfolio(matrix=matrix, portfolio_size=2, names=names)
    print(portfolio)
    