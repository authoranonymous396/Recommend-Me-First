import pandas as pd 
import numpy as np 
import os
import pickle
import random 
from experimental_setup import generate_config_csvs
import Simulations as sim
import math 

def generate_config_files(percentage_matrix , no_folders ):
    script_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_path,f"Initial_matrix_{percentage_matrix}.csv")
    matrix= pd.read_csv(file_path, index_col=0)
    arr_none = matrix.astype(object)
    arr_none[np.isnan(matrix)] = None

    config = {}
    config['num_users']= len(arr_none)
    config['num_CCs'] = len(arr_none.columns)
    np.set_printoptions(threshold=np.inf)
    config['engagement'] = '"' + np.array2string(
        arr_none.to_numpy(),
        separator=',',
        max_line_width=np.inf
    ).replace(" ", "").replace("\n", "") + '"'
    np.set_printoptions(threshold=np.inf)


    config['num_groups'] = config['num_CCs']*(config['num_CCs']-1)
    config['num_steps'] = 1000 # 0 means run until convergence
    config['record_at_timesteps'] = '"[2, 100, 500, 1000]"'
    config['time_agg'] = '"[1001]"'
    config['uniform']= True 
    config['probability'] = 1
    config['alpha'] = 1
    changing = {}
    # we need a list of changing varaibles because lists are not hashable --> useful for dict
    gen = np.random.RandomState(97)
    changing_var_lists = [['random_seed'], ['rs_model']]
    gen = np.random.RandomState(97)
    changing[0] = [[x] for x in gen.randint(1000000, size=1000)]
    changing[1] = [['general'], ['Comparison'], ['UR']]
    base_config = config 
    generate_config_csvs(base_config, changing, changing_var_lists, path= os.path.join(script_path, f'Simulation_results{percentage_matrix}'),
                            no_folders=no_folders, start_config_no=-1, regenerate_seeds=False)
    

