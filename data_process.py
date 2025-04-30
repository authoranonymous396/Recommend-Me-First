import model 
import simulation as sim 
from experimental_setup import generate_config_csvs
import numpy as np 
import pandas as pd
import metrics 
def read_sims(no, first_sim=0, path='Simulation_results/'):
    all_sims = []
    for n in range(first_sim, first_sim + no):
        config_file = path + 'config' + str(n) + '.csv'
        print(f"Loading config file: {config_file}")
        s = sim.Simulation(config_file)
        s.read(True)
        # Print a key metric to verify different simulations
        all_sims.append(s)
    return all_sims

def data_IF(all_sims, fixed_pars = {}, x="CCs", y="num_followers", groups_list =[1], hue = "alpha", style = None,
                            measure_at_time = 0,function_if = metrics.get_if_IF, print_progress = True):
    '''
    Creates a dataframe with a selection of simulation results corresponding to the specifeid input data.
    
    Input
    ------
    all_sims: list
       The list of Simulation objects (with config file loaded)
       
    fixed_pars: dict
       A dictionary which maps parameter names with list of values thay can take; 
       only the simulations with specified pararameters having values withint this list will be included in the final dataframe
    x: string
        Parameter name for the x-axis of the plot to follow
    y: string
        Parameter name for the y-axis of the plot to follow
    hue: string
        Parameter name for the hue which will be used in the plot to follow
    style: None/'timestep'
        Will be used to signal whether the results should be recorded at all intermediate timesteps ('timesteps') 
        or only at final ones (None)
    measure_at_time: int
        Specified the timestep to be considered
    print_progres: bool
        Says whether or not the progress in loading the simulation results should be shown
    ------
    
    Output
    ------
    data: dataframe
        Includes the selected results and configs for the specified parameters
    ------
    '''
    
    # Create a label for the y-axis (which will contain averages of y-values resulted from multiple simulations)
    y_label = 'average_' + y
    
    # Create the dictionary from which we will form the dataframe
    d = {x:[], y_label:[], hue:[]}
    if style:
        d[style] = []
    # We consider the results of each run in turn and add them to the dictionary 
    for s_num, s in enumerate(all_sims):
        
        # Print progress every 1k steps if needed
        if s_num%1000 == 0 and print_progress:
            print(s_num)
        
        # Check whether the curent simulation has the parameter configuration as specified by fixed_pars
        #    if so, it will be added further considered, otherwise it will be ingored
        have = True
        for p in fixed_pars:
            if p in s.config:
                if str(s.config[p]) not in fixed_pars[p]:
                    have = False
        
        # If the simulation s corresponds we proceed to integrate it into the dictionary d
        if have:
            # load the results for simulation s
            results = s.return_results()
            time_agg = s.config['time_agg'][0]
            
            # if we want to find the %of CC who have IF, results[y] need to be computed
            # otherwise the results[y] already exists and we load it at the specified timestep
            if y == 'CC_is_IF':
                if style is None:
                    results[y] = function_if(results['num_followers'][measure_at_time])
                else:
                    results[y] = [function_if(results['num_followers'][i]) 
                                  for i, t in enumerate(s.config['record_at_timesteps'])]
            else:
                results[y_label] = results[y][measure_at_time]
            
            # we consider four cases of plots:
            #     (1) when we record the results at different timesteps
            #     (2) when results are at one specific timestep (e.g., quality vs IF)
            #     (3) when we look at whether the simulation converged
            #     (4) when we look at user satisfaction
            if style is not None:
                for i, t in enumerate(s.config['record_at_timesteps']):
                    for c in range(s.config['num_CCs']):
                        d[hue].append(s.config[hue])
                        d[style].append(t)
                        d[x].append(c)
                        d[y_label].append(results[y][i][c])
            elif x == "CCs":
                for c in range(s.config['num_CCs']):
                    d[hue].append(s.config[hue])
                    d[x].append(c)
                    d[y_label].append(results[y][c])
            elif x == "timestep" and y=="did_converge":
                for i, t in enumerate(s.config['record_at_timesteps']):
                    d[hue].append(s.config[hue])
                    d[x].append(t)
                    d[y_label].append(results[y][i])
            elif x == "timestep":
                # if it converges faster, we keep the results of the last
                max_i = len(results[y_label])
                for i in range(1000):
                    d[hue].append(s.config[hue])
                    d[x].append(i+1)
                    if i<max_i:
                        d[y_label].append(results[y_label][i] + 1)
                    else:
                        d[y_label].append(results[y_label][-1] + 1)
                    
    return pd.DataFrame(data=d)

    

