# This file creates the config files for running the experiments
import os
import scipy.stats as st
import numpy as np


def read_seeds(file_name='seeds_10000.txt'):
    'Reads the file with the 10000 random seeds'

    f = open(file_name, 'r')
    lines = f.read().splitlines()
    lines = [int(l) for l in lines]
    f.close()

    return lines


def generate_config_csvs(base_config, changing, var_lists, path='Simulation_results',
                         no_folders=1, start_config_no=-1, regenerate_seeds=False):
    '''Creates the .csv files given the the fix parameters and the cahnging parameters.
    base_config = dictionary with the fixed parameters {parameter_name: parameter_val, ...}
    changing = dictionary with the changing parameters {[p1, p2]:[[v1, v2], [v1.0, v2.0]], ...}
    '''

    if regenerate_seeds:
        gen = np.random.RandomState(97)

    def write_parameters(dict_param, no_comb):
        ''' Writes a dictionary of parameters to a file indexed by no_comb.
        '''
        os.makedirs(path, exist_ok=True)
        current_path = os.path.join(path, 'config' + str(no_comb) + '.csv')
        with open(current_path, 'w') as f:
            for key in dict_param.keys():
                f.write("%s,%s\n" % (key, dict_param[key]))

    config = {}
    no = start_config_no

    # put the base parameters into config
    for var in base_config:
        config[var] = base_config[var]

    # generate all combinations for the changing parameters
    choice_changing = [0 for i in var_lists]
    constructed_all = False

    while not constructed_all:
        # set all variables according to the current choice of parameter combinations
        for var_list_order_no in changing:
            var_list = var_lists[var_list_order_no]
            for var_pos in range(len(var_list)):
                var = var_list[var_pos]
                alternative_no = choice_changing[var_list_order_no]
                config[var] = changing[var_list_order_no][alternative_no][var_pos]

                # generate random seed if this is the current variable
                if regenerate_seeds and var == 'random_seed':
                    config[var] = gen.randint(1000000)

        # save the current config file
        no += 1
        write_parameters(config, no)

        # iterate twards the next choice_changing
        constructed_all = True
        for var_list_order_no in range(len(var_lists)):
            no_options = len(changing[var_list_order_no])

            # if we can increase
            if choice_changing[var_list_order_no] < (no_options - 1):
                # increase the choce option
                choice_changing[var_list_order_no] += 1
                # the ones before are reseted to 0
                for j in range(var_list_order_no):
                    choice_changing[j] = 0
                # we didn't construct all - there is a new choice
                constructed_all = False
                break

    no += 1

    # create no_folders folders with the names of the generated files
    for i in range(no_folders):
        current_path = os.path.join(path, "file_names" + str(i + 1) + ".txt")
        f = open(current_path, mode='w')
        for j in range(start_config_no + 1, no):
            if j % no_folders == i:
                print_path = os.path.join(path, 'config' + str(j) + '.csv\n')
                f.writelines((print_path))

    return no
