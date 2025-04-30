import time
import json
import numpy as np
import csv
import re
import os
import ast
from copy import deepcopy
import pandas as pd

#import maximzer_satisfers_model as model # the model we are using
import model as model 
import sys

csv.field_size_limit(sys.maxsize) # to avoid error when reading large csv 

class Simulation:
    def __init__(self, file_name):
        self.file_name = file_name
        self.config = self.read_config()

        # set the random seed
        gen = np.random.RandomState(self.config['random_seed'])
        self.config['random_generator'] = gen

        self.results = []

    def get_sim_no(self):
        '''Gets the number of the simulation from the file_name'''

        numbers = re.findall(r'\d+', self.file_name)
        if len(numbers) == 0:
            return ''

        return numbers[0]

    def get_path(self):
        '''If the file_name also has a path to another folder, keeps that separate'''

        file_name_components = self.file_name.split('/')

        if len(file_name_components) == 1:
            return ''

        return file_name_components[:-1]

    def read_config(self):
        ''' Reads the file with parameters, given its name.'''

        infile = open(self.file_name, mode='r')
        reader = csv.reader(infile)
        config = {rows[0]: rows[1] for rows in reader}

        # transforms non-string parameters to correct type
        str_param_names = ['Variable name', 'rs_model']
        for name in config:
            if name not in str_param_names:
                config[name] = ast.literal_eval(config[name])

        self.config = config

        return config

    def return_results(self):
        '''This function returns the results of a simulation without saving it'''

        no = self.get_sim_no()
        path = self.get_path()
        f = open(os.path.join(*path, 'run' + str(no) + '.json'), 'r')
        data = f.read()
        f.close()

        return json.loads(data)

    def read(self, all=True):
        '''This takes the results and configuration for a simulation that was run before'''

        # read the config file
        self.read_config()

        # load the results into the simulation object
        if all:
            no = self.get_sim_no()
            path = self.get_path()
            f = open(os.path.join(*path, 'run' + str(no) + '.json'), 'r')
            data = f.read()
            f.close()

            self.results = {**json.loads(data), **self.config}
            # self.results = pd.read_csv(os.path.join(*path, 'run' + str(no) + '.csv'))

    def simulate(self, read_config=True):
        '''Runs a simulation, for the parameters in the config file.
        This updates the object ranther than returning the results.
        '''

        # Load the parameters from the file "config.csv"
        no_file = self.get_sim_no()

        # variables to track during simulation
        data = {'num_file_sim': no_file}

        # create the platform
        p = model.Platform(self.config)

        # iterate the platform either num_steps or until convergence
        num_iterations = self.config['num_steps']
        record_at_timesteps = self.config['record_at_timesteps']
        data['num_followers'] = []
        data['num_followees'] = []
        data['num_timestep_users_found_best'] = []
        data['average_pos_best_CC'] = []
        data['did_converge'] = []
        data['genres']= []
        data['recorded_timesteps'] = record_at_timesteps

        if num_iterations:
            did_converge = False
            for i in range(1, num_iterations + 1):
                # iterate only if it didn't converge so far
                if not did_converge:
                    p.iterate()

                # record metrics and check convergence if it is one of the timesteps we record data at
                if i in record_at_timesteps:
                    if not did_converge:
                        p.update_searching_users()
                        did_converge = p.check_convergence()

                    # add metrics
                    data['did_converge'].append(did_converge)
                    data['num_followers'].append(
                        p.network.num_followers.tolist())
                    data['num_followees'].append(
                        p.network.num_followees.tolist())
                    data['num_timestep_users_found_best'].append(deepcopy(
                        p.users_found_best))
                    data['average_pos_best_CC'].append(
                        deepcopy(p.average_pos_best_CC))
                    #data['genres']= [deepcopy(p.CCs[i].genre) for i in range(len(p.CCs))]
        else:
            while not p.check_convergence():
                num_iterations += 1
                p.iterate()
                # p.update_searching_users()
        # record statistics after the runs
        data['timesteps'] = num_iterations
        #data['G'] = p.network.G.tolist()

        # record the results
        self.results = data

    def save_network_csvs(self, name=None):
        '''Takes the network fromt he results and saves nodes and edges as csv.
        '''

        network = self.results['G']  # adjeicency matrix  (row = users)

        # 1) make the nodes files
        nodes = {'id': [], 'quality': []}
        # 1a) add CCs
        num_CCs = self.config['num_CCs']
        for i, v in enumerate(network[0]):
            nodes['id'].append(i)
            nodes['quality'].append(num_CCs - i)
        # 1b) add users
        for u, row in enumerate(network):
            nodes['id'].append(u + num_CCs)
            nodes['quality'].append(0)

        # 2) make the edge files
        edges = {'source': [], 'target': []}
        for u, row in enumerate(network):
            for c, does_u_follow_c in enumerate(row):
                if does_u_follow_c:
                    edges['source'].append(u + num_CCs)
                    edges['target'].append(c)

        # transform to dataframes
        df_nodes = pd.DataFrame(nodes)
        df_edges = pd.DataFrame(edges)

        # save them to csv
        no = self.get_sim_no() if name is None else name
        df_nodes.to_csv('df_nodes_' + str(no) + '.csv', index=False)
        df_edges.to_csv('df_edges_' + str(no) + '.csv', index=False)

    def simulate_and_save_network_csvs(self, name=None):
        '''Wrapper for simulation + saving edge&node csv'''
        self.simulate()
        self.save_network_csvs(name)


def run_sim(file_name="Simulation_results/config.csv", path = "Simulation_results"):
    '''Runs the simulation for one config file. The resulting dataframe is saved into a .csv
    path = by default it saves the results into a separate folder;
           define path as the empty string if you want it to be in the saved in the same folder'''

    # Run a simulation with the given parameters

    sim = Simulation(file_name)
    sim.simulate()

    # Find the config file number
    no = sim.get_sim_no()

    # print(sim.results['attributes'])
    # print(sim.results['strategy'])
    # print(sim.results['utility'])

    # Save the results into the respective folder
    save_path = os.path.join(path, 'run' + str(no) + '.json')
    f = open(save_path, 'w')
    #print(sim.results)
    json.dump(sim.results, f)
    f.close()
    # sim.results.to_csv(save_path)


def run_sims(file_name_configs='Simulation_results/file_names1.txt', path = "Simulation_results"):
    '''Runs multiple configXXXXX.csv files.
    file_name_configs = a folder with all the names of the config files that need to be runned.
    '''

    file_name_configs = os.path.join(*file_name_configs.split('/'))
    file_names = open(file_name_configs, mode='r').readlines()
    for file_name in file_names:
        print(file_name)
        file_name = file_name.rstrip("\n\r")
        run_sim(file_name, path)
