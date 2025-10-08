# Recommend-Me-First
## Overview
This repository contains the code to run simulations provided in the paper: "recommend me first: how the order of recommendations might impact fairness for creators". We provide the code to generate the data and do not provide the data itself as the files are extremely large. This repository also includes a file with supplementary materials for the paper. 
## Main Folders

### MovieLens Simulations
As explained in the paper, we select a portion of the dataset (32 millions): can be done by Downloading the dataset and running data_initialization.py. As an output, you need to have a matrix named:  ``` Initial_matrix_'somevariable'.csv ``` 

To generate the data we need to run the following line: 
 ```
from generate_config import generate_config_files
generate_config_files('somevariable', 10)
 ``` 
 This creates similar config files in a folder and then to run simulations : 
  ```
import Simulations as sim
sim.run_sims('somevariable', 1)
 ```

### Synthetic_data Simulations
The two main files  for understanding our code are model.py and Configuration_simulations.ipynb. All functions related to the modeling of our algorithm can be found in model.py. For the inizialition of the parameters, generation of the data and run the code, Configuration_simulations.ipynb contains the necessary setup code with a small example. Additional files are given in order to run and save properly the simulations.






