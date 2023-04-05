import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm
from env.panda_pushing_env import PandaPushingEnv
from utils.visualizers import GIFVisualizer, NotebookVisualizer

# Collect data (it may take some time)
from dataset.data_preprocessing import collect_data_random

def collect_data(N=None, T=None, data_path=None):
    # Data collection parameters
    if N is None:
        N = 100 # Number of trajectories
    if T is None:
        T = 10 # Trajectory length

    # Initialize the environment and collect data
    env = PandaPushingEnv()
    env.reset()
    collected_data = collect_data_random(env, num_trajectories=N, trajectory_length=T)


    # Verify the number of data collected:
    print(f'We have collected {len(collected_data)} trajectories')
    print('A data sample contains: ')
    for k, v in collected_data[0].items():
        assert(type(v) == np.ndarray)
        assert(v.dtype == np.float32)
        print(f'\t {k}: numpy array of shape {v.shape}')

    # Save the collected data into a file
    if data_path is None:
        np.save('collected_data.npy', collected_data)
    else:
        np.save(os.path.join(data_path, 'collected_data.npy'), collected_data)



if __name__ == "__main__":
    N = 100
    T = 10
    data_path = '/home/zlj/Documents/ROB498/project/code/NDE-based-Robot-Learning-Dynamics/data'
    collect_data(N, T, data_path)
    