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
from src.env.panda_pushing_env import PandaPushingEnv

# Process dataset
from src.dataset.data_preprocessing import process_data_single_step
from src.dataset.dynamics_dataset import SingleStepDynamicsDataset

# Model
from src.model.absolute_dynamics_model import AbsoluteDynamicsModel
from src.model.residual_dynamics_model import ResidualDynamicsModel
from src.model.polynet_dynamics_model import Poly_2_DynamicsModel
from src.model.polynet_dynamics_model import mPoly_2_DynamicsModel
from src.model.polynet_dynamics_model import way_2_DynamicsModel
from src.model.fractalnet_dynamics_model import RKNN_2_DynamicsModel

# Loss
from src.dataset.loss import SE2PoseLoss
from src.dataset.loss import SingleStepLoss

# pth path:
# ckpt_path = '/home/zlj/Documents/ROB498/project/code/NDE-based-Robot-Learning-Dynamics/ckpt'

# # Load the collected data:
# data_path = '/home/zlj/Documents/ROB498/project/code/NDE-based-Robot-Learning-Dynamics/data'
# collected_data = np.load(os.path.join(data_path, 'collected_data.npy'), allow_pickle=True)


# train step func
def train_step(model, train_loader, optimizer, loss_func) -> float:
    train_loss = 0.
    for batch_idx, sample_data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = loss_func(model, sample_data['state'], sample_data['action'], sample_data['next_state'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


# val step func
def val_step(model, val_loader, loss_func) -> float:
    val_loss = 0.
    for batch_idx, sample_data in enumerate(val_loader):
        loss = loss_func(model, sample_data['state'], sample_data['action'], sample_data['next_state'])
        val_loss += loss.item()
    return val_loss/len(val_loader)


# Training function
def train(model, path, dataset):
    # Model
    if model == 'absolute':
        pushing_model = AbsoluteDynamicsModel(3, 3)
    elif model == 'RKNN':
        pushing_model = RKNN_2_DynamicsModel(3, 3)
    elif model == 'poly_2':
        pushing_model = Poly_2_DynamicsModel(3, 3)
    elif model == 'residual':
        pushing_model = ResidualDynamicsModel(3, 3)
    elif model == 'mpoly_2':
        pushing_model = mPoly_2_DynamicsModel(3, 3)
    elif model == 'way_2':
        pushing_model = way_2_DynamicsModel(3, 3)
    else:
        print("No model name: ", model, " found, please check the list again. ")

    # Data loader
    train_loader, val_loader = process_data_single_step(dataset) # batchsize default to be 500

    # Loss function
    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SingleStepLoss(pose_loss)

    # training process
    lr = 1e-5
    num_epochs = 20000

    # optimizer = torch.optim.SGD(pushing_absolute_dynamics_model.parameters(), lr = lr, weight_decay=1e-7, momentum=0.9)
    optimizer = torch.optim.Adam(pushing_model.parameters(), lr = lr, weight_decay=1e-6)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss
    val_losses = [] # record the history of validation loss

    for epoch_i in range(num_epochs):
        train_loss_i = train_step(pushing_model, train_loader, optimizer, pose_loss)
        val_loss_i = val_step(pushing_model, val_loader, pose_loss)
        # pbar.set_description(f'Train Loss: {train_loss_i:.6f} | Validation Loss: {val_loss_i:.6f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Print results per 1000 epoch
        if (epoch_i+1) % 1000 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)
    

    # Path
    ckpt_path = os.path.join(path,'/ckpt/Panda_pushing/discrete/')

    # save model:
    save_path = os.path.join(ckpt_path, 'pushing_{}_dynamics_model.pt'.format(model))
    torch.save(pushing_model.state_dict(), save_path)
