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

# Process the data
from src.dataset.data_preprocessing import process_data_single_step
from src.dataset.data_preprocessing import process_data_multiple_step

# Loss
from src.dataset.loss import SE2PoseLoss, SingleStepLoss_ode, MultiStepLoss_ode

# NeuralODE
from torchdiffeq import odeint_adjoint as odeint

# Model
from src.model.neural_ode_model import NeuralODE

# pth path:
# ckpt_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt/Panda_pushing/continuous'

# Load the collected data:
# data_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/data/Panda_pushing'
# collected_data = np.load(os.path.join(data_path, 'collected_data.npy'), allow_pickle=True)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


"""
train and val step
"""
# train step func
def train_step(model, train_loader, optimizer, loss_func) -> float:
    train_loss = 0.
    for batch_idx, sample_data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = loss_func(model, sample_data['state'].to(device), sample_data['action'].to(device), sample_data['next_state'].to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


# val step func
def val_step(model, val_loader, loss_func) -> float:
    val_loss = 0.
    for batch_idx, sample_data in enumerate(val_loader):
        loss = loss_func(model, sample_data['state'].to(device), sample_data['action'].to(device), sample_data['next_state'].to(device))
        val_loss += loss.item()
    return val_loss/len(val_loader)


"""
Single Step
"""
# Train with torch dataset and single step
def train_singlestep(method, dataset, path):
    # dimension
    state_dim = 3
    action_dim = 3

    # Func
    print("Currenlty using ODE Solver: ", method if method else "dopri5")
    ode_model = NeuralODE(state_dim, action_dim, method = method).to(device)

    # Path
    ckpt_path = path + '/ckpt/Panda_pushing/continuous'

    # Data loader
    train_loader, val_loader = process_data_single_step(dataset) # batchsize default to be 500

    # Loss function
    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SingleStepLoss_ode(pose_loss, method= method)

    # training process
    lr = 1e-5
    num_epochs = 30000

    # optimizer
    optimizer = torch.optim.Adam(ode_model.parameters(), lr = lr, weight_decay=1e-6)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_step(ode_model, train_loader, optimizer, pose_loss)
        val_loss_i = val_step(ode_model, val_loader, pose_loss)
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)
    

    # save model:
    ode_model_save_path = os.path.join(ckpt_path, 'ODEFunc_single_step_{}_func3.pt'.format(method if method else "dopri5"))
    torch.save(ode_model.state_dict(), ode_model_save_path)


"""
Multi Step
"""
# Train with torch dataset and multi step
def train_multistep(method, dataset, path, num_steps = 4):
    # dimension
    state_dim = 3
    action_dim = 3

    # Func
    print("Currenlty using ODE Solver: ", method if method else "dopri5")
    ode_model = NeuralODE(state_dim, action_dim).to(device)

    # Path
    ckpt_path = path + '/ckpt/Panda_pushing/continuous'
    print(ckpt_path)
    print(num_steps)

    # Data loader
    train_loader, val_loader = process_data_multiple_step(dataset, batch_size=1000, num_steps=num_steps) # batchsize default to be 500

    # Loss function
    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = MultiStepLoss_ode(pose_loss)

    # training process
    lr = 1e-5
    num_epochs = 30000

    # optimizer
    optimizer = torch.optim.Adam(ode_model.parameters(), lr = lr, weight_decay=1e-6)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_step(ode_model, train_loader, optimizer, pose_loss)
        val_loss_i = val_step(ode_model, val_loader, pose_loss)
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)
    

    # save model:
    ode_model_save_path = os.path.join(ckpt_path, 'ODEFunc_multi_step_{}_func3_{}.pt'.format(method if method else "dopri5", num_steps))
    torch.save(ode_model.state_dict(), ode_model_save_path)


# if __name__ == "__main__":

    # Train single step
    # train_singlestep()
    # Train single step Fixed step rk4
    # train_singlestep('rk4')

    # Train multi step
    # train_multistep()