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

# Process dataset
from dataset.data_preprocessing import process_data_single_step_FK
from dataset.data_preprocessing import process_data_multiple_step_FK
from dataset.dynamics_dataset import SingleStepDynamicsDataset_FK
from dataset.dynamics_dataset import MultiStepDynamicsDataset_FK

# Model
from model.absolute_dynamics_model import AbsoluteDynamicsModel
from model.residual_dynamics_model import ResidualDynamicsModel
from model.polynet_dynamics_model import Poly_2_DynamicsModel
from model.polynet_dynamics_model import mPoly_2_DynamicsModel
from model.polynet_dynamics_model import way_2_DynamicsModel
from model.fractalnet_dynamics_model import RKNN_2_DynamicsModel
from model.neural_ode_model import NeuralODE

# Loss
from dataset.loss import SE2PoseLoss
from dataset.loss import SingleStepLoss
from dataset.loss import SingleStepLoss_ode
from dataset.loss import MultiStepLoss
from dataset.loss import MultiStepLoss_ode

"""
Baxter
"""
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# pth path:
ckpt_path = '/mnt/NDE-based-Robot-Learning-Dynamics/ckpt/FK/Baxter'

# Load the collected data:
data_path = '/mnt/NDE-based-Robot-Learning-Dynamics/data/FK'
fk_data = torch.from_numpy(np.load(os.path.join(data_path, 'BaxterDirectDynamics.npy'), allow_pickle=True)).to(device)


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


# Training function
def discrete_train():
    # Model
    # pushing_absolute_dynamics_model = AbsoluteDynamicsModel(14, 7).to(device)
    pushing_residual_dynamics_model = ResidualDynamicsModel(14, 7).to(device)

    # Data loader
    train_loader, val_loader = process_data_single_step_FK(fk_data) # batchsize default to be 1000

    # Loss function
    state_loss = torch.nn.MSELoss()
    state_loss = SingleStepLoss(state_loss)

    # training process
    lr = 1e-5
    num_epochs = 5000

    # optimizer
    # optimizer = torch.optim.Adam(pushing_absolute_dynamics_model.parameters(), lr = lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(pushing_residual_dynamics_model.parameters(), lr = lr, weight_decay=1e-5)

    train_losses = [] # record the history of training loss
    val_losses = [] # record the history of validation loss

    for epoch_i in range(num_epochs):
        # absolute model
        # train_loss_i = train_step(pushing_absolute_dynamics_model, train_loader, optimizer, state_loss)
        # val_loss_i = val_step(pushing_absolute_dynamics_model, val_loader, state_loss)

        # residual model
        train_loss_i = train_step(pushing_residual_dynamics_model, train_loader, optimizer, state_loss)
        val_loss_i = val_step(pushing_residual_dynamics_model, val_loader, state_loss)

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)
    

    # save model:
    # absolute model
    # save_path = os.path.join(ckpt_path, 'pushing_absolute_dynamics_model.pt')
    # torch.save(pushing_absolute_dynamics_model.state_dict(), save_path)

    # residual model
    save_path = os.path.join(ckpt_path, 'pushing_residual_dynamics_model.pt')
    torch.save(pushing_residual_dynamics_model.state_dict(), save_path)


# Training function
def continuous_train():
    # dimension
    state_dim = 14
    action_dim = 7

    # Func
    ode_model = NeuralODE(state_dim, action_dim).to(device)

    # Data loader
    train_loader, val_loader = process_data_single_step_FK(fk_data) # batchsize default to be 1000

    # Loss function
    state_loss = torch.nn.MSELoss()
    state_loss = SingleStepLoss_ode(state_loss)

    # training process
    lr = 1e-5
    num_epochs = 5000

    # optimizer
    optimizer = torch.optim.Adam(ode_model.parameters(), lr = lr, weight_decay=1e-5)

    train_losses = [] # record the history of training loss
    val_losses = [] # record the history of validation loss

    for epoch_i in range(num_epochs):
        train_loss_i = train_step(ode_model, train_loader, optimizer, state_loss)
        val_loss_i = val_step(ode_model, val_loader, state_loss)
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)

    # save model
    save_path = os.path.join(ckpt_path, 'pushing_ode_model_single_step.pt')
    torch.save(ode_model.state_dict(), save_path)


if __name__ == "__main__":
    # Train with discrete method
    # discrete_train()

    # Train with conitnuous method
    continuous_train()