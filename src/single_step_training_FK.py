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
from src.dataset.data_preprocessing import process_data_single_step_FK
from src.dataset.dynamics_dataset import SingleStepDynamicsDataset_FK

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
from src.dataset.loss import SingleStepLoss_ode
from src.dataset.loss import MultiStepLoss_ode



# NeuralODE
from torchdiffeq import odeint_adjoint as odeint

# Model
from src.model.neural_ode_model import NeuralODE
"""
Baxter
"""
# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# # pth path:
# ckpt_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt/FK/Baxter'

# # Load the collected data:
# data_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/data/FK'
# fk_data = torch.from_numpy(np.load(os.path.join(data_path, 'BaxterDirectDynamics.npy'), allow_pickle=True)).to(device)


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


# # Training function
# def train(model, dataset, path):
#     # Model
#     # Model
#     if model == 'absolute':
#         pushing_model = AbsoluteDynamicsModel(14, 7)
#     elif model == 'RKNN':
#         pushing_model = RKNN_2_DynamicsModel(14, 7)
#     elif model == 'poly_2':
#         pushing_model = Poly_2_DynamicsModel(14, 7)
#     elif model == 'residual':
#         pushing_model = ResidualDynamicsModel(14, 7)
#     elif model == 'mpoly_2':
#         pushing_model = mPoly_2_DynamicsModel(14, 7)
#     elif model == 'way_2':
#         pushing_model = way_2_DynamicsModel(14, 7)
#     else:
#         print("No model name: ", model, " found, please check the list again. ")

#     # Path
#     ckpt_path = os.path.join(path,'/ckpt/FK/Baxter')

#     # Data loader
#     train_loader, val_loader = process_data_single_step_FK(dataset) # batchsize default to be 1000

#     # Loss function
#     state_loss = torch.nn.MSELoss()
#     state_loss = SingleStepLoss(state_loss)

#     # training process
#     lr = 1e-5
#     num_epochs = 5000

#     # optimizer
#     # optimizer = torch.optim.Adam(pushing_absolute_dynamics_model.parameters(), lr = lr, weight_decay=1e-4)
#     optimizer = torch.optim.Adam(pushing_model.parameters(), lr = lr, weight_decay=1e-5)

#     train_losses = [] # record the history of training loss
#     val_losses = [] # record the history of validation loss

#     for epoch_i in range(num_epochs):
#         # absolute model
#         # train_loss_i = train_step(pushing_absolute_dynamics_model, train_loader, optimizer, state_loss)
#         # val_loss_i = val_step(pushing_absolute_dynamics_model, val_loader, state_loss)

#         # residual model
#         train_loss_i = train_step(pushing_model, train_loader, optimizer, state_loss)
#         val_loss_i = val_step(pushing_model, val_loader, state_loss)

#         train_losses.append(train_loss_i)
#         val_losses.append(val_loss_i)

#         # Print results per 1000 epoch
#         if (epoch_i+1) % 500 == 0:
#             print("Epoch %s: ", epoch_i+1)
#             print("Train Loss: ", train_loss_i)
#             print("Val Loss: ", val_loss_i)
    

#     # save model:
#     # absolute model
#     # save_path = os.path.join(ckpt_path, 'pushing_absolute_dynamics_model.pt')
#     # torch.save(pushing_absolute_dynamics_model.state_dict(), save_path)

#     # residual model
#     save_path = os.path.join(ckpt_path, 'pushing_{}_dynamics_model.pt'.format(model))
#     torch.save(pushing_model.state_dict(), save_path)

# Training function
def discrete_train(model, dataset, path):
    # Model
    if model == 'absolute':
        pushing_model = AbsoluteDynamicsModel(14, 7)
    elif model == 'RKNN':
        pushing_model = RKNN_2_DynamicsModel(14, 7)
    elif model == 'poly_2':
        pushing_model = Poly_2_DynamicsModel(14, 7)
    elif model == 'residual':
        pushing_model = ResidualDynamicsModel(14, 7)
    elif model == 'mpoly_2':
        pushing_model = mPoly_2_DynamicsModel(14, 7)
    elif model == 'way_2':
        pushing_model = way_2_DynamicsModel(14, 7)
    else:
        print("No model name: ", model, " found, please check the list again. ")

    # Path
    ckpt_path = os.path.join(path,'/ckpt/FK/Baxter')


    # Data loader
    train_loader, val_loader = process_data_single_step_FK(dataset) # batchsize default to be 1000

    # Loss function
    state_loss = torch.nn.MSELoss()
    state_loss = SingleStepLoss(state_loss)

    # training process
    lr = 1e-5
    num_epochs = 5000

    # optimizer
    # optimizer = torch.optim.Adam(pushing_absolute_dynamics_model.parameters(), lr = lr, weight_decay=1e-4)
    optimizer = torch.optim.Adam(pushing_model.parameters(), lr = lr, weight_decay=1e-5)

    train_losses = [] # record the history of training loss
    val_losses = [] # record the history of validation loss

    for epoch_i in range(num_epochs):
        # absolute model
        # train_loss_i = train_step(pushing_absolute_dynamics_model, train_loader, optimizer, state_loss)
        # val_loss_i = val_step(pushing_absolute_dynamics_model, val_loader, state_loss)

        # residual model
        train_loss_i = train_step(pushing_model, train_loader, optimizer, state_loss)
        val_loss_i = val_step(pushing_model, val_loader, state_loss)

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
    save_path = os.path.join(ckpt_path, 'pushing_{}_dynamics_model.pt'.format(model))
    torch.save(pushing_model.state_dict(), save_path)


# Training function
def continuous_train(method, dataset, path):
    # dimension
    state_dim = 14
    action_dim = 7

    print("Currenlty using ODE Solver: ", method if method else "dopri5")
    # Func
    ode_model = NeuralODE(state_dim, action_dim, method = method).to(device)

    # Path
    ckpt_path = os.path.join(path,'/ckpt/FK/Baxter')

    # Data loader
    train_loader, val_loader = process_data_single_step_FK(dataset) # batchsize default to be 1000

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
