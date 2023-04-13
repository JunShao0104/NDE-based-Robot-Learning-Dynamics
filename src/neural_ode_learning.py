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

# Process the data
from dataset.data_preprocessing import process_data_continuous_batch
from dataset.data_preprocessing import process_data_continuous_batch_no_action
from dataset.data_preprocessing import process_data_continuous_batch_step
from dataset.data_preprocessing import process_data_single_step_continuous

# Loss
from dataset.loss import SE2PoseLoss
from dataset.loss import SE2PoseLoss_3dim

# NeuralODE
from torchdiffeq import odeint_adjoint as odeint

# itertools
import itertools

# pth path:
ckpt_path = '/mnt/NDE-based-Robot-Learning-Dynamics/ckpt'

# Load the collected data:
data_path = '/mnt/NDE-based-Robot-Learning-Dynamics/data'
collected_data = np.load(os.path.join(data_path, 'collected_data.npy'), allow_pickle=True)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# func
class ODEFunc(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ODEFunc, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, 100),
            nn.Tanh(),
            nn.Linear(100, out_channels)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        pred_y = self.net(y)

        return pred_y


class ProjectionNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ProjectionNN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.proj = nn.Sequential(
            nn.Linear(in_channels, 100),
            nn.ReLU(),
            nn.Linear(100, out_channels)
        )

    def forward(self, state_action):
        state = self.proj(state_action)

        return state


# Training function
def train_without_dataset():
    state_dim = 3

    # Data
    batch_y0, batch_t, batch_y = process_data_continuous_batch_step(collected_data, T=2)
    batch_y0 = batch_y0.to(device)
    batch_t = batch_t.to(device)
    batch_y = batch_y.to(device)

    # Func
    func = ODEFunc(batch_y0.shape[1], batch_y0.shape[1]).to(device)

    # Proj
    projNN = ProjectionNN(batch_y0.shape[1], state_dim).to(device)

    # Loss function
    # pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SE2PoseLoss_3dim(block_width=0.1, block_length=0.1)

    # training process
    lr = 5e-5
    num_epochs = 20000

    # optimizer
    params_func = func.parameters()
    params_projNN = projNN.parameters()
    total_params = itertools.chain(params_func, params_projNN)
    # optimizer = optim.RMSprop(total_params, lr=lr)
    optimizer = torch.optim.Adam(total_params, lr = lr, weight_decay=1e-4)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss

    for epoch_i in range(num_epochs):
        optimizer.zero_grad()
        pred_y = odeint(func, batch_y0, batch_t)
        T, M, D = pred_y.shape
        # print(pred_y.shape)
        pred_y_proj = projNN(pred_y.reshape(-1, D)).reshape(T, M, -1)
        batch_y_proj = batch_y[:, :, :3]
        loss = pose_loss(pred_y_proj, batch_y_proj)
        loss.backward()
        optimizer.step()
        train_losses.append(loss)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", loss)
    

    # save model:
    ode_save_path = os.path.join(ckpt_path, 'ODEFunc.pt')
    torch.save(func.state_dict(), ode_save_path)
    proj_save_path = os.path.join(ckpt_path, 'ProjNN.pt')
    torch.save(projNN.state_dict(), proj_save_path)


# Train with torch dataset and single step
def train_with_dataset_singlestep():
    # dimension
    state_dim = 3
    action_dim=3

    # Func
    func = ODEFunc(state_dim+action_dim, state_dim+action_dim).to(device)
    # Proj
    projNN = ProjectionNN(state_dim+action_dim, state_dim).to(device)

    # Data loader
    train_loader, val_loader = process_data_single_step_continuous(collected_data) # batchsize default to be 500

    # Loss function
    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    # pose_loss = SE2PoseLoss_3dim(block_width=0.1, block_length=0.1)

    # training process
    lr = 1e-5
    num_epochs = 20000

    # optimizer
    params_func = func.parameters()
    params_projNN = projNN.parameters()
    total_params = itertools.chain(params_func, params_projNN)
    optimizer = torch.optim.Adam(total_params, lr = lr, weight_decay=1e-4)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss
    val_losses = []

    for epoch_i in range(num_epochs):
        train_loss_i = train_step_single(func, projNN, train_loader, optimizer, pose_loss, state_dim)
        val_loss_i = val_step_single(func, projNN, val_loader, pose_loss, state_dim)
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)
    

    # save model:
    ode_save_path = os.path.join(ckpt_path, 'ODEFunc_single_step.pt')
    torch.save(func.state_dict(), ode_save_path)
    proj_save_path = os.path.join(ckpt_path, 'ProjNN_single_step.pt')
    torch.save(projNN.state_dict(), proj_save_path)


# train step func
def train_step_single(func, projNN, train_loader, optimizer, loss_func, state_dim) -> float:
    train_loss = 0.
    for batch_idx, sample_data in enumerate(train_loader):
        T = 2 # For single step
        optimizer.zero_grad()
        batch_y0 = sample_data['state_action'].to(device) # (M, D)
        batch_y = sample_data['next_state'].to(device) # (M, S)
        batch_t = torch.arange(T).float().to(device)
        pred_y = odeint(func, batch_y0, batch_t) # (T, M, D)
        T, M, D = pred_y.shape
        pred_y_proj = projNN(pred_y.reshape(-1, D)).reshape(T, M, -1) # (T, M, S)
        loss = loss_func(pred_y_proj[-1], batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


# val step func
def val_step_single(func, projNN, val_loader, loss_func, state_dim) -> float:
    val_loss = 0.
    for batch_idx, sample_data in enumerate(val_loader):
        T = 2 # For single step
        batch_y0 = sample_data['state_action'].to(device) # (M, D)
        batch_y = sample_data['next_state'].to(device) # (M, S)
        batch_t = torch.arange(T).float().to(device)
        pred_y = odeint(func, batch_y0, batch_t) # (T, M, D)
        T, M, D = pred_y.shape
        pred_y_proj = projNN(pred_y.reshape(-1, D)).reshape(T, M, -1) # (T, M, S)
        loss = loss_func(pred_y_proj[-1], batch_y)
        val_loss += loss.item()
    return val_loss/len(val_loader)


if __name__ == "__main__":
    # test
    # train_without_dataset()
    train_with_dataset_singlestep()