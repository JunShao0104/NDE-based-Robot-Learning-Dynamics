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
from dataset.data_preprocessing import process_data_multiple_step, process_data_ode_batch
from dataset.dynamics_dataset import MultiStepDynamicsDataset

# Model
from model.absolute_dynamics_model import AbsoluteDynamicsModel
from model.residual_dynamics_model import ResidualDynamicsModel
from model.polynet_dynamics_model import Poly_2_DynamicsModel
from model.polynet_dynamics_model import mPoly_2_DynamicsModel
from model.polynet_dynamics_model import way_2_DynamicsModel
from model.fractalnet_dynamics_model import RKNN_2_DynamicsModel
from model.neural_ode_model_1 import NeuralODE_1

# Loss
from dataset.loss import SE2PoseLoss
from dataset.loss import MultiStepLoss,ODEMultiStepLoss

# pth path:
ckpt_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt'

# Load the collected data:
data_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/data'
collected_data = np.load(os.path.join(data_path, 'collected_data.npy'), allow_pickle=True)


# train step func
def train_step(model, train_loader, optimizer, loss_func) -> float:
    train_loss = 0.
    for batch_idx, sample_data in enumerate(train_loader):
        optimizer.zero_grad()
        loss = loss_func(model, sample_data['state'], sample_data['action'], sample_data['next_state'], sample_data['t'])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


# val step func
def val_step(model, val_loader, loss_func) -> float:
    val_loss = 0.
    for batch_idx, sample_data in enumerate(val_loader):
        loss = loss_func(model, sample_data['state'], sample_data['action'], sample_data['next_state'], sample_data['t'])
        val_loss += loss.item()
    return val_loss/len(val_loader)


# Training function
def train():
    print(torch.cuda.is_available())
    if(torch.cuda.is_available()):
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    # Model
    ODE_dynamics_model = NeuralODE_1(3, 3, DEVICE).to(DEVICE)
    print("Currently using device: ", DEVICE)

    # Data loader
    # batch_size=500, num_steps=4,ode_t=3
    train_loader, val_loader = process_data_ode_batch(collected_data, 500, 4,3, DEVICE) # batchsize default to be 500

    # Loss function
    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1, device= DEVICE)
    # pose_loss = MultiStepLoss(pose_loss, discount=0.9) # TOBE CHANGED !!!
    pose_loss = ODEMultiStepLoss(pose_loss, discount=0.9, device=DEVICE)
    # training process
    lr = 5e-5
    num_epochs = 17000

    # optimizer = torch.optim.SGD(pushing_absolute_dynamics_model.parameters(), lr = lr, weight_decay=1e-7, momentum=0.9)
    optimizer = torch.optim.Adam(ODE_dynamics_model.parameters(), lr = lr, weight_decay=1e-6)
    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss
    val_losses = [] # record the history of validation loss

    for epoch_i in range(num_epochs):
        train_loss_i = train_step(ODE_dynamics_model, train_loader, optimizer, pose_loss)
        val_loss_i = val_step(ODE_dynamics_model, val_loader, pose_loss)
        # pbar.set_description(f'Train Loss: {train_loss_i:.6f} | Validation Loss: {val_loss_i:.6f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)
        # Print results per 1000 epoch
        if (epoch_i+1) % 50 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", train_loss_i)
            print("Val Loss: ", val_loss_i)
    

    # save model:
    save_path = os.path.join(ckpt_path, 'ODE_multistep_dynamic_model.pt')
    torch.save(ODE_dynamics_model.state_dict(), save_path)


if __name__ == "__main__":
    train()