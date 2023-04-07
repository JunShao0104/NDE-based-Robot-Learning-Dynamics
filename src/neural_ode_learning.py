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

# Loss
from dataset.loss import SE2PoseLoss
from dataset.loss import SE2PoseLoss_3dim

# NeuralODE
from torchdiffeq import odeint_adjoint as odeint

# itertools
import itertools

# pth path:
ckpt_path = '/home/zlj/Documents/ROB498/project/code/NDE-based-Robot-Learning-Dynamics/ckpt'

# Load the collected data:
data_path = '/home/zlj/Documents/ROB498/project/code/NDE-based-Robot-Learning-Dynamics/data'
collected_data = np.load(os.path.join(data_path, 'collected_data.npy'), allow_pickle=True)

# func
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 100),
            nn.Tanh(),
            nn.Linear(100, 6)
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        pred_y = self.net(y) # (T, 3)
        # pred_y_action = torch.zeros_like(pred_y) # (T, 3)
        # pred_y = torch.cat((pred_y, pred_y_action), dim=1) # (T, 6)

        return pred_y


class ProjectionNN(nn.Module):
    def __init__(self):
        super(ProjectionNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(6, 100),
            nn.ReLU(),
            nn.Linear(100, 3)
        )

    def forward(self, state_action):
        state = self.proj(state_action)

        return state


# Training function
def train():
    # Data
    batch_y0, batch_t, batch_y = process_data_continuous_batch(collected_data)

    # Func
    func = ODEFunc()

    # Proj
    projNN = ProjectionNN()

    # Loss function
    # pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SE2PoseLoss_3dim(block_width=0.1, block_length=0.1)

    # training process
    lr = 1e-5
    num_epochs = 20000

    # optimizer
    params_func = func.parameters()
    params_projNN = projNN.parameters()
    total_params = itertools.chain(params_func, params_projNN)
    # optimizer = optim.RMSprop(total_params, lr=lr)
    optimizer = torch.optim.Adam(total_params, lr = lr, weight_decay=1e-6)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss

    for epoch_i in range(num_epochs):
        optimizer.zero_grad()
        pred_y = odeint(func, batch_y0, batch_t)
        T, M, D = pred_y.shape
        # print(pred_y.shape)
        pred_y_proj = projNN(pred_y.reshape(-1, D)).reshape(T, M, -1)
        batch_y_proj = batch_y[:, :, :3]
        # print("batch_y_proj shape: ", batch_y_proj.shape) # (10, 100, 3) (T, M, D)
        # print("pred_y_proj shape: ", pred_y_proj.shape)
        loss = pose_loss(pred_y_proj, batch_y_proj)
        loss.backward()
        optimizer.step()
        train_losses.append(loss)

        # Print results per 1000 epoch
        if (epoch_i+1) % 500 == 0:
            print("Epoch %s: ", epoch_i+1)
            print("Train Loss: ", loss)
    

    # save model:
    save_path = os.path.join(ckpt_path, 'ODEFunc.pt')
    torch.save(func.state_dict(), save_path)


if __name__ == "__main__":
    # test
    train()