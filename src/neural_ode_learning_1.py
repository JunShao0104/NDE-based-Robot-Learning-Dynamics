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

from model.neural_ode_model_1 import NeuralODE
# itertools
import itertools

# pth path:
ckpt_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt'

# Load the collected data:
data_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/data'
collected_data = np.load(os.path.join(data_path, 'collected_data.npy'), allow_pickle=True)


# Training function
def train():
    if(torch.cuda.is_available()):
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    # Data
    batch_y0, batch_t, batch_y = process_data_continuous_batch(collected_data)

    # Func
    func = NeuralODE(3,3,DEVICE).to(DEVICE)


    # Loss function
    # pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SE2PoseLoss_3dim(block_width=0.1, block_length=0.1, device= DEVICE)

    # training process
    lr = 5e-5
    num_epochs = 17000

    # optimizer
    params_func = func.parameters()
    # optimizer = optim.RMSprop(total_params, lr=lr)
    optimizer = torch.optim.Adam(params_func, lr = lr, weight_decay=1e-6)

    # pbar = tqdm(range(num_epochs))
    train_losses = [] # record the history of training loss

    for epoch_i in range(num_epochs):
        optimizer.zero_grad()
        batch_y0 = batch_y0.to(DEVICE)
        print("batch_y0 shape: ", batch_y0.shape) # (10, 100, 3) (T, M, D)
        state = batch_y0[:, :3]
        action = batch_y0[:, 3:]
        print(state.shape, action.shape)
        batch_t = batch_t.to(DEVICE)
        pred_y_proj = func(state,action, batch_t)
        batch_y_proj = batch_y[:, :, :3]
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
    ode_save_path = os.path.join(ckpt_path,'ODEFunc_w_proj.pt' )
    torch.save(func.state_dict(), ode_save_path)


if __name__ == "__main__":
    # test
    train()