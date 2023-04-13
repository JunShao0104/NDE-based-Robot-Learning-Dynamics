import torch
import numpy as np
import os
from dataset.dynamics_dataset import ContinuousDynamicsDataset_SingleStep
from dataset.loss import SE2PoseLoss_3dim, SE2PoseLoss
from model.neural_ode_model import NeuralODE


# Load the collected data: 
data_path = '/mnt/NDE-based-Robot-Learning-Dynamics/data'
validation_data = np.load(os.path.join(data_path, 'validation_data.npy'), allow_pickle=True)

# dataset
val_dataset = ContinuousDynamicsDataset_SingleStep(validation_data)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

# model
ode_pth_path = '/mnt/NDE-based-Robot-Learning-Dynamics/ckpt/ODEFunc_single_step.pt'
proj_pth_path = '/mnt/NDE-based-Robot-Learning-Dynamics/ckpt/ProjNN_single_step.pt' 
state_dim = 3
action_dim = 3
model = NeuralODE(ode_pth_path, proj_pth_path, state_dim, action_dim)

pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)

loss_ode = 0.0

for item in val_loader:
    state = item['state_action'][:, :state_dim]
    action = item['state_action'][:, state_dim:]
    pred_state = model(state, action)
    loss_ode += pose_loss(pred_state, item['next_state'])

print(f'Validation loss for Neural ODE model is {loss_ode:.8f}')
