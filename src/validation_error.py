import torch
import numpy as np
import os
from dataset.dynamics_dataset import SingleStepDynamicsDataset, MultiStepDynamicsDataset
from dataset.loss import SE2PoseLoss, SingleStepLoss_ode, MultiStepLoss_ode
from model.neural_ode_model import NeuralODE

# Load the collected data: 
data_path = '/mnt/NDE-based-Robot-Learning-Dynamics/data/Panda_pushing'
validation_data = np.load(os.path.join(data_path, 'validation_data.npy'), allow_pickle=True)


"""
Single Step model
"""
def validation_single_step_ode():
    # dataset
    val_dataset = SingleStepDynamicsDataset(validation_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

    # model
    ode_pth_path = '/mnt/NDE-based-Robot-Learning-Dynamics/ckpt/Panda_pushing/continuous/ODEFunc_single_step.pt'
    state_dim = 3
    action_dim = 3
    model = NeuralODE(state_dim, action_dim)
    model.load_state_dict(torch.load(ode_pth_path))

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SingleStepLoss_ode(pose_loss)

    loss_ode = 0.0

    for item in val_loader:
        state = item['state']
        action = item['action']
        next_state = item['next_state']
        loss_ode += pose_loss(model, state, action, next_state)

    print(f'Validation loss for Neural ODE model is {loss_ode:.8f}') # 0.00024968


"""
Multi Step model
"""
def validation_multi_step_ode():
    # dataset
    val_dataset = MultiStepDynamicsDataset(validation_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

    # model
    ode_pth_path = '/mnt/NDE-based-Robot-Learning-Dynamics/ckpt/Panda_pushing/continuous/ODEFunc_multi_step.pt'
    state_dim = 3
    action_dim = 3
    model = NeuralODE(state_dim, action_dim)
    model.load_state_dict(torch.load(ode_pth_path))

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = MultiStepLoss_ode(pose_loss)

    loss_ode = 0.0

    for item in val_loader:
        state = item['state']
        action = item['action']
        next_state = item['next_state']
        loss_ode += pose_loss(model, state, action, next_state)

    print(f'Validation loss for Neural ODE model is {loss_ode:.8f}') # 0.00226656


if __name__ == "__main__":
    # single step ode model
    # validation_single_step_ode()

    # multi step ode model
    validation_multi_step_ode() 