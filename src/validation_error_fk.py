import torch
import numpy as np
import os
from dataset.dynamics_dataset import SingleStepDynamicsDataset, MultiStepDynamicsDataset
from dataset.dynamics_dataset import SingleStepDynamicsDataset_FK, MultiStepDynamicsDataset_FK
from dataset.loss import SE2PoseLoss, SingleStepLoss, MultiStepLoss, SingleStepLoss_ode, MultiStepLoss_ode
from model.neural_ode_model import NeuralODE
from model.absolute_dynamics_model import AbsoluteDynamicsModel
from model.residual_dynamics_model import ResidualDynamicsModel
from model.polynet_dynamics_model import Poly_2_DynamicsModel, mPoly_2_DynamicsModel, way_2_DynamicsModel
from model.fractalnet_dynamics_model import RKNN_2_DynamicsModel

# Load the collected data: 
data_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/data/FK'
validation_data = np.load(os.path.join(data_path, 'BaxterDirectDynamics_val.npy'), allow_pickle=True)

# Device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
Single Step model
"""
def validation_single_step():
    # dataset
    val_dataset = SingleStepDynamicsDataset_FK(validation_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

    # model
    model_pth_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt/FK/Baxter/pushing_mpoly_dynamics_model_single_step.pt'
    state_dim = 14
    action_dim = 7
    model = mPoly_2_DynamicsModel(state_dim, action_dim)
    model.load_state_dict(torch.load(model_pth_path))

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SingleStepLoss(pose_loss)

    loss = 0.0

    for item in val_loader:
        state = item['state']
        action = item['action']
        next_state = item['next_state']
        loss += pose_loss(model, state, action, next_state)

    print(f'Validation loss for model is {loss:.8f}')


# Single step mode ode
def validation_single_step_ode():
    # dataset
    val_dataset = SingleStepDynamicsDataset_FK(validation_data)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

    # model
    ode_pth_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt/FK/Baxter/ODEFunc_multi_step_rk4_4.pt'
    state_dim = 14
    action_dim = 7
    model = NeuralODE(state_dim, action_dim, method='rk4')
    model.load_state_dict(torch.load(ode_pth_path))

    pose_loss = SE2PoseLoss(block_width=0.1, block_length=0.1)
    pose_loss = SingleStepLoss_ode(pose_loss)

    loss_ode = 0.0

    for item in val_loader:
        state = item['state']
        action = item['action']
        next_state = item['next_state']
        loss_ode += pose_loss(model, state, action, next_state)

    print(f'Validation loss for Neural ODE model is {loss_ode:.8f}') 


if __name__ == "__main__":
    # single step model
    # validation_single_step()

    # Multi step model
    # validation_multi_step()

    # single step ode model
    validation_single_step_ode()

    # multi step ode model
    # validation_multi_step_ode() 