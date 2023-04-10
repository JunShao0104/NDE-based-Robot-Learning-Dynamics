import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys
sys.path.append("..")

from neural_ode_learning import ODEFunc, ProjectionNN
from torchdiffeq import odeint_adjoint as odeint

# NeuralODE
class NeuralODE(nn.Module):
  def __init__(self, ode_pth_path, proj_pth_path):
    super().__init__()
    self.odefunc = ODEFunc()
    self.odefunc.load_state_dict(torch.load(ode_pth_path))
    self.projnn = ProjectionNN()
    self.projnn.load_state_dict(torch.load(proj_pth_path))

  def forward(self, state, action):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim) (B, 3)
      :param action: torch tensor of shape (..., action_dim) (B, 3)
      :return: next_state: torch tensor of shape (..., state_dim) (B, 3)
      """
      next_state = None
      state_action = torch.cat((state, action), dim=1) # (B, 6)
      t = torch.arange(10).float() # (10, )
      next_state_action = odeint(self.odefunc, state_action, t) # (10, B, 6)
      next_state = self.projnn(next_state_action.reshape(-1, next_state_action.shape[2])).reshape(t.shape[0], state.shape[0], -1) # (10, B, 3)
      next_state = next_state[1]

      return next_state