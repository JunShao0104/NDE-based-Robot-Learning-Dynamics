import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys
sys.path.append("..")

from torchdiffeq import odeint_adjoint as odeint

# func
class ODEFunc_1(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ODEFunc_1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Linear(in_channels, out_channels)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    
    def forward(self, t, y):
        pred_y = self.net(y)

        return pred_y


class ODEFunc_2(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ODEFunc_2, self).__init__()
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


class ODEFunc_3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ODEFunc_3, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Linear(in_channels, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
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


# NeuralODE
class NeuralODE(nn.Module):
  def __init__(self, state_dim=3, action_dim=3, method = None):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.method = method

    # func 1
    # self.odefunc = ODEFunc_1(state_dim+action_dim, state_dim+action_dim)

    # func 2
    # self.odefunc = ODEFunc_2(state_dim+action_dim, state_dim+action_dim)

    # func 3
    self.odefunc = ODEFunc_3(state_dim+action_dim, state_dim+action_dim)

  def forward(self, state, action):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim) (B, 3)
      :param action: torch tensor of shape (..., action_dim) (B, 3)
      :return: next_state: torch tensor of shape (..., state_dim) (B, 3)
      """
      next_state = None
      state_action = torch.cat((state, action), dim=1) # (B, 6)
      t = torch.arange(2).float() # (2, )
      # Compute
      if(self.method):
        # print(self.method)
        next_state_action = odeint(self.odefunc, state_action, t, method=self.method, options=dict(step_size=0.5)) # (2, B, 6)
      else:
        next_state_action = odeint(self.odefunc, state_action, t)
      next_state = next_state_action[-1, :, :self.state_dim] # (B, 3)

      return next_state