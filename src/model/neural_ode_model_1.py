import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import sys
sys.path.append("..")

# from neural_ode_learning import ODEFunc, ProjectionNN
from torchdiffeq import odeint_adjoint as odeint

class ODEFunc(nn.Module):
    def __init__(self, state_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.Tanh(),
            nn.Linear(100, state_dim)
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
    def __init__(self, state_dim, action_dim):
        super(ProjectionNN, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim)
        )
    def forward(self, state_action):
        state = self.proj(state_action)

        return state

class Encoder(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Encoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim)
        )
    def forward(self, state_action):
        latent_state = self.proj(state_action)

        return latent_state
  
class Decoder(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Decoder, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(state_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim)
        )
    def forward(self, latent_state):
        state = self.proj(latent_state)

        return state


# NeuralODE
class NeuralODE(nn.Module):
  def __init__(self, state_dim, action_dim, device = 'cpu'):
    super().__init__()
    self.device = device
    self.odefunc = ODEFunc(state_dim).to(device)
    # self.odefunc.load_state_dict(torch.load(ode_pth_path))
    self.encoder = Encoder(state_dim, action_dim).to(device)
    self.decoder = Decoder(state_dim, action_dim).to(device)
    # self.projnn.load_state_dict(torch.load(proj_pth_path))

  def forward(self, state, action, T):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim) (B, 3)
      :param action: torch tensor of shape (..., action_dim) (B, 3)
      :return: next_state: torch tensor of shape (..., state_dim) (B, 3)
      """
      next_state = None
      state= state.to(self.device)
      action= action.to(self.device)
      state_action = torch.cat((state, action), dim=1)
      latent_space = self.encoder(state_action)
      latent_state_ode= odeint(self.odefunc, latent_space, T) # (10, B, 3)
      next_state =  self.decoder(latent_state_ode[-1]) # (B, 3)
    #   print(next_state.shape)
    #   next_state = next_state[0]

      return next_state
  
# NeuralODE_multi step
class NeuralODE_1(nn.Module):
  def __init__(self, state_dim, action_dim, device = 'cpu'):
    super().__init__()
    self.device = device
    self.odefunc = ODEFunc(state_dim).to(device)
    # self.odefunc.load_state_dict(torch.load(ode_pth_path))
    self.projnn = ProjectionNN(state_dim, action_dim).to(device)
    # self.projnn.load_state_dict(torch.load(proj_pth_path))

  def forward(self, state_action, T):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim) (B, 3)
      :param action: torch tensor of shape (..., action_dim) (B, 3)
      :return: next_state: torch tensor of shape (..., state_dim) (B, 3)
      """
      next_state_ode = None
      state_action= state_action.to(self.device)
      encoder = self.projnn(state_action)
      next_state_ode= odeint(self.odefunc, encoder, T) # (T, B, 3)
    #   state_action = torch.cat((next_state_ode[-1], action), dim=1)
    #   next_state =  state + self.projnn(state_action) # (10, B, 3)
    #   print(next_state.shape)
    #   next_state = next_state[0]

      return next_state_ode