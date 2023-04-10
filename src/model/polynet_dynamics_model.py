import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# PolyInception-based Dynamics Model
class Poly_2_DynamicsModel(nn.Module):
  """
  Model the dynamics using poly-2 structures s_{t+1} = s_{t} + f(s_{t}) + f(f(s_{t}))
  """
  def __init__(self, state_dim, action_dim, device = 'cpu'):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.device = device
    self.f = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100), # input layer
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim) # output layer
    )
    # self.relu = nn.ReLU()
    self.beta_1 = nn.Parameter(torch.FloatTensor(1)) # dampening factor in order to prevent unstable training process
    self.beta_1.data.fill_(0.5)
    self.beta_2 = nn.Parameter(torch.FloatTensor(1))
    self.beta_2.data.fill_(0.4)

  def forward(self, state, action):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim)
      :param action: torch tensor of shape (..., action_dim)
      :return: next_state: torch tensor of shape (..., state_dim)
      """
      next_state = None
      state = state.to(self.device)
      action = action.to(self.device)
      state_action_input_f = torch.cat((state, action), dim=1)
      res_state_f = self.f(state_action_input_f)
      state_action_input_f_f = torch.cat((res_state_f, action), dim=1)
      res_state_f_f = self.f(state_action_input_f_f)

      next_state = state + self.beta_1 * res_state_f + self.beta_2 * res_state_f_f
      # next_state = state + self.beta_1 * res_state_f

      return next_state
  

  # PolyInception-based Dynamics Model
class mPoly_2_DynamicsModel(nn.Module):
  """
  Model the dynamics using mpoly-2 structures s_{t+1} = s_{t} + f(s_{t}) + g(f(s_{t}))
  """
  def __init__(self, state_dim, action_dim, device = 'cpu'):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.device = device
    self.f = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100), # input layer
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim) # output layer
    )
    self.g = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100), # input layer
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim) # output layer
    )
    # self.relu = nn.ReLU()
    self.beta_1 = nn.Parameter(torch.FloatTensor(1)) # dampening factor in order to prevent unstable training process
    self.beta_1.data.fill_(0.5)
    self.beta_2 = nn.Parameter(torch.FloatTensor(1))
    self.beta_2.data.fill_(0.5)

  def forward(self, state, action):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim)
      :param action: torch tensor of shape (..., action_dim)
      :return: next_state: torch tensor of shape (..., state_dim)
      """
      next_state = None
      state = state.to(self.device)
      action = action.to(self.device)
      state_action_input_f = torch.cat((state, action), dim=1)
      res_state_f = self.f(state_action_input_f)
      state_action_input_g = torch.cat((res_state_f, action), dim=1)
      res_state_g = self.g(state_action_input_g)
      next_state = state + self.beta_1 * res_state_f + self.beta_2 * res_state_g

      return next_state


  # PolyInception-based Dynamics Model
class way_2_DynamicsModel(nn.Module):
  """
  Model the dynamics using mpoly-2 structures s_{t+1} = s_{t} + f(s_{t}) + g(s_{t})
  """
  def __init__(self, state_dim, action_dim, device = 'cpu'):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.device = device
    self.f = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100), # input layer
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim) # output layer
    )
    self.g = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100), # input layer
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim) # output layer
    )
    # self.relu = nn.ReLU()
    self.beta_1 = nn.Parameter(torch.FloatTensor(1)) # dampening factor in order to prevent unstable training process
    self.beta_1.data.fill_(0.5)
    self.beta_2 = nn.Parameter(torch.FloatTensor(1))
    self.beta_2.data.fill_(0.5)

  def forward(self, state, action):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim)
      :param action: torch tensor of shape (..., action_dim)
      :return: next_state: torch tensor of shape (..., state_dim)
      """
      next_state = None
      state = state.to(self.device)
      action = action.to(self.device)
      state_action_input = torch.cat((state, action), dim=1)
      res_state_f = self.f(state_action_input)
      res_state_g = self.g(state_action_input)
      # next_state = self.relu(state + self.beta * res_state_f + self.beta * res_state_g)
      next_state = state + self.beta_1 * res_state_f + self.beta_2 * res_state_g

      return next_state