import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# FractalNet
class RKNN_2_DynamicsModel(nn.Module):
  """
  Model the dynamics using RKNN-2-order structures
  s_{t+1} = s_{t} + 1 / 2 * (f(s_{t}) + f(s_{t} + f(s_{t})))
  """
  def __init__(self, state_dim, action_dim):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
    self.f = nn.Sequential(
            nn.Linear(state_dim+action_dim, 100), # input layer
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, state_dim) 
    )
    # self.output = nn.Linear(state_dim+action_dim, state_dim) # output layer
    # self.relu = nn.ReLU()
    self.beta_1 = nn.Parameter(torch.FloatTensor(1)) # dampening factor in order to prevent unstable training process
    self.beta_1.data.fill_(0.5)
    # self.beta_2 = nn.Parameter(torch.FloatTensor(1))
    # self.beta_2.data.fill_(0.5)

  def forward(self, state, action):
      """
      Compute next_state resultant of applying the provided action to provided state
      :param state: torch tensor of shape (..., state_dim)
      :param action: torch tensor of shape (..., action_dim)
      :return: next_state: torch tensor of shape (..., state_dim)
      """
      next_state = None
      state_action_input = torch.cat((state, action), dim=1)
      K1 = self.f(state_action_input)
      K1_action = torch.cat((K1, action), dim=1)
      K2 = self.f(state_action_input + K1_action)
      # next_state = self.output(state_action_input + self.beta_1 * K1 + self.beta_2 * K2)
      next_state = state + self.beta_1 * K1 + (1 - self.beta_1) * K2

      return next_state