import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim, device = 'cpu'):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        # --- Your code here
        self.linear1 = nn.Linear(state_dim+action_dim, 100)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(100, state_dim)

        # ---

    def forward(self, state, action):
        """
        Compute next_state resultant of applying the provided action to provided state
        :param state: torch tensor of shape (..., state_dim)
        :param action: torch tensor of shape (..., action_dim)
        :return: next_state: torch tensor of shape (..., state_dim)
        """
        next_state = None
        # --- Your code here
        state = state.to(self.device)
        action = action.to(self.device)
        state_action = torch.cat((state, action), dim=1)
        next_state = self.linear1(state_action)
        next_state = self.relu1(next_state)
        next_state = self.linear2(next_state)
        next_state = self.relu2(next_state)
        next_state = self.linear3(next_state)

        # ---
        return next_state