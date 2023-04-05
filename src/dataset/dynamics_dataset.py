import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


class SingleStepDynamicsDataset(Dataset):
    """
    Each data sample is a dictionary containing (x_t, u_t, x_{t+1}) in the form:
    {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (state_size,)
     u_t: torch.float32 tensor of shape (action_size,)
     x_{t+1}: torch.float32 tensor of shape (state_size,)
    """

    def __init__(self, collected_data):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0]

    def __len__(self):
        return len(self.data) * self.trajectory_length

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None,
        }
        # --- Your code here
        trajectory_idx = item // self.trajectory_length
        action_idx = item % self.trajectory_length
        trajectory = self.data[trajectory_idx]
        if type(trajectory['actions']) is np.ndarray:
          action = torch.from_numpy(trajectory['actions'][action_idx])
          state = torch.from_numpy(trajectory['states'][action_idx])
          next_state = torch.from_numpy(trajectory['states'][action_idx+1])
        else:
          action = trajectory['actions'][action_idx]
          state = trajectory['states'][action_idx]
          next_state = trajectory['states'][action_idx+1]
        sample['state'] = state.float()
        sample['action'] = action.float()
        sample['next_state'] = next_state.float()
        # ---
        return sample


class MultiStepDynamicsDataset(Dataset):
    """
    Dataset containing multi-step dynamics data.

    Each data sample is a dictionary containing (state, action, next_state) in the form:
    {'state': x_t, -- initial state of the multipstep torch.float32 tensor of shape (state_size,)
     'action': [u_t,..., u_{t+num_steps-1}] -- actions applied in the muli-step.
                torch.float32 tensor of shape (num_steps, action_size)
     'next_state': [x_{t+1},..., x_{t+num_steps} ] -- next multiple steps for the num_steps next steps.
                torch.float32 tensor of shape (num_steps, state_size)
    }
    """

    def __init__(self, collected_data, num_steps=4):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return len(self.data) * (self.trajectory_length)

    def __iter__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)

    def __getitem__(self, item):
        """
        Return the data sample corresponding to the index <item>.
        :param item: <int> index of the data sample to produce.
            It can take any value in range 0 to self.__len__().
        :return: data sample corresponding to encoded as a dictionary with keys (state, action, next_state).
        The class description has more details about the format of this data sample.
        """
        sample = {
            'state': None,
            'action': None,
            'next_state': None
        }
        # --- Your code here
        trajectory_idx = item // self.trajectory_length
        action_idx = item % self.trajectory_length
        trajectory = self.data[trajectory_idx]
        if type(trajectory['actions']) is np.ndarray:
          action = torch.from_numpy(trajectory['actions'][action_idx:(action_idx+self.num_steps), :])
          state = torch.from_numpy(trajectory['states'][action_idx])
          next_state = torch.from_numpy(trajectory['states'][(action_idx+1):(action_idx+self.num_steps+1), :])
        else:
          action = trajectory['actions'][action_idx:(action_idx+self.num_steps), :]
          state = trajectory['states'][action_idx]
          next_state = trajectory['states'][(action_idx+1):(action_idx+self.num_steps+1), :]
        sample['state'] = state.float()
        sample['action'] = action.float()
        sample['next_state'] = next_state.float()

        # ---
        return sample