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


class SingleStepDynamicsDataset_FK(Dataset):
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
        self.data = collected_data # (trajlen * traj * num, 35)

    def __len__(self):
        return int(self.data.shape[0])

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
        if type(self.data) is np.ndarray:
          action = torch.from_numpy(self.data[item, 14:21])
          state = torch.from_numpy(self.data[item, :14])
          next_state = torch.from_numpy(self.data[item, 21:])
        else:
          action = self.data[item, 14:21]
          state = self.data[item, :14]
          next_state = self.data[item, 21:]
        
        sample['state'] = state.float()
        sample['action'] = action.float()
        sample['next_state'] = next_state.float()
        # ---
        return sample


class MultiStepDynamicsDataset_FK(Dataset):
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

    def __init__(self, collected_data, num_steps=4, trajectory_num=10):
        self.data = collected_data # (trajlen * trajnum, 35)
        self.trajectory_num = trajectory_num
        self.trajectory_length_full = int(self.data.shape[0] / trajectory_num)
        self.trajectory_length = int(self.data.shape[0] / trajectory_num) - num_steps + 1
        self.num_steps = num_steps

    def __len__(self):
        return self.trajectory_num * self.trajectory_length

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
        traj_idx = item // self.trajectory_length # From 0 to len
        data_idx = traj_idx * self.trajectory_length_full # To locate the start of the traj in data array
        if type(self.data) is np.ndarray:
            # state
            state = torch.from_numpy(self.data[data_idx, :14])
            # action
            action = []
            for step in range(self.num_steps):
                action.append(self.data[(data_idx+step):(data_idx+step+1), 14:21])
            action = np.stack(action, axis=0)
            action = torch.from_numpy(action)
            # next state
            next_state = []
            for step in range(self.num_steps):
                next_state.append(self.data[(data_idx+step):(datat_idx+step+1), 21:])
                next_state = np.stack(next_state, axis=0)
            next_state = torch.from_numpy(next_state)
        else:
            # state
            state = self.data[data_idx, :14]
            # action
            action = []
            for step in range(self.num_steps):
                action.append(self.data[(data_idx+step):(data_idx+step+1), 14:21])
            action = np.stack(action, axis=0)
            # next state
            next_state = []
            for step in range(self.num_steps):
                next_state.append(self.data[(data_idx+step):(datat_idx+step+1), 21:])
                next_state = np.stack(next_state, axis=0)
        
        sample['state'] = state.float()
        sample['action'] = action.float()
        sample['next_state'] = next_state.float()

        # ---
        return sample