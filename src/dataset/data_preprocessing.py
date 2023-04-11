import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from dataset.dynamics_dataset import SingleStepDynamicsDataset, MultiStepDynamicsDataset

def collect_data_random(env, num_trajectories=1000, trajectory_length=10):
    """
    Collect data from the provided environment using uniformly random exploration.
    :param env: Gym Environment instance.
    :param num_trajectories: <int> number of data to be collected.
    :param trajectory_length: <int> number of state transitions to be collected
    :return: collected data: List of dictionaries containing the state-action trajectories.
    Each trajectory dictionary should have the following structure:
        {'states': states,
        'actions': actions}
    where
        * states is a numpy array of shape (trajectory_length+1, state_size) containing the states [x_0, ...., x_T]
        * actions is a numpy array of shape (trajectory_length, actions_size) containing the actions [u_0, ...., u_{T-1}]
    Each trajectory is:
        x_0 -> u_0 -> x_1 -> u_1 -> .... -> x_{T-1} -> u_{T_1} -> x_{T}
        where x_0 is the state after resetting the environment with env.reset()
    All data elements must be encoded as np.float32.
    """
    collected_data = None
    # --- Your code here
    collected_data = []
    for i in range(num_trajectories):
      trajectory = dict()

      # Check the size of state
      x_0 = env.reset()
      # state_size = x_0.shape[0]
      # assert state_size == 3

      # Check the size of action
      # sample_action = env.action_space.sample()
      # action_size = sample_action.shape[0]
      # assert action_size == 3

      # Initialize states and actions
      states = []
      actions = []
      states.append(x_0)
      # For loop to create the data
      for j in range(trajectory_length):
        action_j = env.action_space.sample()
        state_j_1, _, done, _ = env.step(action_j)
        actions.append(action_j)
        states.append(state_j_1)
      
      # Form dict
      states = np.vstack(states)
      actions = np.vstack(actions)
      trajectory['states'] = states.astype(np.float32)
      trajectory['actions'] = actions.astype(np.float32)
      collected_data.append(trajectory)

    # ---
    return collected_data


def process_data_single_step(collected_data, batch_size=500):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as {'state': x_t,
     'action': u_t,
     'next_state': x_{t+1},
    }
    where:
     x_t: torch.float32 tensor of shape (batch_size, state_size)
     u_t: torch.float32 tensor of shape (batch_size, action_size)
     x_{t+1}: torch.float32 tensor of shape (batch_size, state_size)

    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement SingleStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    entire_dataset = SingleStepDynamicsDataset(collected_data)
    # len_train_dataset = int(len(entire_dataset)*0.8)
    # len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(entire_dataset, [0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # ---
    return train_loader, val_loader


def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    """
    Process the collected data and returns a DataLoader for train and one for validation.
    The data provided is a list of trajectories (like collect_data_random output).
    Each DataLoader must load dictionary as
    {'state': x_t,
     'action': u_t, ..., u_{t+num_steps-1},
     'next_state': x_{t+1}, ... , x_{t+num_steps}
    }
    where:
     state: torch.float32 tensor of shape (batch_size, state_size)
     next_state: torch.float32 tensor of shape (batch_size, num_steps, action_size)
     action: torch.float32 tensor of shape (batch_size, num_steps, state_size)

    Each DataLoader must load dictionary dat
    The data should be split in a 80-20 training-validation split.
    :param collected_data:
    :param batch_size: <int> size of the loaded batch.
    :param num_steps: <int> number of steps to load the multistep data.
    :return:

    Hints:
     - Pytorch provides data tools for you such as Dataset and DataLoader and random_split
     - You should implement MultiStepDynamicsDataset below.
        This class extends pytorch Dataset class to have a custom data format.
    """
    print(torch.__version__)
    train_loader = None
    val_loader = None
    # --- Your code here
    entire_dataset = MultiStepDynamicsDataset(collected_data, num_steps)
    # len_train_dataset = int(len(entire_dataset)*0.8)
    # len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(entire_dataset,[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---
    return train_loader, val_loader

                                                                                                                                                                                                                                                                                                                                                                                                                             
def process_data_continuous_batch(collected_data):
    """
    Process the collected data and returns batch data for train.
    The data provided is a list of trajectories (like collect_data_random output).
    : param collected_data: dict of dict. len(dict) = num trajectory. each dict: "states" (T+1, S); "actions" (T, A)
    Return: batch_y0, batch_t, batch_y
    batch_y0: (M, D). M is the bacth size, num trajectory. D is the concatenation dimension of state and action. D = S + A
    batch_t: (T, ). 1-D tensor. T is the trajectory length.
    batch_y: (T, M, D).
    """
    M = len(collected_data) # batch size; num trajectory
    T = collected_data[0]['actions'].shape[0] # time step, trajectory length
    ode_T = 3
    S = collected_data[0]['states'].shape[1] # state dimension
    A = collected_data[0]['actions'].shape[1] # action dimension

    # Form batch_y0
    batch_y0 = None
    for m in np.arange(0,M):
        for n in np.arange(0, T, ode_T):
            state_action = torch.cat((torch.from_numpy(collected_data[m]['states'][n:n+1, :]), torch.from_numpy(collected_data[m]['actions'][n:n+1, :])), dim=1) # (1, D)
            if batch_y0 is None:
                batch_y0 = state_action
            else:
                batch_y0 = torch.vstack((batch_y0, state_action)) # (M/odeT, D)
            batch_t = torch.arange(m, m+ode_T).float() # (odeT, )
    
    # Form batch_t

    # Form batch_y
    batch_y = None
    for m in range(M):
        state_T = torch.from_numpy(collected_data[m]['states'][:ode_T, :]) # (T, S)
        action_T = torch.from_numpy(collected_data[m]['actions'][:ode_T, :]) # (T, A)
        state_action_T = torch.cat((state_T, action_T), dim=1) # (T, D)
        if batch_y is None:
            batch_y = state_action_T.unsqueeze(1)
        else:
            batch_y = torch.cat((batch_y, state_action_T.unsqueeze(1)), dim=1) # (T, M, D)
    

    return batch_y0, batch_t, batch_y

def process_data_ode_batch(collected_data, batch_size=500, num_steps=4,ode_t=3,device = 'cpu'):
    """
    Process the collected data and returns batch data for train.
    The data provided is a list of trajectories (like collect_data_random output).
    : param collected_data: dict of dict. len(dict) = num trajectory. each dict: "states" (T+1, S); "actions" (T, A)
    Return: batch_y0, batch_t, batch_y
    batch_y0: (M, D). M is the bacth size, num trajectory. D is the concatenation dimension of state and action. D = S + A
    batch_t: (T, ). 1-D tensor. T is the trajectory length.
    batch_y: (T, M, D).
    """
    train_loader = None
    val_loader = None
    # --- Your code here
    entire_dataset = ODE_Batch_StepDynamicsDataset(collected_data, num_steps, ode_t, device)
    # len_train_dataset = int(len(entire_dataset)*0.8)
    # len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(entire_dataset,[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    return train_loader, val_loader


class ODE_Batch_StepDynamicsDataset(Dataset):
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

    def __init__(self, collected_data, num_steps=4, ode_t = 3, device = 'cpu'):
        self.data = collected_data
        self.trajectory_length = self.data[0]['actions'].shape[0] - num_steps + 1
        self.num_steps = num_steps
        self.ode_t = ode_t # num of samples per step
        self.device = device

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
            'next_state': None,
            't': None
        }
        # --- Your code here
        trajectory_idx = item // self.trajectory_length
        action_idx = item % self.trajectory_length
        trajectory = self.data[trajectory_idx]
        action = torch.from_numpy(trajectory['actions'][action_idx:(action_idx+self.num_steps), :])
        state = torch.from_numpy(trajectory['states'][action_idx])
        next_state = torch.from_numpy(trajectory['states'][(action_idx+1):(action_idx+self.num_steps+1), :])
        sample['state'] = state.float().to(self.device) #current state
        sample['action'] = action.float().to(self.device) #actions
        sample['next_state'] = next_state.float().to(self.device)  #next states
        # create t from time  **this includes start and end points
        sample['t'] = torch.from_numpy(np.arange(action_idx,(action_idx+self.num_steps+1/(self.ode_t-1)), 1/(self.ode_t-1))).float().to(self.device)
        # print(sample['t'])
        # ---
        return sample