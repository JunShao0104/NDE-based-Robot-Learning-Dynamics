import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from src.dataset.dynamics_dataset import SingleStepDynamicsDataset, MultiStepDynamicsDataset
from src.dataset.dynamics_dataset import SingleStepDynamicsDataset_FK, MultiStepDynamicsDataset_FK

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
    len_train_dataset = int(len(entire_dataset)*0.8)
    len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(dataset=entire_dataset, lengths=[len_train_dataset, len_val_dataset])
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
    print(num_steps)
    train_loader = None
    val_loader = None
    # --- Your code here
    entire_dataset = MultiStepDynamicsDataset(collected_data, num_steps)
    len_train_dataset = int(len(entire_dataset)*0.8)
    len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(dataset=entire_dataset, lengths=[len_train_dataset, len_val_dataset])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---
    return train_loader, val_loader


def process_data_single_step_FK(collected_data, batch_size=1000):
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
    entire_dataset = SingleStepDynamicsDataset_FK(collected_data)
    len_train_dataset = int(len(entire_dataset)*0.8)
    len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(dataset=entire_dataset, lengths=[len_train_dataset, len_val_dataset])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    # ---
    return train_loader, val_loader


def process_data_multiple_step_FK(collected_data, batch_size=1000, num_steps=4):
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
    train_loader = None
    val_loader = None
    # --- Your code here
    entire_dataset = MultiStepDynamicsDataset_FK(collected_data, num_steps, trajectory_num=10)
    train_dataset, val_dataset = random_split(dataset=entire_dataset, lengths=[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---
    return train_loader, val_loader