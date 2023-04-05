import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# from tqdm import tqdm
from panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, OBSTACLE_HALFDIMS, OBSTACLE_CENTRE, BOX_SIZE

TARGET_POSE_FREE_TENSOR = torch.as_tensor(TARGET_POSE_FREE, dtype=torch.float32)
TARGET_POSE_OBSTACLES_TENSOR = torch.as_tensor(TARGET_POSE_OBSTACLES, dtype=torch.float32)
OBSTACLE_CENTRE_TENSOR = torch.as_tensor(OBSTACLE_CENTRE, dtype=torch.float32)[:2]
OBSTACLE_HALFDIMS_TENSOR = torch.as_tensor(OBSTACLE_HALFDIMS, dtype=torch.float32)[:2]


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
    train_dataset, val_dataset = random_split(dataset=entire_dataset, lengths=[0.8, 0.2])
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
    train_loader = None
    val_loader = None
    # --- Your code here
    entire_dataset = MultiStepDynamicsDataset(collected_data, num_steps)
    # len_train_dataset = int(len(entire_dataset)*0.8)
    # len_val_dataset = len(entire_dataset) - len_train_dataset
    train_dataset, val_dataset = random_split(dataset=entire_dataset, lengths=[0.8, 0.2])
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # ---
    return train_loader, val_loader


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


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        rg = ((self.w**2 + self.l**2)/12)**0.5
        x_loss = F.mse_loss(pose_pred[:, 0], pose_target[:, 0])
        y_loss = F.mse_loss(pose_pred[:, 1], pose_target[:, 1])
        theta_loss = rg * F.mse_loss(pose_pred[:, 2], pose_target[:, 2])
        se2_pose_loss = x_loss + y_loss + theta_loss

        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        pred_state = model(state, action)
        single_step_loss = self.loss(pred_state, target_state)

        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        # state: (batchsize, state_size)
        # actions: (batchsize, num_steps, action_size)
        # target_states: (batchsize, num_steps, state_size)
        multi_step_loss = 0.0
        num_steps = int(actions.shape[1])
        current_state = state
        discount = 1.0
        for i in range(num_steps):
          pred_state = model(current_state, actions[:, i, :])
          single_step_loss = discount * self.loss(pred_state, target_states[:, i, :])
          multi_step_loss += single_step_loss
          current_state = pred_state
          discount *= self.discount

        # ---
        return multi_step_loss


class AbsoluteDynamicsModel(nn.Module):
    """
    Model the absolute dynamics x_{t+1} = f(x_{t},a_{t})
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
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
        state_action = torch.cat((state, action), dim=1)
        next_state = self.linear1(state_action)
        next_state = self.relu1(next_state)
        next_state = self.linear2(next_state)
        next_state = self.relu2(next_state)
        next_state = self.linear3(next_state)

        # ---
        return next_state


# PolyInception-based Dynamics Model
class Poly_2_DynamicsModel(nn.Module):
  """
  Model the dynamics using poly-2 structures s_{t+1} = s_{t} + f(s_{t}) + f(f(s_{t}))
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
  def __init__(self, state_dim, action_dim):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
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
  def __init__(self, state_dim, action_dim):
    super().__init__()
    self.state_dim = state_dim
    self.action_dim = action_dim
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
      state_action_input = torch.cat((state, action), dim=1)
      res_state_f = self.f(state_action_input)
      res_state_g = self.g(state_action_input)
      # next_state = self.relu(state + self.beta * res_state_f + self.beta * res_state_g)
      next_state = state + self.beta_1 * res_state_f + self.beta_2 * res_state_g

      return next_state


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


class ResidualDynamicsModel(nn.Module):
    """
    Model the residual dynamics s_{t+1} = s_{t} + f(s_{t}, u_{t})

    Observation: The network only needs to predict the state difference as a function of the state and action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        # --- Your code here
        self.linear1 = nn.Linear(state_dim+action_dim, 100) # input layer
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(100, 100)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(100, state_dim) # output layer

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
        state_action = torch.cat((state, action), dim=1)
        res_state = self.linear1(state_action)
        res_state = self.relu1(res_state)
        res_state = self.linear2(res_state)
        res_state = self.relu2(res_state)
        res_state = self.linear3(res_state)
        next_state = state + res_state

        # ---
        return next_state


def free_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup without obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_FREE_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    B = int(state.shape[0])
    target_pose = torch.unsqueeze(target_pose, 0)
    target_pose = target_pose.repeat(B, 1)
    Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]])
    cost = torch.matmul((state-target_pose), Q) # (B, 3) @ (3, 3) = (B, 3)
    cost = torch.matmul(cost, (state-target_pose).T) # (B, 3) @ (3, B) = (B, B)
    cost = torch.diagonal(cost, 0)

    # ---
    return cost


def collision_detection(state):
    """
    Checks if the state is in collision with the obstacle.
    The obstacle geometry is known and provided in obstacle_centre and obstacle_halfdims.
    :param state: torch tensor of shape (B, state_size)
    :return: in_collision: torch tensor of shape (B,) containing 1 if the state is in collision and 0 if not.
    """
    obstacle_centre = OBSTACLE_CENTRE_TENSOR  # torch tensor of shape (2,) consisting of obstacle centre (x, y)
    obstacle_dims = 2 * OBSTACLE_HALFDIMS_TENSOR  # torch tensor of shape (2,) consisting of (w_obs, l_obs)
    box_size = BOX_SIZE  # scalar for parameter w
    in_collision = None
    # --- Your code here
    obstacle_centre = torch.unsqueeze(obstacle_centre, 0).repeat(state.shape[0], 1) # (B, 2)
    centre_dist = torch.zeros((state.shape[0], 2)) # (B, 2)
    centre_dist[:, 0] = torch.abs(state[:, 0] - obstacle_centre[:, 0])
    centre_dist[:, 1] = torch.abs(state[:, 1] - obstacle_centre[:, 1])
    obstacle_halfdims = (obstacle_dims / 2).unsqueeze(0).repeat(state.shape[0], 1) # (B, 2)
    object_halfdims = max_dims(state, box_size) # (B, 2)
    
    res_dist = centre_dist - obstacle_halfdims - object_halfdims # (B, 2)
    res_mask = res_dist < 0 # (B, 2)
    bool_mask = res_mask[:, 0] * res_mask[:, 1] # If x and y are both < 0, then True * True = True; else, False
    in_collision = torch.ones((state.shape[0]))
    in_collision *= bool_mask

    # ---
    return in_collision


def obstacle_avoidance_pushing_cost_function(state, action):
    """
    Compute the state cost for MPPI on a setup with obstacles.
    :param state: torch tensor of shape (B, state_size)
    :param action: torch tensor of shape (B, state_size)
    :return: cost: torch tensor of shape (B,) containing the costs for each of the provided states
    """
    target_pose = TARGET_POSE_OBSTACLES_TENSOR  # torch tensor of shape (3,) containing (pose_x, pose_y, pose_theta)
    cost = None
    # --- Your code here
    # Obstacle free cost
    B = int(state.shape[0])
    target_pose = torch.unsqueeze(target_pose, 0)
    target_pose = target_pose.repeat(B, 1)
    Q = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 0.1]])
    cost = torch.matmul((state-target_pose), Q) # (B, 3) @ (3, 3) = (B, 3)
    cost = torch.matmul(cost, (state-target_pose).T) # (B, 3) @ (3, B) = (B, B)
    cost = torch.diagonal(cost, 0) # (B, )

    # Obstacle collision cost
    in_collision = 100 * collision_detection(state) # (B, )
    new_cost = cost
    new_cost += in_collision
    # print(torch.sum(new_cost-cost))

    # ---
    return new_cost


class PushingController(object):
    """
    MPPI-based controller
    Since you implemented MPPI on HW2, here we will give you the MPPI for you.
    You will just need to implement the dynamics and tune the hyperparameters and cost functions.
    """

    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        self.target_state = None
        # MPPI Hyperparameters:
        # --- You may need to tune them
        state_dim = env.observation_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low)
        u_max = torch.from_numpy(env.action_space.high)
        noise_sigma = 0.5 * torch.eye(env.action_space.shape[0])
        lambda_value = 0.01
        # ---
        from mppi import MPPI
        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max)

    def _compute_dynamics(self, state, action):
        """
        Compute next_state using the dynamics model self.model and the provided state and action tensors
        :param state: torch tensor of shape (B, state_size)
        :param action: torch tensor of shape (B, action_size)
        :return: next_state: torch tensor of shape (B, state_size) containing the predicted states from the learned model.
        """
        next_state = None
        # --- Your code here
        next_state = self.model(state, action)

        # ---
        return next_state

    def control(self, state):
        """
        Query MPPI and return the optimal action given the current state <state>
        :param state: numpy array of shape (state_size,) representing current state
        :return: action: numpy array of shape (action_size,) representing optimal action to be sent to the robot.
        TO DO:
         - Prepare the state so it can be send to the mppi controller. Note that MPPI works with torch tensors.
         - Unpack the mppi returned action to the desired format.
        """
        action = None
        state_tensor = None
        # --- Your code here
        state_tensor = torch.from_numpy(state)

        # ---
        action_tensor = self.mppi.command(state_tensor)
        # --- Your code here
        action = action_tensor.detach().numpy()

        # ---
        return action

# =========== AUXILIARY FUNCTIONS AND CLASSES HERE ===========
# --- Your code here
def max_dims(state, box_size):
    """
    :param state: torch tensor of shape (B, state_size)
    :param box_size: float
    :return max_half_dims: torch tensor of shape (B, 2)
    """
    # Compute the coordinates of four conors
    cornor_dist = ((box_size/2)**2 + (box_size/2)**2)**0.5
    # right top cornor
    right_top_coord = torch.zeros((state.shape[0], 2))
    right_top_coord[:, 0] = state[:, 0]+cornor_dist*torch.cos(deg2rad(45)+state[:, 2])
    right_top_coord[:, 1] = state[:, 1]+cornor_dist*torch.sin(deg2rad(45)+state[:, 2])
    # left top cornor
    left_top_coord = torch.zeros((state.shape[0], 2))
    left_top_coord[:, 0] = state[:, 0]+cornor_dist*torch.cos(deg2rad(135)+state[:, 2])
    left_top_coord[:, 1] = state[:, 1]+cornor_dist*torch.sin(deg2rad(135)+state[:, 2])
    # right bottom cornor
    right_bottom_coord = torch.zeros((state.shape[0], 2))
    right_bottom_coord[:, 0] = state[:, 0]+cornor_dist*torch.cos(deg2rad(-45)+state[:, 2])
    right_bottom_coord[:, 1] = state[:, 1]+cornor_dist*torch.sin(deg2rad(-45)+state[:, 2])
    # left bottom cornor
    left_bottom_coord = torch.zeros((state.shape[0], 2))
    left_bottom_coord[:, 0] = state[:, 0]+cornor_dist*torch.cos(deg2rad(-135)+state[:, 2])
    left_bottom_coord[:, 1] = state[:, 1]+cornor_dist*torch.sin(deg2rad(-135)+state[:, 2])

    # Compute the max half x and max half y
    # coord_x: (B, 4)
    coord_x = torch.zeros((state.shape[0], 4))
    coord_x[:, 0] = right_top_coord[:, 0]
    coord_x[:, 1] = left_top_coord[:, 0]
    coord_x[:, 2] = right_bottom_coord[:, 0]
    coord_x[:, 3] = left_bottom_coord[:, 0]
    # coord_y: (B, 4)
    coord_y = torch.zeros((state.shape[0], 4))
    coord_y[:, 0] = right_top_coord[:, 1]
    coord_y[:, 1] = left_top_coord[:, 1]
    coord_y[:, 2] = right_bottom_coord[:, 1]
    coord_y[:, 3] = left_bottom_coord[:, 1]
    # Compute the max x and max y
    max_x = torch.max(coord_x, 1)[0] # (B, )
    max_y = torch.max(coord_y, 1)[0] # (B, )
    max_half_x = (max_x - state[:, 0]).unsqueeze(1) # (B, 1)
    max_half_y = (max_y - state[:, 1]).unsqueeze(1) # (B, 1)
    max_half_dims = torch.cat((max_half_x, max_half_y), dim=1) # (B, 2)

    return max_half_dims


def deg2rad(degree):
    return torch.Tensor([degree / 180 * torch.pi])

# ---
# ============================================================
