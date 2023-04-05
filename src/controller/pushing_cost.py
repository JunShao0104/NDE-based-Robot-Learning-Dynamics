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