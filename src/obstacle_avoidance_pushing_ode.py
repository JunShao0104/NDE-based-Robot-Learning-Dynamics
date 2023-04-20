import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from numpngw import write_apng
from IPython.display import Image
from tqdm.notebook import tqdm
from src.env.panda_pushing_env import PandaPushingEnv
from src.utils.visualizers import GIFVisualizer, NotebookVisualizer

# Model
from src.model.neural_ode_model import NeuralODE

# Cost function and pushing controller
from src.controller.pushing_controller import PushingController
from src.controller.pushing_cost import collision_detection
from src.controller.pushing_cost import free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function
from src.env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE


def obstacle_avoidance_pushing_ode_single_step(method, path):
    # GIF Visualizer
    visualizer = GIFVisualizer()

    # Path
    ckpt_path = os.path.join(path,'/ckpt/Panda_pushing/continuous')

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=2, path = path)
    print("Currenlty using ODE Solver: ", method if method else "dopri5")
    # Load the pushing dynamics model
    ode_pth_path = os.path.join(ckpt_path, 'ODEFunc_single_step_{}.pt'.format(method if method else "dopri5"))
    state_dim = 3
    action_dim = 3
    NeuralODE_model = NeuralODE(state_dim=state_dim, action_dim=action_dim, method=method)
    NeuralODE_model.load_state_dict(torch.load(ode_pth_path))

    controller = PushingController(env, NeuralODE_model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=30)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 100

    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            break
    
    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')
            
            
    # Evaluate state
    # plt.close(fig)
    Image(filename=visualizer.get_gif(given_name='obstacle_avoidance_pushing_visualization_ode_single_step.gif'))


def obstacle_avoidance_pushing_ode_multi_step(method, path):
    # GIF Visualizer
    visualizer = GIFVisualizer()

    # Path
    ckpt_path = os.path.join(path,'/ckpt/Panda_pushing/continuous')

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=2, path = path)

    # Load the pushing dynamics model
    # ode_pth_path = os.path.join(path, 'ODEFunc_multi_step.pt')
    ode_pth_path = os.path.join(ckpt_path, 'ODEFunc_multi_step_{}.pt'.format(method if method else "dopri5"))
    state_dim = 3
    action_dim = 3
    NeuralODE_model = NeuralODE(state_dim=state_dim, action_dim=action_dim, method=method)
    NeuralODE_model.load_state_dict(torch.load(ode_pth_path))

    controller = PushingController(env, NeuralODE_model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=30)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 100

    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            break
    
    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')
            
            
    # Evaluate state
    # plt.close(fig)
    Image(filename=visualizer.get_gif(given_name='obstacle_avoidance_pushing_visualization_ode_multi_step.gif'))


# if __name__ == "__main__":
#     # single step ode model
#     # obstacle_avoidance_pushing_ode_single_step()
#     obstacle_avoidance_pushing_ode_single_step(method = 'rk4')

    # multi step ode model
    # obstacle_avoidance_pushing_ode_multi_step()