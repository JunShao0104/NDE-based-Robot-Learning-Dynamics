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
from src.model.absolute_dynamics_model import AbsoluteDynamicsModel
from src.model.residual_dynamics_model import ResidualDynamicsModel
from src.model.polynet_dynamics_model import Poly_2_DynamicsModel
from src.model.polynet_dynamics_model import mPoly_2_DynamicsModel
from src.model.polynet_dynamics_model import way_2_DynamicsModel
from src.model.fractalnet_dynamics_model import RKNN_2_DynamicsModel
from src.model.neural_ode_model import NeuralODE

# Cost function and pushing controller
from src.controller.pushing_controller import PushingController
from src.controller.pushing_cost import collision_detection
from src.controller.pushing_cost import free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function
from src.env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE

def obstacle_free_pushing_ode(method, path):
    # Control on an obstacle free environment

    # Notebook Visualizer
    # fig = plt.figure(figsize=(8,8))
    # hfig = display(fig, display_id=True)
    # visualizer = NotebookVisualizer(fig=fig, hfig=hfig)

    # GIF Visualizer
    visualizer = GIFVisualizer()

    # Path
    ckpt_path = (path + '/ckpt/Panda_pushing/continuous')

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  camera_heigh=800, camera_width=800, render_every_n_steps=5, path = path)

    # Load the pushing dynamics model
    ode_pth_path = os.path.join(ckpt_path, 'ODEFunc_single_step_{}.pt'.format(method if method else "dopri5"))
    # proj_pth_path = os.path.join(ckpt_path, 'ProjNN_single_step.pt')
    state_dim = 3
    action_dim = 3
    #NeuralODE_model = NeuralODE(ode_pth_path=ode_pth_path, proj_pth_path=None, state_dim=state_dim, action_dim=action_dim)
    NeuralODE_model = NeuralODE(state_dim=state_dim, action_dim=action_dim, method= method)
    NeuralODE_model.load_state_dict(torch.load(ode_pth_path))

    controller = PushingController(env, NeuralODE_model,
                                free_pushing_cost_function, num_samples=100, horizon=20)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 20

    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            break

            
    # Evaluate if goal is reached
    end_state = env.get_state()
    # print("end_state: ", end_state)
    target_state = TARGET_POSE_OBSTACLES
    # print("target_state: ", target_state)
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')
            
            
    # Evaluate state
    # plt.close(fig)
    Image(filename=visualizer.get_gif(given_name='obstacle_free_pushing_visualization_ode.gif'))
