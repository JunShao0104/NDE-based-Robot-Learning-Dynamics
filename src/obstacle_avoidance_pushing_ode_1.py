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
from env.panda_pushing_env import PandaPushingEnv
from utils.visualizers import GIFVisualizer, NotebookVisualizer

# Model
from model.absolute_dynamics_model import AbsoluteDynamicsModel
from model.residual_dynamics_model import ResidualDynamicsModel
from model.polynet_dynamics_model import Poly_2_DynamicsModel
from model.polynet_dynamics_model import mPoly_2_DynamicsModel
from model.polynet_dynamics_model import way_2_DynamicsModel
from model.fractalnet_dynamics_model import RKNN_2_DynamicsModel
from model.neural_ode_model_1 import NeuralODE

# Cost function and pushing controller
from controller.pushing_controller import PushingController
from controller.pushing_cost import collision_detection
from controller.pushing_cost import free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function
from env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE

# pth path
ckpt_path = '/home/lidonghao/rob498proj/NDE-based-Robot-Learning-Dynamics/ckpt'

def obstacle_avoidance_pushing_ode():
    # Control on an obstacle free environment

    # Notebook Visualizer
    # fig = plt.figure(figsize=(8,8))
    # hfig = display(fig, display_id=True)
    # visualizer = NotebookVisualizer(fig=fig, hfig=hfig)
    if(torch.cuda.is_available()):
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")
    # GIF Visualizer
    visualizer = GIFVisualizer()

    # set up controller and environment
    env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5)

    # Load the pushing dynamics model
    NeuralODE_model = NeuralODE(3,3,DEVICE).to(DEVICE)
    ode_pth_path = os.path.join(ckpt_path, 'ODEFunc_w_proj.pt')
    NeuralODE_model.load_state_dict(torch.load(ode_pth_path))

    controller = PushingController(env, NeuralODE_model,
                                obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20, device=DEVICE)
    env.reset()

    state_0 = env.reset()
    state = state_0

    num_steps_max = 30
    # print(state.shape)
    cur_step = 0
    for i in range(num_steps_max):
        action = controller.control(state)
        state, reward, done, _ = env.step(action)
        if done:
            cur_step = i
            break
    print("total num of steps: ", cur_step)


    # Evaluate if goal is reached
    end_state = env.get_state()
    target_state = TARGET_POSE_OBSTACLES
    goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
    goal_reached = goal_distance < BOX_SIZE

    print(f'GOAL REACHED: {goal_reached}')


    # Evaluate state
    # plt.close(fig)
    name = 'ode_obs_avoid_visual_1.gif'
    Image(filename=visualizer.get_gif(given_name=name))
    print("Gif has been writted to: ", name)

if __name__ == "__main__":
    obstacle_avoidance_pushing_ode()