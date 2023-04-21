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
from PIL import Image

# Model
from src.model.absolute_dynamics_model import AbsoluteDynamicsModel
from src.model.residual_dynamics_model import ResidualDynamicsModel
from src.model.polynet_dynamics_model import Poly_2_DynamicsModel
from src.model.polynet_dynamics_model import mPoly_2_DynamicsModel
from src.model.polynet_dynamics_model import way_2_DynamicsModel
from src.model.fractalnet_dynamics_model import RKNN_2_DynamicsModel

# Cost function and pushing controller
from src.controller.pushing_controller import PushingController
from src.controller.pushing_cost import collision_detection
from src.controller.pushing_cost import free_pushing_cost_function, collision_detection, obstacle_avoidance_pushing_cost_function
from src.env.panda_pushing_env import TARGET_POSE_FREE, TARGET_POSE_OBSTACLES, BOX_SIZE


def obstacle_avoidance_pushing(model, path):
    # Control on an obstacle free environment
    ckpt_path = (path + '/ckpt/Panda_pushing/discrete/')

    # Notebook Visualizer
    # fig = plt.figure(figsize=(8,8))
    # hfig = display(fig, display_id=True)
    # visualizer = NotebookVisualizer(fig=fig, hfig=hfig)
    if model == 'absolute':
        pushing_model = AbsoluteDynamicsModel(3, 3)
    elif model == 'RKNN':
        pushing_model = RKNN_2_DynamicsModel(3, 3)
    elif model == 'poly_2':
        pushing_model = Poly_2_DynamicsModel(3, 3)
    elif model == 'residual':
        pushing_model = ResidualDynamicsModel(3, 3)
    elif model == 'mpoly_2':
        pushing_model = mPoly_2_DynamicsModel(3, 3)
    elif model == 'way_2':
        pushing_model = way_2_DynamicsModel(3, 3)
    else:
        print("No model name: ", model, " found, please check the list again. ")

    print("Currently testing model: ", model)
    # GIF Visualizer

    # Load the pushing dynamics model
    model_path = os.path.join(ckpt_path, 'pushing_{}_dynamics_model.pt'.format(model))
    for tries in range(10):
        visualizer = GIFVisualizer()
        # print(tries)
        # set up controller and environment
        env = PandaPushingEnv(visualizer=visualizer, render_non_push_motions=False,  include_obstacle=True, camera_heigh=800, camera_width=800, render_every_n_steps=5, path = path)
        pushing_model.load_state_dict(torch.load(model_path))

        controller = PushingController(env, pushing_model,
                                    obstacle_avoidance_pushing_cost_function, num_samples=1000, horizon=20)
        env.reset()

        state_0 = env.reset()
        state = state_0

        num_steps_max = 20

        for i in range(num_steps_max):
            action = controller.control(state)
            try:
                state, reward, done, _ = env.step(action)
                if done:
                    break
            except AttributeError:
                print('Action out of limit, retry...')
        # Evaluate if goal is reached
        end_state = env.get_state()
        target_state = TARGET_POSE_OBSTACLES
        goal_distance = np.linalg.norm(end_state[:2]-target_state[:2]) # evaluate only position, not orientation
        goal_reached = goal_distance < BOX_SIZE
        if goal_reached:
            break
    
    print(f'GOAL REACHED: {goal_reached}')
    if goal_reached:
        gif = visualizer.get_gif(given_name='obstacle_avoidance_pushing_visualization_success.gif', path='demo/')
        # imageObject = Image.open(gif)
    return goal_reached
            
    # Evaluate state
    # plt.close(fig)


# if __name__ == "__main__":
#     obstacle_avoidance_pushing()