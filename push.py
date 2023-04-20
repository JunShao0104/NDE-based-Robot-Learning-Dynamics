import yaml
import os
import numpy as np
import argparse
# from src.single_step_training import train as single_step_train
# from src.single_step_training_FK import train as single_step_train_FK
# from src.multi_step_training import train as multi_step_train
# from src.neural_ode_learning import train_singlestep as ode_single_train
# from src.neural_ode_learning import train_multistep as ode_multi_train

from src.obstacle_avoidance_pushing import obstacle_avoidance_pushing as obstacle_avoid_push
from src.obstacle_avoidance_pushing_ode import obstacle_avoidance_pushing_ode_single_step as obstacle_avoid_single_ode
from src.obstacle_avoidance_pushing_ode import obstacle_avoidance_pushing_ode_multi_step as obstacle_avoid_multi_ode
from src.obstacle_free_pushing_ode import obstacle_free_pushing_ode as obstacle_free_ode
from src.obstacle_free_pushing import obstacle_free_pushing as obstacle_free_push
import torch
# from src.model.absolute_dynamics_model import AbsoluteDynamicsModel
# from src.model.residual_dynamics_model import ResidualDynamicsModel
# from src.model.polynet_dynamics_model import Poly_2_DynamicsModel
# from src.model.polynet_dynamics_model import mPoly_2_DynamicsModel
# from src.model.polynet_dynamics_model import way_2_DynamicsModel
# from src.model.fractalnet_dynamics_model import RKNN_2_DynamicsModel
# from src.env.panda_pushing_env import PandaPushingEnv


    

def push(configs):
    model = configs['model']
    path = configs['base_dir']
    method= configs['ODE_solver']
    # obstacle push
    if(configs['obstacle']):
        # neural ode
        if(model=='ode'):
            # single step
            if(configs['step'] == 'single'):
                print("Pushing with obstacle, ODE with method: ", method," , single step")
                obstacle_avoid_single_ode(method= method, path=path)
            # multi step
            elif(configs['step'] == 'multi'):
                print("Pushing with obstacle, ODE with method: ", method," , multi step")
                obstacle_avoid_multi_ode(method= method, path = path)
        else:
            # other models
            print("Pushing with obstacle, model: ", model)
            obstacle_avoid_push(model = model, path = path)

    #obstacle free push
    else:
        # neural ode
        if(model=='ode'):
            # single step
            if(configs['step'] == 'single'):
                print("Pushing with obstacle, ODE with method: ", method," , single step")
                obstacle_free_ode(method= method, path=path)
            # multi step
            elif(configs['step'] == 'multi'):
                #**************************NEED IMPLEMENT***********************************
                print("NOT YET IMPLEMENTED")
                # print("Pushing with obstacle, ODE with method: ", method," , multi step")
                # obstacle_avoid_multi_ode(method= method, path = path)
        else:
            # other models
            print("Pushing with obstacle, model: ", model)
            obstacle_free_push(model = model, path = path)

# def train_model()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDE_Based_Robot_Learning_Dynamics fusion model")
    parser.add_argument("--config", help="YAML config file", default="models.yaml")
    args = parser.parse_args()

    # Add the yaml to the config args parse
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # Merge configs and args
    for arg in vars(args):
        configs[arg] = getattr(args, arg)
    model = configs['model']
    path = configs['base_dir']
    obstacle : configs['obstacle']
    step : configs['step']     #single or multi
    dataset: configs['dataset']      #panda, fk or ik
    method: configs['ODE_solver']
    push(configs)

            


