import yaml
import os
import numpy as np
import argparse

from src.single_step_training import train as single_step_train
from src.single_step_training_FK import discrete_train as single_step_train_FK
from src.single_step_training_FK import continuous_train as single_step_train_ode_FK
from src.multi_step_training import train as multi_step_train
from src.neural_ode_learning import train_singlestep as ode_single_train
from src.neural_ode_learning import train_multistep as ode_multi_train
from src.multi_step_training_FK import discrete_train as multi_step_train_FK
from src.multi_step_training_FK import continuous_train as multi_step_train_ode_FK

from src.obstacle_avoidance_pushing import obstacle_avoidance_pushing as obstacle_avoid_push
from src.obstacle_avoidance_pushing_ode import obstacle_avoidance_pushing_ode_single_step as obstacle_avoid_single_ode
from src.obstacle_avoidance_pushing_ode import obstacle_avoidance_pushing_ode_multi_step as obstacle_avoid_multi_ode
import torch
# from src.model.absolute_dynamics_model import AbsoluteDynamicsModel
# from src.model.residual_dynamics_model import ResidualDynamicsModel
# from src.model.polynet_dynamics_model import Poly_2_DynamicsModel
# from src.model.polynet_dynamics_model import mPoly_2_DynamicsModel
# from src.model.polynet_dynamics_model import way_2_DynamicsModel
# from src.model.fractalnet_dynamics_model import RKNN_2_DynamicsModel
# from src.env.panda_pushing_env import PandaPushingEnv

def train(configs):
    # Initialize the trainer
    method = configs['ODE_solver']
    model = configs['model']
    path = configs['base_dir']
    # Single step training
    if(configs['step'] == 'single'):
        # Using Panda dataset from HW
        if(configs['dataset'] == 'panda'):
            dataset = np.load(os.path.join(path, 'data/Panda_pushing/collected_data.npy'), allow_pickle=True)
            # Neural ODE with method
            if(model == 'ode'):
                print("Training ODE, ",method, ", Single step, panda dataset")
                ode_single_train(method = method, dataset = dataset, path = path)
            # Other models
            else:
                print("Training ",model, ", Single step, panda dataset")
                single_step_train(model = model, dataset = dataset, path=path)
        
        # Using Baxter Forward Kinamatics Dataset
        elif(configs['dataset'] == 'baxterfk'):
            dataset = torch.from_numpy(np.load(os.path.join(path, 'data/FK/BaxterDirectDynamics.npy'), allow_pickle=True))
            # Neural ODE with method
            if(model== 'ode'):
                print("Training ODE, ",method, ", Single step, baxter FK dataset")
                single_step_train_ode_FK(method = method, dataset = dataset, path = path)
            # Other models
            else:
                print("Training ",model, ", Single step, baxter FK dataset")
                single_step_train_FK(model = model, dataset = dataset, path = path)
        
        # Using Baxter Inverse Kinamatics Dataset
        elif(configs['dataset'] == 'baxterik'):
            dataset = torch.from_numpy(np.load(os.path.join(path, 'data/IK/BaxterRand.npy'), allow_pickle=True))
            #**************************NEED IMPLEMENT***********************************
            print("NOT YET IMPLEMENTED")
            # if(model== 'ode'):
            #     print("Need ODE single train for IK")
            #     # ode_single_train(method=method, dataset = dataset, path = path)
            # else:
            #   single_step_train_IK(model = model, dataset = dataset)

        # Invalid dataset input
        else:
            print("Wrong dataset input, check yaml again")
    
    # Multi step training
    elif(configs['step'] == 'multi'):
        num_steps = configs['num_steps']
        # Using Panda dataset from HW
        if(configs['dataset'] == 'panda'):
            dataset = np.load(os.path.join(path, 'data/Panda_pushing/collected_data.npy'), allow_pickle=True)
            # Neural ODE with method
            if(model== 'ode'):
                print("Training ODE, ",method, ", Multi step, panda dataset")
                ode_multi_train(method=method, dataset = dataset, path = path, num_steps = num_steps)
            # Other models
            else:
                print("Training ",model, ", Multi step with step: ", num_steps,", panda dataset")
                multi_step_train(model = model, dataset = dataset, path = path, num_steps = num_steps)

        # Using Baxter Forward Kinamatics Dataset
        elif(configs['dataset'] == 'baxterfk'):
            dataset = torch.from_numpy(np.load(os.path.join(path, 'data/FK/BaxterDirectDynamics.npy'), allow_pickle=True))
            # Neural ODE with method
            if(model== 'ode'):
                print("Training ODE, ",method, ", Multi step, BaxterFK dataset")
                multi_step_train_ode_FK(method=method, dataset = dataset, path = path, num_steps = num_steps)
            # Other models
            else:
                print("Training ",model, ", Multi step with step: ", num_steps,", BaxterFK dataset")
                multi_step_train_FK(model = model, dataset = dataset, path = path, num_steps = num_steps)

        # Using Baxter Inverse Kinamatics Dataset
        elif(configs['dataset'] == 'baxterik'):
             #**************************NEED IMPLEMENT***********************************
            dataset = torch.from_numpy(np.load(os.path.join(path, 'data/IK/BaxterRand.npy'), allow_pickle=True))
            print("Training ",model, ", Multi step, baxter IK dataset")
            print("NOT YET IMPLEMENTED")
            # multi_step_train_IK(model = model, dataset = dataset)
        
        # Invalid dataset input
        else:
            print("Wrong dataset input, check yaml again")
    
    # Invalid step input
    else:
        print("Wrong step input, check yaml again")

# def train_model()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NDE_Based_Robot_Learning_Dynamics")
    parser.add_argument("--config", help="YAML config file", default="models.yaml")
    args = parser.parse_args()

    # Add the yaml to the config args parse
    with open(args.config) as f:
        configs = yaml.safe_load(f)

    # Merge configs and args
    for arg in vars(args):
        configs[arg] = getattr(args, arg)
    # model = configs['model']
    # path = configs['base_dir']
    # obstacle : configs['obstacle']
    # step : configs['step']     #single or multi
    # dataset: configs['dataset']      #panda, fk or ik
    # method: configs['ODE_solver']
    train(configs)

            


