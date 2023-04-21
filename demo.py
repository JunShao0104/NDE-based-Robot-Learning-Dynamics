import yaml
import os
import numpy as np
import argparse

from PIL import Image, ImageSequence
import cv2
import imageio 
# from src.single_step_training import train as single_step_train
# from src.single_step_training_FK import train as single_step_train_FK
# from src.multi_step_training import train as multi_step_train
# from src.neural_ode_learning import train_singlestep as ode_single_train
# from src.neural_ode_learning import train_multistep as ode_multi_train

from src.utils.visualizers import GIFVisualizer, NotebookVisualizer
from PIL import Image
from src.obstacle_avoidance_pushing import obstacle_avoidance_pushing as obstacle_avoid_push
from src.obstacle_avoidance_pushing_ode import obstacle_avoidance_pushing_ode_single_step as obstacle_avoid_single_ode
from src.obstacle_avoidance_pushing_ode import obstacle_avoidance_pushing_ode_multi_step as obstacle_avoid_multi_ode
from src.obstacle_free_pushing_ode import obstacle_free_pushing_ode as obstacle_free_ode
from src.obstacle_free_pushing import obstacle_free_pushing as obstacle_free_push
import torch
import os



if __name__ == "__main__":
    path = os.path.abspath(os.getcwd())
    # print(path)
    parser = argparse.ArgumentParser(description="NDE_Based_Robot_Learning_Dynamics Demo")
    parser.add_argument("--model", help="Change if other model if preferred. Default: mpoly_2", default='mpoly_2')
    parser.add_argument("--ode_method", help="Change if other method if preferred. Default: rk4", default='rk4')
    # parser.add_argument("--config", help="YAML config file", default="models.yaml")
    args = parser.parse_args()
    
    # if args.path:
    #     path = args.path
    # else: 
    #     path = configs['base_dir']

    model_demo_1 = obstacle_avoid_push(args.model, path)
    model_demo_2 = obstacle_avoid_single_ode(args.ode_method, path)
    # model_demo_1, model_demo_2 = False, False
    if(model_demo_1):
        demo1_file = path + "/demo/obstacle_avoidance_pushing_visualization_success.gif"
    else:
        demo1_file = path + "/demo/obstacle_avoidance_pushing_visualization.gif"
    if(model_demo_2):
        demo2_file = path + "/demo/obstacle_avoidance_pushing_visualization_ode_single_success.gif"
    else:
        demo2_file = path + "/demo/obstacle_avoidance_pushing_visualization_ode_single_step_3.gif"
    
    #Create reader object for the gif
    gif1 = imageio.get_reader(demo1_file)
    gif2 = imageio.get_reader(demo2_file)

    #If they don't have the same number of frame take the shorter
    number_of_frames = max(gif1.get_length(), gif2.get_length()) 

    #Create writer object
    output_path = path + '/demo/output.gif'
    new_gif = imageio.get_writer(output_path)

    for frame_number in range(number_of_frames):
        # print(gif1.get_next_data())
        if(frame_number < gif1.get_length()):
            img1 = gif1.get_next_data()
        else:
            img1 = img1
        img2 = gif2.get_next_data()
        #here is the magic
        new_image = np.hstack((img1, img2))
        new_gif.append_data(new_image)

    gif1.close()
    gif2.close()    
    new_gif.close()
    
    pic_name = path + '/demo/output.gif'
    im = Image.open(pic_name)
    
    for frame in ImageSequence.Iterator(im):
        frame = frame.convert('RGB')
        cv2_frame = np.array(frame)
        show_frame = cv2.cvtColor(cv2_frame, cv2.COLOR_RGB2BGR)
        cv2.imshow(pic_name, show_frame)
        cv2.waitKey(50)



    