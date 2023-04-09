# NDE-based-Robot-Learning-Dynamics
Numerical Differential Equations based Learning Dynamics for Robotic Systems
![image](https://github.com/JunShao0104/NDE-based-Robot-Learning-Dynamics/blob/main/fig/obstacle_avoidance_pushing_visualization.gif)

## 0 Installation
### A Prerequisites
Install Python 3, pybullet, numpy, pytorch (version >=1.13.0), and matplotlib.

### B Repository Clone and Packages Installation
```bash
# Clone Repo
git clone https://github.com/JunShao0104/NDE-based-Robot-Learning-Dynamics.git

# Install Packages
bash install.sh
# or
# ./install.sh
```

## 1 Robot Learning Dynamics for Panda Planar Pushing
### 1.1 Discrete ODE Learning V.S. Continuous ODE Learning
Compare the prediction performance of the discrete ODE learning methods and the continuous ODE learning method.

Discrete ODE Learning methods:
- Absolute Dynamics Model (Baseline)
- ResNet Dynamics Model (Forward Euler)
- PolyNet Dynamics Model (Backward Euler)
- FractalNet Dynamics Model (Runge-Kutta)

Continuous ODE Learning method:
- Neural ODE    

### 1.2 Single Step Learning V.S. Multi Step Learning
Learn the robot dynamics in single step manner and multi step manner, and then compare their performance.
Firstly, collect data we need for training. We provide collected data and validation data under the /data folder.
```bash
# Change the data_path in the main function in collect_data.py
python src/collect_data.py
```
 Then, run the single step training and multi step training. We also provide the weight we trained under the folder /ckpt.
 ```bash
# Before training, change the ckpt_path and data_path
python src/single_step_training.py
python src/multi_step_training.py
 ```
 
### 1.3 Neural ODE based Learning Dynamics
Need a discussion!! Not sure about whether this is correct...

We refer to torchdiffeq (https://github.com/rtqichen/torchdiffeq) for implementing the Neural ODE based Learning Dynamics. Since the odeint(_adjoint) method require the input and output dimension are the same and only take in two arguments (a time variable and a state variable), we have to concatenate the state x and the action u together and learn the IVP (Initial Value Problem) with the differential equation: concat(x(t+1), u(t+1))= concat(x(t), u(t)) + f'(x(t), u(t), t).

Then, we use a MLP to project the prediction concat(x(t+1), u(t+1))' back to the state x(t+1)'.
```bash
# Before training, change the ckpt_path and data_path
python src/neural_ode_learning.py
```


## 2 Panda Robot Planning and Control
### 2.1 Obstacle Free Pushing
```bash
# Change the ckpt_path and the dynamics model you need.
python src/obstacle_free_pushing.py
```

### 2.2 Obstacle Avoidance Pushing
```bash
# Change the ckpt_path and the dynamics model you need.
python src/obstacle_avoidance_pushing.py
```

## 3 Robot Learning Dynamics for Other Open-Source Datasets
### 3.1 Forward Kinematics Learning


### 3.2 Inverse Kinematics Learning
