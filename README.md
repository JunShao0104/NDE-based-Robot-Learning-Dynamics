# NDE-based-Robot-Learning-Dynamics
Numerical Differential Equations based Learning Dynamics for Robotic Systems
![image](https://github.com/JunShao0104/NDE-based-Robot-Learning-Dynamics/blob/main/fig/obstacle_avoidance_pushing_visualization.gif)

## 0 Installation
### A Prerequisites
Install Python 3, pybullet, numpy, pytorch, and matplotlib.

### B Repository Clone and Packages Installation
```bash
# Clone Repo
git clone https://github.com/JunShao0104/NDE-based-Robot-Learning-Dynamics.git

# Install Packages
bash install.sh
./install.sh
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
Firstly, collect data we need for training.
```bash
# Change the data_path in the main function in collect_data.py
python collect_data.py
```
 Then, run the single step training and multi step training.
 ```bash
# Before training, change the ckpt_path and data_path
python single_step_training.py
python multi_step_training.py
 ```

## 2 Panda Robot Planning and Control
### 2.1 Obstacle Free Pushing
```bash
# Change the ckpt_path and the dynamics model you need.
python obstacle_free_pushing.py
```

### 2.2 Obstacle Avoidance Pushing
```bash
# Change the ckpt_path and the dynamics model you need.
python obstacle_avoidance_pushing.py
```

## 3 Robot Learning Dynamics for Other Open-Source Datasets
### 3.1 Forward Kinematics Learning


### 3.2 Inverse Kinematics Learning
