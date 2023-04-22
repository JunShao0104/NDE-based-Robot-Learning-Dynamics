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
```

## 1 Robot Learning Dynamics for Panda Planar Pushing

### Quick Start
```bash
# Train models with setup designed in models.yaml
python train.py
# Pushing with trained weights from ckpts folder
python push.py
```
### Config
- model: `absolute`; `mpoly_2`; `poly_2`; `residual`; `RKNN`; `way_2`; `ode`.
- ODE_solver: `False`, rk4; `False`, dopri5(default).
- obstacle : `True`, with obstacle; `False`, without obstacle.
- base_dir : `.../NDE-based-Robot-Learning-Dynamics`, change before running.
- step : `single`, single step; `multi`, multi-step.
- dataset: `panda`, Panda robot pushing dataset; `baxterfk`, Baxter forward dynamics dataset; `baxterik`, Baxter inverse dynamics dataset.

### 1.1 Discrete ODE Learning V.S. Continuous ODE Learning
Compare the prediction performance of the discrete ODE learning methods and the continuous ODE learning methods.

Discrete ODE Learning methods:
- Absolute Dynamics Model (Baseline)
- ResNet Dynamics Model (Forward Euler)
- PolyNet Dynamics Model (Backward Euler)
- FractalNet Dynamics Model (Runge-Kutta)

Continuous ODE Learning method:
- Neural ODE (`dopri5` integration algorithm)
- Neural ODE (`rk4` integration algorithm) (different step size: `0.5`, `0.25`, `0.1`)

### 1.2 Single Step Training V.S. Multi Step Training
Learn the robot dynamics in single step manner and multi step manner, and then compare their performance.
Collect the data for training. We provide collected data and validation data under the `/data` folder.
```bash
python src/collect_data.py
```
To run the training. We provide the weights we trained under th3 `/ckpt` folder.
 ```bash
python src/single_step_training.py # Training discrete dynamics models in the single step pipeline
python src/multi_step_training.py # Training discrete dynamics models in the multi step pipeline
python src/neural_ode_learning.py # Training Neural ODE models in both single step and multi step pipelines
 ```
 To run the validation.
 ```bash
 python src/validation_error.py # Error validation for dynamics models on Panda pushing dataset
 ```


## 2 Panda Robot Planning and Control
### 2.1 Obstacle Free Pushing
```bash
python src/obstacle_free_pushing.py # Obstacle free pushing using discrete models
python src/obstacle_free_pushing_ode.py # Obstacle free pushing using Neural ODE models
```

### 2.2 Obstacle Avoidance Pushing
```bash
python src/obstacle_avoidance_pushing.py # Obstacle avoidance pushing using discrete models
python src/obstacle_avoidance_pushing_ode.py # Obstacle avoidance pushing using Neural ODE models
```

## 3 Robot Learning Dynamics for Other Open-Source Datasets
### 3.1 Forward Kinematics Learning
We run training and testing on the Forward Dynamics Dataset Using KUKA LWR and Baxter.

To run the training.
```bash
python src/single_step_training_FK.py # Training both discrete dynamics models and Neural ODE models in the single step pipeline
python src/multi_step_training_FK.py # Training both discrete dynamics models and Neural ODE models in the multi step pipeline
```
To run the validation.
```bash
python src/validation_error.py # Error validation for dynamics models on Panda pushing dataset
```

### 3.2 Inverse Kinematics Learning
To be continued...


## 4 References
```bash
@article{chen2018neuralode,
  title={Neural Ordinary Differential Equations},
  author={Chen, Ricky T. Q. and Rubanova, Yulia and Bettencourt, Jesse and Duvenaud, David},
  journal={Advances in Neural Information Processing Systems},
  year={2018}
}
```
```bash
@inproceedings{polydoros2016reservoir,
  title={A reservoir computing approach for learning forward dynamics of industrial manipulators},
  author={Polydoros, Athanasios S and Nalpantidis, Lazaros},
  booktitle={2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={612--618},
  year={2016},
  organization={IEEE}
}
```
