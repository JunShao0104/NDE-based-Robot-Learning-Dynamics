import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

# NeuralODE
from torchdiffeq import odeint_adjoint as odeint


class SE2PoseLoss(nn.Module):
    """
    Compute the SE2 pose loss based on the object dimensions (block_width, block_length).
    Need to take into consideration the different dimensions of pose and orientation to aggregate them.

    Given a SE(2) pose [x, y, theta], the pose loss can be computed as:
        se2_pose_loss = MSE(x_hat, x) + MSE(y_hat, y) + rg * MSE(theta_hat, theta)
    where rg is the radious of gyration of the object.
    For a planar rectangular object of width w and length l, the radius of gyration is defined as:
        rg = ((l^2 + w^2)/12)^{1/2}

    """

    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length

    def forward(self, pose_pred, pose_target):
        se2_pose_loss = None
        # --- Your code here
        rg = ((self.w**2 + self.l**2)/12)**0.5
        x_loss = F.mse_loss(pose_pred[:, 0], pose_target[:, 0])
        y_loss = F.mse_loss(pose_pred[:, 1], pose_target[:, 1])
        theta_loss = rg * F.mse_loss(pose_pred[:, 2], pose_target[:, 2])
        se2_pose_loss = x_loss + y_loss + theta_loss

        # ---
        return se2_pose_loss


class SingleStepLoss(nn.Module):

    def __init__(self, loss_fn):
        super().__init__()
        self.loss = loss_fn

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        pred_state = model(state, action)
        single_step_loss = self.loss(pred_state, target_state)

        # ---
        return single_step_loss


class SingleStepLoss_ode(nn.Module):

    def __init__(self, loss_fn, method = None):
        super().__init__()
        self.loss = loss_fn
        self.method = method

    def forward(self, model, state, action, target_state):
        """
        Compute the single step loss resultant of querying model with (state, action) and comparing the predictions with target_state.
        """
        single_step_loss = None
        # --- Your code here
        T = 2 # For single step
        state_action = torch.cat((state, action), dim=1)
        batch_y0 = state_action # (M, D)
        batch_y = target_state # (M, S)
        batch_t = torch.arange(T).float()
        if(self.method):
            pred_y = odeint(model.odefunc, batch_y0, batch_t,method=self.method, options=dict(step_size=0.5))
        else:
            pred_y = odeint(model.odefunc, batch_y0, batch_t) # (T, M, D)
        T, M, D = pred_y.shape
        single_step_loss = self.loss(pred_y[-1, :, :state.shape[1]], batch_y)

        # ---
        return single_step_loss


class MultiStepLoss(nn.Module):

    def __init__(self, loss_fn, discount=0.99):
        super().__init__()
        self.loss = loss_fn
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        # state: (batchsize, state_size)
        # actions: (batchsize, num_steps, action_size)
        # target_states: (batchsize, num_steps, state_size)
        multi_step_loss = 0.0
        num_steps = int(actions.shape[1])
        current_state = state
        discount = 1.0
        for i in range(num_steps):
          pred_state = model(current_state, actions[:, i, :])
          single_step_loss = discount * self.loss(pred_state, target_states[:, i, :])
          multi_step_loss += single_step_loss
          current_state = pred_state
          discount *= self.discount

        # ---
        return multi_step_loss


class MultiStepLoss_ode(nn.Module):

    def __init__(self, loss_fn, method = None):
        super().__init__()
        self.loss = loss_fn
        self.method = method

    def forward(self, model, state, actions, target_states):
        """
        Compute the multi-step loss resultant of multi-querying the model from (state, action) and comparing the predictions with targets.
        """
        multi_step_loss = None
        # --- Your code here
        # state: (batchsize, state_size)
        # actions: (batchsize, num_steps, action_size)
        # target_states: (batchsize, num_steps, state_size)
        multi_step_loss = 0.0
        num_steps = int(actions.shape[1])
        current_state = state
        for step in range(num_steps):
            T = 2
            state_action = torch.cat((current_state, actions[:, step, :]), dim=1)
            batch_y0 = state_action # (M, D)
            batch_y = target_states[:, step, :] # (M, S)
            batch_t = torch.arange(T).float()
            if(self.method):
                pred_y = odeint(model.odefunc, batch_y0, batch_t,method=self.method, options=dict(step_size=0.5))
            else:
                pred_y = odeint(model.odefunc, batch_y0, batch_t)
            T, M, D = pred_y.shape
            multi_step_loss += self.loss(pred_y[-1, :, :state.shape[1]], batch_y)
            # update the state
            current_state = pred_y[-1, :, :state.shape[1]]

        # ---
        return multi_step_loss