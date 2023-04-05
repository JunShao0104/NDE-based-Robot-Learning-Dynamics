import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split


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