import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import distributions as torch_dist


class Actor(nn.Module):
    def __init__(self, num_actions, h_dim, device, discrete=False, std_cond_on_input=False, embed_action = None):
        super().__init__()
        self.mu = embed_action #nn.Linear(h_dim, num_actions, device=device)
        self.std_linear_layer = std_cond_on_input
        self.log_std = nn.Parameter(
            torch.zeros(num_actions, dtype=torch.float32, device=device)) if not std_cond_on_input \
            else (nn.Linear(h_dim, num_actions, device=device))
        self.log_std_min = -20
        self.log_std_max = 2
        self.categorical_dist = discrete

    def forward(self, x):
        #w = self.mu.weight
        #action_mean = torch.matmul(x,w).clamp(-1,1)+
        action_mean = torch.tanh(self.mu(x))
        action_std = torch.exp(self.log_std.clamp(self.log_std_min, self.log_std_max)) if not self.std_linear_layer \
            else torch.exp(self.log_std(x).clamp(self.log_std_min, self.log_std_max))
        return torch_dist.Normal(action_mean,action_std)

    def forward_cat(self, x):
        action_mean = torch.softmax(self.mu(x), dim=-1)
        return torch_dist.Categorical(action_mean)
