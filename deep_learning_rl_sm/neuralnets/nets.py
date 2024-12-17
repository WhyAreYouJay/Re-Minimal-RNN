import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import distributions as torch_dist


class Actor(nn.Module):
    def __init__(self, num_actions, h_dim, device, discrete=False):
        super().__init__()
        self.mu = nn.Linear(h_dim, num_actions, device=device)
        # (FROM PAPER) self.log_std = nn.Parameter(torch.zeros(act_dim, dtype=torch.float32, device = device))
        self.log_std = nn.Linear(h_dim, num_actions, device=device)
        self.log_std_min = -20
        self.log_std_max = 2
        self.categorical_dist = discrete

    @torch.compile()
    def forward(self, x):
        action_mean = torch.tanh(self.mu(x))
        action_std = torch.exp(self.log_std(x).clamp(self.log_std_min, self.log_std_max))
        return torch_dist.Normal(action_mean,
                                 action_std)
        # return torch_dist.Normal(action_mean, action_std)

    def forward_cat(self, x):
        action_mean = torch.softmax(self.mu(x), dim=-1)
        action_std = None
        return torch_dist.Categorical(action_mean)


# TODO delete below when finished testing


class MaskedCausalAttention(nn.Module):
    def __init__(
            self,
            h_dim,
            max_T,
            n_heads,
            drop_p,
            num_inputs,
            mgdt=False,
            dt_mask=False,
            att_mask=None,
    ):
        super().__init__()

        self.n_heads = n_heads
        self.max_T = max_T
        self.num_inputs = num_inputs

        self.q_net = nn.Linear(h_dim, h_dim)
        self.k_net = nn.Linear(h_dim, h_dim)
        self.v_net = nn.Linear(h_dim, h_dim)

        self.proj_net = nn.Linear(h_dim, h_dim)

        self.att_drop = nn.Dropout(drop_p)
        self.proj_drop = nn.Dropout(drop_p)

        if att_mask is not None:
            mask = att_mask
        else:
            ones = torch.ones((max_T, max_T))
            mask = torch.tril(ones).view(1, 1, max_T, max_T)
            if (mgdt and not dt_mask):
                # need to mask the return except for the first return entry
                # this is the default practice used by their notebook
                # for every inference, we first estimate the return value for the first return
                # then we estimate the action for at timestamp t
                # it is actually not mentioned in the paper. (ref: ret_sample_fn, single_return_token)
                # mask other ret entries (s, R, a, s, R, a)
                period = num_inputs
                ret_order = 2
                ret_masked_rows = torch.arange(
                    period + ret_order - 1, max_T, period
                ).long()
                # print(ret_masked_rows)
                # print(max_T, ret_masked_rows, mask.shape)
                mask[:, :, :, ret_masked_rows] = 0

        # register buffer makes sure mask does not get updated
        # during backpropagation
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape  # batch size, seq length, h_dim * n_heads

        N, D = (
            self.n_heads,
            C // self.n_heads,
        )  # N = num heads, D = attention dim

        # rearrange q, k, v as (B, N, T, D)
        q = self.q_net(x).view(B, T, N, D).transpose(1, 2)
        k = self.k_net(x).view(B, T, N, D).transpose(1, 2)
        v = self.v_net(x).view(B, T, N, D).transpose(1, 2)

        # weights (B, N, T, T)
        weights = q @ k.transpose(2, 3) / math.sqrt(D)
        # causal mask applied to weights
        print(weights.shape)
        print(self.mask[...,:T,:T].shape)
        weights = weights.masked_fill(
            self.mask[..., :T, :T] == 0, float("-inf")
        )
        # normalize weights, all -inf -> 0 after softmax
        normalized_weights = F.softmax(weights, dim=-1)

        # attention (B, N, T, D)
        attention = self.att_drop(normalized_weights @ v)

        # gather heads and project (B, N, T, D) -> (B, T, N*D)
        attention = attention.transpose(1, 2).contiguous().view(B, T, N * D)

        out = self.proj_drop(self.proj_net(attention))
        return out


class TB(nn.Module):
    def __init__(
            self,
            h_dim,
            max_T,
            n_heads,
            drop_p,
            num_inputs,
            mgdt=False,
            dt_mask=False,
            att_mask=None,
    ):
        super().__init__()
        self.num_inputs = num_inputs
        self.attention = MaskedCausalAttention(
            h_dim,
            max_T,
            n_heads,
            drop_p,
            num_inputs,
            mgdt=mgdt,
            dt_mask=dt_mask,
            att_mask=att_mask,
        )
        self.mlp = nn.Sequential(
            nn.Linear(h_dim, 4 * h_dim),
            nn.GELU(),
            nn.Linear(4 * h_dim, h_dim),
            nn.Dropout(drop_p),
        )
        self.ln1 = nn.LayerNorm(h_dim)
        self.ln2 = nn.LayerNorm(h_dim)

    def forward(self, x):
        # Attention -> LayerNorm -> MLP -> LayerNorm
        x = x + self.attention(x)  # residual
        x = self.ln1(x)
        x = x + self.mlp(x)  # residual
        x = self.ln2(x)
        return x
