import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch import nn

def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))


def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))


def parallel_scan_log(log_coefficients, log_values):
    # log_coefficients: (batch_size, device, input_size)
    # log_values: (batch_size, device + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coefficients, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

class minGRU(Module):
    def __init__(self, dim,batch_size, device, expansion_factor = 1., dropout = 0.):
        super().__init__()
        self.dim=dim
        self.device = device
        self.exp_dim = int(dim * expansion_factor)
        self.batch_size = batch_size
        self.f = Linear(dim, 2*self.exp_dim, device = device)
        self.drop_f = nn.Dropout(dropout)
        self.down_projection = Linear(self.exp_dim, dim, bias=False, device = device) if expansion_factor != 1.0 else None
        self.drop_proj = nn.Dropout(dropout)
        # output of f_z can be viewed as the proportion of the info from the current timestep that is incorporated into
        # the next hidden state (for more info see paper "Were RNNs All We Needed?")

        # This code is also available in the paper "Were RNNs All We Needed?"
        # We could still change the code to our specifications for this project
        # however original code is already written extremely cleanly and i see no reason to
        # change names, rearrange code etc. for the sake of it

        """
        Note: This version is not in log-space
        Sequential Algorithm for minGRU:
                z_t ← σ(f_z(x_t))
                h˜_t ← g(f_h(x_t))
                h_t ← (1 − z_t) ⊙ h_{t−1} + z_t ⊙ h˜_t
        """

        # We use the log-space version of the algorithm for additional numerical stability
        # (i.e. long sequences more likely to result in numerical underflow)
        # by e.g. converting to log-values, summing and then exponentiating we achieve the same result as
        # multiplying the original values but with better numerical stability
    
    def reset_h_prev(self):
        self.h_prev = torch.zeros((1,1,self.exp_dim),device=self.device)
    
    
    def forward(self, x:torch.Tensor, log_h_0:torch.Tensor):
        # x: (batch_size, seq_len, hidden_size)
        # h_0: (batch_size, 1, hidden_size)
        k,h_x = self.f(x).chunk(2,dim = -1)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_tilde_h = log_g(h_x)
        h_t = parallel_scan_log(log_coeffs, torch.cat([log_h_0,log_tilde_h + log_z], dim=1))
        if self.down_projection is not None:
            h =  self.down_projection(h_t)
        return self.drop_proj(h), h_t[:,2:-1:3].log()
    
    def seq_forward(self, x:torch.Tensor):
        # x: (1,1, hidden_size)
        # h_0: (1,1 hidden_size)
        k,h_x = self.f(x).chunk(2,dim = -1)
        z = torch.sigmoid(k)
        h_tilde = g(h_x)
        h_t = (1 - z) * self.h_prev + z * h_tilde
        self.h_prev = h_t.detach()
        return self.down_projection(h_t)

        
    
class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size, device):
        """Simple sequential CONV1D net"""
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim, device = device),
            nn.Conv1d(dim, dim, kernel_size = 1, device = device)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d    
    
class minGRUCell(Module):
    def __init__(self,dim,drop_p,kernel_size,expansion_factor,batch_size,device, conv, mult=4):
        super().__init__()
        self.conv = CausalDepthWiseConv1d(dim, kernel_size, device = device) if conv else None 
        self.ln1 = torch.nn.LayerNorm(dim, device = device)
        self.cell = minGRU(dim,batch_size,device,expansion_factor, drop_p)
        self.ln2 = torch.nn.LayerNorm(dim, device = device)
        self.mlp = nn.Sequential(
                nn.Linear(dim, int(mult * dim), device = device),
                nn.GELU(),#Reinformer uses GELU
                nn.Linear(int(mult * dim), dim, device = device),
                nn.Dropout(drop_p),
            ) if mult != 0 else None
        self.ln3 = torch.nn.LayerNorm(dim, device = device)
    
    def forward(self,x, h_0s):
        if self.conv is not None:
            x = self.ln1(x + self.conv(x))
        cell_out, h_0 = self.cell(x, h_0s[0])
        x = self.ln2(x + cell_out)
        if self.mlp is not None:
            return self.ln3(x + self.mlp(x)), h_0s[1:] + [h_0]
        
    def seq_forward(self,x):
        if self.conv is not None:
            x = self.ln1(x + self.conv(x))
        x = self.ln2(x + self.cell.seq_forward(x))
        if self.mlp is not None:
            return self.ln3(x + self.mlp(x))

