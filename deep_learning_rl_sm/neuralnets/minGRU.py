import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch import nn

def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

#@torch.compile
def log_g(x):
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

#@torch.compile
def parallel_scan_log(log_coefficients, log_values):
    # log_coefficients: (batch_size, device, input_size)
    # log_values: (batch_size, device + 1, input_size)
    a_star = F.pad(torch.cumsum(log_coefficients, dim=1), (0, 0, 1, 0))
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    log_h = a_star + log_h0_plus_b_star
    return torch.exp(log_h)[:, 1:]

class Conv1dLayer(Module):
    def __init__(self, dim, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim)
    #@torch.compile()
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d

class minGRU(Module):
    def __init__(self, dim,batch_size, device, expansion_factor = 1.):
        super().__init__()
        self.dim=dim
        self.exp_dim = int(dim * expansion_factor)
        self.log_h = log_g(torch.zeros((batch_size, 1, self.exp_dim), device = device))
        self.batch_size = batch_size
        self.f = Linear(dim, 2*self.exp_dim, device = device)
        self.down_projection = Linear(self.exp_dim,dim, bias=False, device = device) if expansion_factor != 1 else None
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
    
    def eval_mode(self):
        self.log_h = log_g(torch.zeros((1, 1, self.exp_dim), device = self.log_h.device))
        
    def train_mode(self):
        self.log_h = log_g(torch.zeros((self.batch_size, 1, self.exp_dim), device = self.log_h.device))
    
    #@torch.compile
    def forward(self, x:torch.Tensor, h0=None):
        # x: (batch_size, device, input_size)
        # h_0: (batch_size, 1, hidden_size)
        k,h_x = self.f(x).chunk(2,dim = -1)
        log_z = -F.softplus(-k)
        log_coeffs = -F.softplus(k)
        log_tilde_h = log_g(h_x)
        if self.down_projection is not None:
            return self.down_projection(parallel_scan_log(log_coeffs, torch.cat([self.log_h,log_tilde_h + log_z], dim=1)))
        return parallel_scan_log(log_coeffs, torch.cat([self.log_h,log_tilde_h + log_z], dim=1))
    
class CausalDepthWiseConv1d(Module):
    def __init__(self, dim, kernel_size, device):
        """Simple sequential CONV1D net"""
        super().__init__()
        self.kernel_size = kernel_size
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size = kernel_size, groups = dim, device = device),
            nn.Conv1d(dim, dim, kernel_size = 1, device = device)
        )
    #@torch.compile
    def forward(self, x):
        x = x.transpose(1, 2) # b n d -> b d n
        x = F.pad(x, (self.kernel_size - 1, 0), value = 0.)
        x = self.net(x)
        return x.transpose(1, 2) # b d n -> b n d    
    
class minGRUCell(Module):
    """This Version corresponds to what has been done in https://github.com/lucidrains/minGRU-pytorch/"""
    def __init__(self,dim,drop_p,kernel_size,expansion_factor,batch_size,device, conv, mult=4):
        """This Version corresponds to what has been done in https://github.com/lucidrains/minGRU-pytorch/"""
        super().__init__()
        self.conv = CausalDepthWiseConv1d(dim, kernel_size, device = device) if conv else None #Conv1dLayer(dim,kernel_size)
        self.ln1 = torch.nn.LayerNorm(dim, device = device)
        self.cell = minGRU(dim,batch_size,device,expansion_factor)
        self.ln2 = torch.nn.LayerNorm(dim, device = device)
        self.mlp = nn.Sequential(
                nn.Linear(dim, int(mult * dim), device = device),
                nn.GELU(),#Reinformer uses GELU
                nn.Linear(int(mult * dim), dim, device = device),
                nn.Dropout(drop_p),
            ) 
        self.ln3 = torch.nn.LayerNorm(dim, device = device)
    #@torch.compile
    def forward(self,x):
        residual = x
        if self.conv is not None:
            x = self.conv(self.ln1(x)) + residual
            residual = x
        x = self.cell(self.ln2(x)) + residual
        residual = x
        return self.mlp(self.ln3(x)) + residual



class minGRUBlock(Module):
    """This Version corresponds to what has been done in https://github.com/cheind/mingru/blob/main/mingru/modules.py"""
    def __init__(self,dim,drop_p,kernel_size,expansion_factor,batch_size,device, conv, n_layers, mult=4):
        """This Version corresponds to what has been done in https://github.com/cheind/mingru/blob/main/mingru/modules.py"""
        super().__init__()
        self.conv = CausalDepthWiseConv1d(dim, kernel_size, device = device) if conv else None #Conv1dLayer(dim,kernel_size)
        self.cells = []
        for i in range(n_layers):
            self.cells.append(minGRU(dim,batch_size,device,expansion_factor))
        self.cells = nn.ModuleList(self.cells)
        self.mlp = nn.Sequential(
                nn.Linear(dim, mult * dim, device = device),
                nn.GELU(),#Reinformer uses GELU
                nn.Linear(mult * dim, dim, device = device),
                nn.Dropout(drop_p),
            ) 
        self.ln3 = torch.nn.LayerNorm(dim, device = device)
    #@torch.compile
    def forward(self,x):
        residual = x
        if self.conv is not None:
            x = self.conv(nn.LayerNorm()(x)) + residual
            residual = x
        for cell in self.cells:
            x = cell(nn.LayerNorm()(x)) + residual
        residual = x
        return self.mlp(self.ln3(x))

