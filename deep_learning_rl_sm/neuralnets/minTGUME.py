import torch
import torch.nn.functional as F
from torch.nn import Linear, Module
from torch import nn
from typing import List, Optional, Tuple

# --- Helper functions from minLSTM.py, kept for consistency ---
def g(x):
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

def log_g(x):
    """ Custom log-space activation function. """
    return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

def parallel_scan_log(log_coefficients, log_values):
    """
    Parallel scan implementation in log space.
    This performs the core recurrent calculation efficiently.
    Recurrence solved: C_t = a_t * C_{t-1} + b_t
    
    Args:
        log_coefficients (Tensor): Log of the multiplicative coefficients (a_t), i.e., log(t_mult).
        log_values (Tensor): Log of the additive values (b_t), i.e., log(t_add).
    
    Returns:
        Tensor: The result of the recurrence (C) in normal space.
    """
    # Pad coefficients to align with values for cumulative sum
    a_star = F.pad(torch.cumsum(log_coefficients, dim=1), (0, 0, 1, 0))
    # Perform the log-space cumulative sum (logcumsumexp)
    log_h0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
    # Combine the terms to get the final log-space result
    log_h = a_star + log_h0_plus_b_star
    # Return states in normal space, removing the initial state
    return torch.exp(log_h)[:, 1:]

class C0_Predictor(Module):
    """
    A dedicated module to predict the initial cell state C_0 from the
    content of the input sequence x using a backward GRU summarizer.
    """
    def __init__(self, dim, exp_dim, device):
        super().__init__()
        # A lightweight GRU to process the sequence backward and create a summary.
        self.summarizer = nn.GRU(input_size=dim, hidden_size=dim, num_layers=1, batch_first=True, device=device)
        # A linear layer to project the GRU's summary to the C_0 dimension.
        self.projector = nn.Linear(dim, exp_dim, device=device)

    def forward(self, x):
        """
        Predicts log_C_0 from the input sequence x.
        """
        # Reverse the sequence along the time dimension (L)
        # x shape: (B, L, D)
        x_reversed = torch.flip(x, dims=[1])

        # The final hidden state of the GRU on the reversed sequence is our summary.
        # It captures information from the beginning of the original sequence.
        # GRU returns (output_sequence, final_hidden_state_h_n)
        # h_n shape: (num_layers*num_directions, B, D_hidden) -> (1, B, D)
        _, summary = self.summarizer(x_reversed)

        # Squeeze to remove the num_layers dimension: (B, D)
        summary = summary.squeeze(0)

        # Project the summary to get the initial state C_0 logits
        c0_logits = self.projector(summary)

        # Apply activation and add sequence dimension for concatenation
        # Shape: (B, 1, D_exp)
        log_C_0 = log_g(c0_logits).unsqueeze(1)
        return log_C_0
    
class minTGU_ME(nn.Module):
    """
    A parallel implementation of the TGU-ME architecture, inspired by minLSTM.
    """
    def __init__(self, dim,batch_size, device, expansion_factor = 1., dropout = 0.):
        super(minTGU_ME, self).__init__()
        self.dim = dim
        self.exp_dim = int(dim * expansion_factor)
        self.batch_size = batch_size

        self.log_h = log_g(torch.zeros((batch_size, 1, self.exp_dim), device = device))
        """self.c0_predictor = C0_Predictor(dim, self.exp_dim, device=device)
        self.C_0 = g(torch.zeros((batch_size, 1, self.exp_dim), device=device))
        self.log_C_0 = log_g(torch.zeros((batch_size, 1, self.exp_dim), device=device))"""
        
        # A single linear layer to project the input `x` to get all 6 gate components.
        self.linear_x_to_all = nn.Linear(self.dim, self.exp_dim * 6, device=device)
        self.drop_gates = nn.Dropout(dropout)
        
        # Optional down-projection layer if expansion is used
        self.down_projection = Linear(self.exp_dim, dim, bias=False, device=device) if expansion_factor != 1.0 else None
        self.down_projection_m = Linear(self.exp_dim, dim, bias=False, device=device) if expansion_factor != 1.0 else None
        self.drop_proj = nn.Dropout(dropout)
        
    def step(self, x_t):
        """
        Performs a single, sequential, non-log-space step for efficient inference.
        
        Args:
            x_t (Tensor): Input for the current timestep, shape (B, D).
            c_prev (Tensor): Cell state from the previous timestep, shape (B, D_exp).

        Returns:
            tuple: A tuple containing:
                - h (Tensor): The hidden state for the current timestep.
                - m (Tensor): The memory state for the current timestep.
                - c_new (Tensor): The cell state for the current timestep.
        """

        # Get gate logits from the single timestep input
        k_c, k_x, k_t_mult, k_t_add, k_e, k_o = self.linear_x_to_all(x_t).chunk(6, dim=-1)

        # --- Standard Space Gate Calculations ---
        c_gate = torch.sigmoid(k_c)
        context_candidate = torch.tanh(k_x) # Using tanh as the non-log equivalent of g(x)
        context_input = c_gate * context_candidate

        t_mult = torch.sigmoid(k_t_mult)
        t_add = torch.tanh(k_t_add) # Using tanh as the non-log equivalent of g(x)
        
        e_gate = torch.sigmoid(k_e)
        o_gate = torch.sigmoid(k_o)

        # --- Core Recurrence ---
        c_new = t_mult * self.C_0 + t_add
        self.C_0 = c_new
        # --- Final State Calculations ---
        tanh_c = torch.tanh(c_new)
        m = e_gate * tanh_c
        h = o_gate * tanh_c + (1 - o_gate) * context_input
        
        # Apply optional down-projection
        if self.down_projection is not None:
            h = self.down_projection(h)
            m = self.down_projection(m)

        return h
    
    
    def eval_mode(self):
        self.log_h = log_g(torch.zeros((1, 1, self.exp_dim), device = self.log_h.device))
        
    def train_mode(self):
        self.log_h = log_g(torch.zeros((self.batch_size, 1, self.exp_dim), device = self.log_h.device))
    

    def forward(self, x):
        """
        Processes a whole sequence in parallel.
        
        Args:
            x (Tensor): The input sequence, shape (batch_size, seq_len, dim).
        
        Returns:
            tuple: A tuple containing:
                   - h (Tensor): The blended hidden state signal. (batch_size, seq_len, dim)
                   - m (Tensor): The memory state signal. (batch_size, seq_len, dim)
        """
        """#if self.predict_c0:
        self.log_C_0 = self.c0_predictor(x)"""
        # Project input `x` and chunk into 6 parts for gate logits
        k_c, k_x, k_t_mult, k_t_add, k_e, k_o = self.drop_gates(self.linear_x_to_all(x)).chunk(6, dim=-1)

        # --- Log-space gate calculations ---
        # Context Gate and Focused Input
        log_c_t = -F.softplus(-k_c)
        log_context_candidate = log_g(k_x)
        log_context_input = log_c_t + log_context_candidate

        # Transform Gate (for C_t recurrence)
        log_t_mult = -F.softplus(-k_t_mult) # This is log(a_t) for the recurrence
        log_t_add = log_g(k_t_add)          # This is log(b_t) for the recurrence
        
        # Exposure and Output Gates
        log_e_t = -F.softplus(-k_e)
        log_o_t = -F.softplus(-k_o)

        # --- Core Recurrence ---
        # Solve for the cell state C across the sequence in parallel
        # The result C is returned in normal space
        C = parallel_scan_log(log_t_mult, torch.cat([self.log_h, log_t_add], dim=1))
        
        # --- Final State Calculations ---
        # Get log(tanh(C)) for use in h and m calculations
        log_tanh_C = torch.log(torch.tanh(C).abs() + 1e-9)

        # 1. Calculate the memory state m_t = e_t * tanh(C_t)
        log_m = log_e_t + log_tanh_C
        m = torch.exp(log_m)
        
        # 2. Calculate the hidden state h_t = o_t * tanh(C_t) + (1-o_t) * context_input
        # In log space: h = logaddexp(log(o_t*tanh(C_t)), log((1-o_t)*context_input))
        log_term1_h = log_o_t + log_tanh_C
        log_term2_h = -F.softplus(k_o) + log_context_input # -softplus(k) is log(1-sigmoid(k))
        log_h = torch.logaddexp(log_term1_h, log_term2_h)
        h = torch.exp(log_h)

        # Apply optional down-projection
        if self.down_projection is not None:
            h = self.down_projection(h)
            m = self.down_projection_m(m)
        
        return self.drop_proj(h)
  
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
    
class minTGU_MECell(Module):
    """
    A full TGU-ME block structured similarly to the provided minLSTMCell example.
    It includes optional causal convolution, layer normalization, residual connections,
    and a final MLP layer.
    """
    def __init__(self,dim,drop_p,kernel_size,expansion_factor,batch_size,device, conv, mult=4):
        super().__init__()
        self.conv = CausalDepthWiseConv1d(dim, kernel_size, device=device) if conv else None
        self.ln1 = torch.nn.LayerNorm(dim, device=device)
        self.cell = minTGU_ME(dim,batch_size,device,expansion_factor, drop_p)
        self.ln2 = torch.nn.LayerNorm(dim, device=device)
        
        self.hm_projection = nn.Linear(dim * 2, dim, device=device)

        self.mlp = nn.Sequential(
            nn.Linear(dim, int(mult * dim), device=device),
            nn.GELU(),
            nn.Linear(int(mult * dim), dim, device=device),
            nn.Dropout(drop_p),
        )
        self.ln3 = torch.nn.LayerNorm(dim, device=device)

    def forward(self, x):
        """
        Forward pass through the TGU-ME block.
        Follows the pattern: Pre-Norm -> SubLayer -> Residual Connection
        """
        residual = x

        # 1. Optional Causal Convolution Block
        if self.conv is not None:
            x = self.conv(self.ln1(x)) + residual
            residual = x

        # 2. Core TGU-ME Cell Block
        # The TGU-ME cell returns (h, m). We combine them with a linear projection.
        h, m = self.cell(self.ln2(x))
        """combined_hm = torch.cat([h, m], dim=-1)
        projected_hm = self.hm_projection(combined_hm)
        x = projected_hm + residual
        residual = x"""

        # 3. Final MLP Block
        x = self.mlp(self.ln3(h)) + residual
        return x, m


class minTGU_ME_Block(Module):
    """
    A stack of multiple minTGU_MECell layers that correctly handles
    the passing of states for sequential chunk processing.
    This is the main entry point for using the model.
    """
    def __init__(self,dim,drop_p,kernel_size,expansion_factor,batch_size,device, conv, n_layers, mult=4):
        super().__init__()
        self.conv = CausalDepthWiseConv1d(dim, kernel_size, device = device) if conv else None
        self.layers = nn.ModuleList([
            minTGU_MECell(dim, drop_p, kernel_size, expansion_factor, device, conv, mult)
            for _ in range(n_layers)
        ])
        
    def reset(self):
        for l in self.layers:
            self.C_0 = g(torch.zeros((l.batch_size, 1, l.exp_dim), device=l.device))

    def forward(self, x):
        """
        Forward pass through the stack of TGU-ME layers.

        Args:
            x (Tensor): The input sequence for the current chunk.
            c0s (List[Tensor], optional): A list of initial log cell states, one for each layer.
                                          Used for sequential inference/training.

        Returns:
            tuple: A tuple containing:
                - x (Tensor): The final output of the entire block.
                - all_m (List[Tensor]): A list of memory states from each layer.
                - all_log_c_finals (List[Tensor]): A list of final cell states from each layer,
                                                   to be passed as `c0s` to the next chunk.
        """
        for i, layer in enumerate(self.layers):
            # Pass the input x and the initial state for the current layer
            x, m = layer(x)
        return x, m
    
    def step(self, x_t):
        """
        Forward pass through the stack of TGU-ME layers.

        Args:
            x (Tensor): The input sequence for the current chunk.
            c0s (List[Tensor], optional): A list of initial log cell states, one for each layer.
                                          Used for sequential inference/training.

        Returns:
            tuple: A tuple containing:
                - x (Tensor): The final output of the entire block.
                - all_m (List[Tensor]): A list of memory states from each layer.
                - all_log_c_finals (List[Tensor]): A list of final cell states from each layer,
                                                   to be passed as `c0s` to the next chunk.
        """
        for i, layer in enumerate(self.layers):
            # Pass the input x and the initial state for the current layer
            x_t, m = layer.step(x_t)
        return x_t, m