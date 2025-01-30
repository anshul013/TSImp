import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """Residual block of TSMixer with correct normalization strategy."""
    def __init__(self, input_dim, seq_len, norm_type, activation, dropout, ff_dim):
        super().__init__()
        self.norm_type = norm_type
        self.activation = getattr(F, activation) if hasattr(F, activation) else None
        
        # Select normalization type
        if norm_type == 'L':
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
        else:
            self.norm1 = nn.BatchNorm1d(seq_len)
            self.norm2 = nn.BatchNorm1d(input_dim)
        
        # Temporal Mixing Block
        self.temporal_fc = nn.Linear(seq_len, seq_len)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feature Mixing Block
        self.feature_fc1 = nn.Linear(input_dim, ff_dim)
        self.feature_fc2 = nn.Linear(ff_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        # All operations will automatically use the same device as input x
        
        # Temporal Mixing Block
        if self.norm_type == 'L':
            x_norm = self.norm1(x)
        else:
            x_norm = self.norm1(x)
        
        x_t = self.temporal_fc(x_norm.transpose(1, 2)).transpose(1, 2)
        if self.activation:
            x_t = self.activation(x_t)
        x_t = self.dropout1(x_t)
        res = x_t + x
        
        # Feature Mixing Block
        if self.norm_type == 'L':
            x_norm = self.norm2(res)
        else:
            x_norm = self.norm2(res.transpose(1, 2)).transpose(1, 2)
        
        x_f = self.feature_fc1(x_norm)
        if self.activation:
            x_f = self.activation(x_f)
        x_f = self.dropout2(x_f)
        x_f = self.feature_fc2(x_f)
        x_f = self.dropout3(x_f)
        
        return x_f + res

class TSMixer(nn.Module):
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice=None):
        super().__init__()
        self.input_dim = input_shape[-1]
        self.seq_len = input_shape[0]
        self.pred_len = pred_len
        self.target_slice = target_slice
        
        self.blocks = nn.ModuleList([
            ResidualBlock(self.input_dim, self.seq_len, norm_type, activation, dropout, ff_dim)
            for _ in range(n_block)
        ])
        
        self.output_fc = nn.Linear(self.seq_len, pred_len)
    
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        
        if self.target_slice:
            x = x[:, :, self.target_slice]
        
        x = self.output_fc(x.transpose(1, 2)).transpose(1, 2)
        return x