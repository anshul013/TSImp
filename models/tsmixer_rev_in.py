import torch
import torch.nn as nn
from models.rev_in import RevNorm
from models.tsmixer import ResidualBlock

class TSMixerRevNorm(nn.Module):
    """TSMixer with Reversible Instance Normalization."""
    
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice=None):
        super().__init__()
        self.input_dim = input_shape[1]  # number of features
        self.seq_len = input_shape[0]    # sequence length
        self.pred_len = pred_len
        self.target_slice = target_slice
        
        self.rev_norm = RevNorm(axis=-1, num_features=self.input_dim)
        self.blocks = nn.ModuleList([
            ResidualBlock(input_dim=self.input_dim, 
                         seq_len=self.seq_len,
                         norm_type=norm_type,
                         activation=activation,
                         dropout=dropout,
                         ff_dim=ff_dim)
            for _ in range(n_block)
        ])
        
        self.output_fc = nn.Linear(self.seq_len, pred_len)
    
    def forward(self, x):
        # All operations will automatically use the same device as input x
        x = self.rev_norm(x, mode='norm')
        for block in self.blocks:
            x = block(x)
        
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]
        
        x = self.output_fc(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev_norm(x, mode='denorm', target_slice=self.target_slice)
        
        return x
