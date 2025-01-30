import torch
import torch.nn as nn
from models.rev_in import RevNorm
from models.tsmixer import ResidualBlock

class TSMixerRevNorm(nn.Module):
    """TSMixer with Reversible Instance Normalization."""
    
    def __init__(self, input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice=None):
        super().__init__()
        print("input_shape: ", input_shape)
        self.input_dim = input_shape[-1]
        self.seq_len = input_shape[-2]
        self.pred_len = pred_len
        self.target_slice = target_slice
        
        self.rev_norm = RevNorm(num_features=self.input_dim, axis=-1)
        self.blocks = nn.ModuleList([
            ResidualBlock(self.input_dim, self.seq_len, norm_type, activation, dropout, ff_dim)
            for _ in range(n_block)
        ])
        
        self.output_fc = nn.Linear(self.seq_len, pred_len)
    
    def forward(self, x):
        print("x shape before rev_norm: ", x.shape)
        x = self.rev_norm(x, mode='norm')
        print("x shape after rev_norm: ", x.shape)
        for block in self.blocks:
            x = block(x)
        
        if self.target_slice:
            x = x[:, :, self.target_slice]
        
        x = self.output_fc(x.transpose(1, 2)).transpose(1, 2)
        x = self.rev_norm(x, mode='denorm', target_slice=self.target_slice)
        
        return x
