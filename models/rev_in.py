import torch
import torch.nn as nn

class RevNorm(nn.Module):
    """Reversible Instance Normalization in PyTorch."""
    
    def __init__(self, axis, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine
        self.mean = None
        self.stdev = None
        
    def build(self, input_shape):
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(input_shape[-1]))
            self.affine_bias = nn.Parameter(torch.zeros(input_shape[-1]))
    
    def forward(self, x, mode, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError("Mode must be 'norm' or 'denorm'")
        return x
    
    def _get_statistics(self, x):
        self.mean = x.mean(dim=self.axis, keepdim=True).detach()
        self.stdev = (x.var(dim=self.axis, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
    
    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x, target_slice=None):
        if self.affine:
            x = (x - self.affine_bias[target_slice]) / self.affine_weight[target_slice]
        x = x * self.stdev[:, :, target_slice] + self.mean[:, :, target_slice]
        return x