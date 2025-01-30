import torch
import torch.nn as nn
import torch.nn.functional as F

class RevNorm(nn.Module):
    """Reversible Instance Normalization in PyTorch."""
    
    def __init__(self, axis, eps=1e-5, affine=True, num_features=None):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine
        self.mean = None
        self.stdev = None
        
        if self.affine:
            if num_features is None:
                raise ValueError("num_features must be provided when affine=True")
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))
    
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
        # Calculate statistics along the sequence dimension (axis=-2)
        self.mean = x.mean(dim=self.axis, keepdim=True).detach()
        self.stdev = (x.var(dim=self.axis, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
        
        # Ensure statistics are on the same device as input
        self.mean = self.mean.to(x.device)
        self.stdev = self.stdev.to(x.device)
    
    def _normalize(self, x):
        x = (x - self.mean) / self.stdev
        if self.affine:
            x = x * self.affine_weight + self.affine_bias
        return x
    
    def _denormalize(self, x, target_slice=None):
      if self.affine:
        x = (x - self.affine_bias[target_slice]) / self.affine_weight[target_slice]
    
      # Always resize self.stdev and self.mean to match x.shape[1] & (pred_len)
      stdev_resized = F.interpolate(self.stdev.permute(0, 2, 1), size=x.shape[1], mode='nearest').permute(0, 2, 1)
      mean_resized = F.interpolate(self.mean.permute(0, 2, 1), size=x.shape[1], mode='nearest').permute(0, 2, 1)
      
      x = x * stdev_resized + mean_resized
      return x

