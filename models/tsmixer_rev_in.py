# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementation of TSMixer with Reversible Instance Normalization."""

import torch
import torch.nn as nn
from models.rev_in import RevNorm
from models.tsmixer import ResBlock

class TSMixerRevIn(nn.Module):
    def __init__(
        self,
        input_shape,
        pred_len,
        norm_type,
        activation,
        n_block,
        dropout,
        ff_dim,
        target_slice=None
    ):
        super().__init__()
        seq_len, channels = input_shape
        self.target_slice = target_slice
        
        # RevNorm layer - normalize across channel dimension (last dimension)
        self.rev_norm = RevNorm(num_features=channels, axis=-1)
        
        # Stack multiple residual blocks
        self.blocks = nn.ModuleList([
            ResBlock(
                input_dim=channels,
                ff_dim=ff_dim,
                norm_type=norm_type,
                activation=activation,
                dropout=dropout
            ) for _ in range(n_block)
        ])
        
        # Final prediction layer
        self.predictor = nn.Linear(seq_len, pred_len)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        
        # Apply reversible normalization
        x = self.rev_norm(x, mode='norm')
        
        # Pass through residual blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply target slice if specified
        if self.target_slice is not None:
            x = x[:, :, self.target_slice]
            
        # Make predictions
        x = x.transpose(1, 2)  # [batch, channels, seq_len]
        x = self.predictor(x)  # [batch, channels, pred_len]
        x = x.transpose(1, 2)  # [batch, pred_len, channels]
        
        # Denormalize the output
        x = self.rev_norm(x, mode='denorm', target_slice=self.target_slice)
        
        return x

def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
    """Build TSMixer with Reversible Instance Normalization model."""
    model = TSMixerRevIn(
        input_shape=input_shape,
        pred_len=pred_len,
        norm_type=norm_type,
        activation=activation,
        n_block=n_block,
        dropout=dropout,
        ff_dim=ff_dim,
        target_slice=target_slice
    )
    return model