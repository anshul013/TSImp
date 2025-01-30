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

"""Implementation of TSMixer."""

import torch
import torch.nn as nn

class ResBlock(nn.Module):
    """Residual block of TSMixer with correct normalization strategy."""
    
    def __init__(self, input_dim, ff_dim, norm_type='L', activation='relu', dropout=0.1):
        super().__init__()
        self.norm_type = norm_type
        
        # Select normalization type
        if norm_type == 'L':
            self.norm1 = nn.LayerNorm(input_dim)
            self.norm2 = nn.LayerNorm(input_dim)
        else:  # 'B'
            self.norm1 = nn.BatchNorm1d(input_dim)
            self.norm2 = nn.BatchNorm1d(input_dim)
            
        # Select activation
        self.activation = nn.ReLU() if activation == 'relu' else nn.GELU()
        
        # Temporal mixing layers
        self.temporal_mix = nn.Linear(input_dim, input_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feature mixing layers
        self.feature_mix1 = nn.Linear(input_dim, ff_dim)
        self.feature_mix2 = nn.Linear(ff_dim, input_dim)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, channels]
        
        # 1️⃣ Temporal Mixing Block
        if self.norm_type == 'L':
            normed = self.norm1(x)
        else:
            normed = self.norm1(x.transpose(1, 2)).transpose(1, 2)
            
        # Temporal mixing
        temporal = normed.transpose(1, 2)  # [batch, channels, seq_len]
        temporal = self.temporal_mix(temporal)  # Mix across sequence dimension
        temporal = self.activation(temporal)
        temporal = temporal.transpose(1, 2)  # [batch, seq_len, channels]
        temporal = self.dropout1(temporal)
        
        # First residual connection
        res = temporal + x
        
        # 2️⃣ Feature Mixing Block
        if self.norm_type == 'L':
            normed = self.norm2(res)
        else:
            normed = self.norm2(res.transpose(1, 2)).transpose(1, 2)
            
        # Feature mixing
        feature = self.feature_mix1(normed)
        feature = self.activation(feature)
        feature = self.dropout2(feature)
        feature = self.feature_mix2(feature)
        feature = self.dropout3(feature)
        
        # Second residual connection
        return feature + res

class TSMixer(nn.Module):
    def __init__(
        self,
        input_shape,
        pred_len,
        norm_type='L',
        activation='relu',
        n_block=2,
        dropout=0.1,
        ff_dim=2048,
        target_slice=None
    ):
        super().__init__()
        seq_len, channels = input_shape
        self.target_slice = target_slice
        
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
    """Build TSMixer model."""
    model = TSMixer(
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