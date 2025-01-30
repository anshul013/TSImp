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

"""Implementation of Reversible Instance Normalization."""

import torch
import torch.nn as nn


class RevNorm(nn.Module):
  """Reversible Instance Normalization."""

  def __init__(self, axis=-2, eps=1e-5, affine=True):
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
      raise NotImplementedError
    return x

  def _get_statistics(self, x):
    # Keep dims for broadcasting
    keepdim = True
    self.mean = x.mean(dim=self.axis, keepdim=keepdim).detach()
    self.stdev = torch.sqrt(
        x.var(dim=self.axis, keepdim=keepdim, unbiased=False) + self.eps
    ).detach()

  def _normalize(self, x):
    x = (x - self.mean) / self.stdev
    if self.affine:
      x = x * self.affine_weight + self.affine_bias
    return x

  def _denormalize(self, x, target_slice=None):
    if self.affine:
      if target_slice is not None:
        weight = self.affine_weight[target_slice]
        bias = self.affine_bias[target_slice]
      else:
        weight = self.affine_weight
        bias = self.affine_bias
      x = (x - bias) / weight

    if target_slice is not None:
      stdev = self.stdev[:, :, target_slice]
      mean = self.mean[:, :, target_slice]
    else:
      stdev = self.stdev
      mean = self.mean

    x = x * stdev + mean
    return x