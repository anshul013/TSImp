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

"""Load raw data and generate time series dataset."""

import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './dataset/'

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len, target_slice=None):
        self.data = torch.FloatTensor(data)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_slice = target_slice
        
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        if self.target_slice is not None:
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_slice]
        else:
            y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return x, y

class TSFDataLoader:
    """Generate data loader from raw data."""

    def __init__(
        self, data, batch_size, seq_len, pred_len, feature_type, target='OT'
    ):
        self.data = data
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.target = target
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        if not os.path.isdir(LOCAL_CACHE_DIR):
            os.mkdir(LOCAL_CACHE_DIR)

        file_name = self.data + '.csv'
        cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
        
        df_raw = pd.read_csv(cache_filepath)
        
        # S: univariate-univariate, M: multivariate-multivariate, MS: multivariate-univariate
        df = df_raw.set_index('date')
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        # split train/valid/test
        n = len(df)
        if self.data.startswith('ETTm'):
            train_end = 12 * 30 * 24 * 4
            val_end = train_end + 4 * 30 * 24 * 4
            test_end = val_end + 4 * 30 * 24 * 4
        elif self.data.startswith('ETTh'):
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        else:
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]

        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def inverse_transform(self, data):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy()
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        dataset = TimeSeriesDataset(
            self.train_df.values,
            self.seq_len,
            self.pred_len,
            self.target_slice
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0
        )

    def get_val(self):
        dataset = TimeSeriesDataset(
            self.val_df.values,
            self.seq_len,
            self.pred_len,
            self.target_slice
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )

    def get_test(self):
        dataset = TimeSeriesDataset(
            self.test_df.values,
            self.seq_len,
            self.pred_len,
            self.target_slice
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0
        )