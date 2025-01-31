"""Load raw data and generate time series dataset."""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

DATA_DIR = 'gs://time_series_datasets'
LOCAL_CACHE_DIR = './dataset/'

class TSFDataLoader(Dataset):
    """Generate data loader from raw data."""
    
    def __init__(self, data, seq_len, pred_len, feature_type, batch_size, target='OT'):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.target = target
        self.target_slice = slice(0, None)
        
        self._read_data()
    
    def _read_data(self):
        """Load raw data and split datasets."""
        os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)
        
        file_name = self.data + '.csv'
        cache_filepath = os.path.join(LOCAL_CACHE_DIR, file_name)
        if not os.path.isfile(cache_filepath):
            import shutil
            shutil.copy(os.path.join(DATA_DIR, file_name), cache_filepath)
        
        df_raw = pd.read_csv(cache_filepath)
        df = df_raw.set_index('date')
        
        if self.feature_type == 'S':
            df = df[[self.target]]
        elif self.feature_type == 'MS':
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)
        
        # Split train/valid/test
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
        val_df = df[train_end - self.seq_len:val_end]
        test_df = df[val_end - self.seq_len:test_end]
        
        # Standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)
        
        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)
        
        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]
        
    def __len__(self):
        return len(self.train_df) - (self.seq_len + self.pred_len) + 1
    
    def __getitem__(self, idx):
        data = self.train_df.values
        inputs = data[idx:idx + self.seq_len, :]
        labels = data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_slice]
        
        # Convert to tensors and ensure float32 dtype
        inputs = torch.FloatTensor(inputs)
        labels = torch.FloatTensor(labels)
        
        return inputs, labels
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def get_train(self, shuffle=True):
        print(f"Train DataLoader Samples: {len(self.train_df)}")
        return DataLoader(self, batch_size=self.batch_size, shuffle=shuffle, drop_last=True)
    
    def get_val(self):
        print(f"Validation DataLoader Samples: {len(self.val_df)}")
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=True)
    
    def get_test(self):
        print(f"Test DataLoader Samples: {len(self.test_df)}")
        return DataLoader(self, batch_size=self.batch_size, shuffle=False, drop_last=True)
