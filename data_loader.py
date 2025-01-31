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
    
    def __init__(self, data, seq_len, pred_len, feature_type, batch_size, dataset_type, target='OT'):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.feature_type = feature_type
        self.batch_size = batch_size
        self.dataset_type = dataset_type
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
        
        # ✅ Load only the dataset corresponding to `dataset_type`
        if self.dataset_type == 'train':
            self.df = df[:train_end]
        elif self.dataset_type == 'val':
            self.df = df[train_end:val_end]
        elif self.dataset_type == 'test':
            self.df = df[val_end:test_end]
        else:
            raise ValueError(f"Unknown dataset_type: {self.dataset_type}")
        
        # Standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(df[:train_end].values)  # ✅ Always fit on training set only
        
        # Scale the dataset
        self.df = pd.DataFrame(self.scaler.transform(self.df.values), index=self.df.index, columns=self.df.columns)

        self.n_feature = self.df.shape[-1]
        
    def __len__(self):
        dataset_length = len(self.df) - self.seq_len - self.pred_len
        print(f"[DEBUG] Dataset Type: {self.dataset_type}, Dataset Length: {len(self.df)}, Computed Length: {dataset_length}")
        return max(1, dataset_length)
    
    def __getitem__(self, idx):
        max_index = len(self.df) - self.seq_len - self.pred_len
        if idx >= max_index:  # ✅ Prevent out-of-range indexing
            raise IndexError(f"[ERROR] Index {idx} out of range for dataset of size {len(self.df)}")

        # print(f"[DEBUG] Fetching index: {idx} of {max_index}")  # ✅ Debug index range

        data = self.df.values
        inputs = data[idx:idx + self.seq_len, :]
        labels = data[idx + self.seq_len:idx + self.seq_len + self.pred_len, self.target_slice]
        
        # Convert to tensors and ensure float32 dtype
        inputs = torch.FloatTensor(inputs)
        labels = torch.FloatTensor(labels)
        
        return inputs, labels
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
