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

"""Train and evaluate models for time series forecasting."""

import argparse
import glob
import logging
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import TSFDataLoader
import models.tsmixer as tsmixer
import models.tsmixer_rev_in as tsmixer_rev_in

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse the arguments for experiment configuration."""
    parser = argparse.ArgumentParser(
        description='TSMixer for Time Series Forecasting'
    )

    # basic config
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--model',
        type=str,
        default='tsmixer',
        help='model name, options: [tsmixer, tsmixer_rev_in]',
    )

    # data loader
    parser.add_argument(
        '--data',
        type=str,
        default='weather',
        choices=[
            'electricity',
            'exchange_rate',
            'national_illness',
            'traffic',
            'weather',
            'ETTm1',
            'ETTm2',
            'ETTh1',
            'ETTh2',
        ],
        help='data name',
    )
    parser.add_argument(
        '--feature_type',
        type=str,
        default='M',
        choices=['S', 'M', 'MS'],
        help=(
            'forecasting task, options:[M, S, MS]; M:multivariate predict'
            ' multivariate, S:univariate predict univariate, MS:multivariate'
            ' predict univariate'
        ),
    )
    parser.add_argument(
        '--target', type=str, default='OT', help='target feature in S or MS task'
    )
    parser.add_argument(
        '--checkpoint_dir',
        type=str,
        default='./checkpoints/',
        help='location of model checkpoints',
    )
    parser.add_argument(
        '--delete_checkpoint',
        action='store_true',
        help='delete checkpoints after the experiment',
    )

    # forecasting task
    parser.add_argument(
        '--seq_len', type=int, default=336, help='input sequence length'
    )
    parser.add_argument(
        '--pred_len', type=int, default=96, help='prediction sequence length'
    )

    # model hyperparameter
    parser.add_argument(
        '--n_block',
        type=int,
        default=2,
        help='number of block for deep architecture',
    )
    parser.add_argument(
        '--ff_dim',
        type=int,
        default=2048,
        help='fully-connected feature dimension',
    )
    parser.add_argument(
        '--dropout', type=float, default=0.05, help='dropout rate'
    )
    parser.add_argument(
        '--norm_type',
        type=str,
        default='B',
        choices=['L', 'B'],
        help='LayerNorm or BatchNorm',
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        choices=['relu', 'gelu'],
        help='Activation function',
    )
    parser.add_argument(
        '--kernel_size', type=int, default=4, help='kernel size for CNN'
    )
    parser.add_argument(
        '--temporal_dim', type=int, default=16, help='temporal feature dimension'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=64, help='hidden feature dimension'
    )

    # optimization
    parser.add_argument(
        '--num_workers', type=int, default=10, help='data loader num workers'
    )
    parser.add_argument(
        '--train_epochs', type=int, default=100, help='train epochs'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='batch size of input data'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=0.0001,
        help='optimizer learning rate',
    )
    parser.add_argument(
        '--patience', type=int, default=5, help='number of epochs to early stop'
    )

    # save results
    parser.add_argument(
        '--result_path', default='result.csv', help='path to save result'
    )

    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    return args

def add_model_args(parser):
    """Add model-specific arguments."""
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--n_block', type=int, default=2)
    parser.add_argument('--ff_dim', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--norm_type', type=str, default='B', choices=['L', 'B'])
    parser.add_argument('--activation', type=str, default='relu')
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--result_path', default='result.csv')
    return parser

class EarlyStopping:
    """Early stopping to prevent overfitting."""
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train one epoch."""
    model.train()
    total_loss = 0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create experiment ID
    if 'tsmixer' or 'tsmixer_rev_in' in args.model:
        exp_id = f'{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}'
    else:
        raise ValueError(f'Unknown model type: {args.model}')

    # Ensure checkpoint directory exists
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = os.path.join(args.checkpoint_dir, f'{exp_id}_best.pth')

    # Load datasets
    data_loader = TSFDataLoader(
        args.data,
        args.batch_size,
        args.seq_len,
        args.pred_len,
        args.feature_type,
        args.target,
    )
    train_loader = data_loader.get_train()
    val_loader = data_loader.get_val()
    test_loader = data_loader.get_test()

    # Build model
    input_shape = (args.seq_len, data_loader.n_feature)
    if args.model == 'tsmixer':
        model = tsmixer.build_model(
            input_shape=input_shape,
            pred_len=args.pred_len,
            norm_type=args.norm_type,
            activation=args.activation,
            n_block=args.n_block,
            dropout=args.dropout,
            ff_dim=args.ff_dim,
            target_slice=data_loader.target_slice,
        )
    elif args.model == 'tsmixer_rev_in':
        model = tsmixer_rev_in.build_model(
            input_shape=input_shape,
            pred_len=args.pred_len,
            norm_type=args.norm_type,
            activation=args.activation,
            n_block=args.n_block,
            dropout=args.dropout,
            ff_dim=args.ff_dim,
            target_slice=data_loader.target_slice,
        )

    model = model.to(device)
    
    # Setup training
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.1)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training loop
    logger.info("Starting training...")
    start_training_time = time.time()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(args.train_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        logger.info(f'Epoch {epoch+1}/{args.train_epochs}')
        logger.info(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model, checkpoint_path)
        
        if early_stopping.early_stop:
            logger.info("Early stopping triggered")
            break

    training_time = time.time() - start_training_time
    logger.info(f'Training finished in {training_time:.2f} seconds')

    # Load best model and evaluate
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    test_loss = validate(model, test_loader, criterion, device)
    test_mae = validate(model, test_loader, nn.L1Loss(), device)
    
    # Save results
    best_epoch = np.argmin(val_losses)
    results = {
        'data': [args.data],
        'model': [args.model],
        'seq_len': [args.seq_len],
        'pred_len': [args.pred_len],
        'lr': [args.learning_rate],
        'mse': [test_loss],
        'mae': [test_mae],
        'val_mse': [val_losses[best_epoch]],
        'val_mae': [validate(model, val_loader, nn.L1Loss(), device)],
        'train_mse': [train_losses[best_epoch]],
        'train_mae': [validate(model, train_loader, nn.L1Loss(), device)],
        'training_time': training_time,
        'norm_type': args.norm_type,
        'activation': args.activation,
        'n_block': args.n_block,
        'dropout': args.dropout,
    }
    
    if 'tsmixer' in args.model:
        results['ff_dim'] = args.ff_dim

    df = pd.DataFrame(results)
    if os.path.exists(args.result_path):
        df.to_csv(args.result_path, mode='a', index=False, header=False)
    else:
        df.to_csv(args.result_path, mode='w', index=False, header=True)

    # Cleanup
    if args.delete_checkpoint:
        for f in glob.glob(checkpoint_path + '*'):
            os.remove(f)

if __name__ == '__main__':
    main()