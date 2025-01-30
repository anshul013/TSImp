import argparse
import glob
import logging
import os
import time

from data_loader import TSFDataLoader
import models
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

def parse_args():
    """Parse the arguments for experiment configuration."""
    parser = argparse.ArgumentParser(description='TSMixer for Time Series Forecasting')
    
    # Basic config
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--model', type=str, default='tsmixer',
                        help='model name, options: [tsmixer, tsmixer_rev_in]')
    
    # Data loader
    parser.add_argument('--data', type=str, default='weather',
                        choices=['electricity', 'exchange_rate', 'national_illness', 'traffic', 'weather',
                                 'ETTm1', 'ETTm2', 'ETTh1', 'ETTh2'], help='data name')
    parser.add_argument('--feature_type', type=str, default='M', choices=['S', 'M', 'MS'],
                        help='forecasting task type')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints/', help='checkpoint directory')
    parser.add_argument('--delete_checkpoint', action='store_true', help='delete checkpoints after the experiment')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=336, help='input sequence length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    
    # Model hyperparameters
    parser.add_argument('--n_block', type=int, default=2, help='number of blocks')
    parser.add_argument('--ff_dim', type=int, default=2048, help='fully-connected feature dimension')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout rate')
    parser.add_argument('--norm_type', type=str, default='B', choices=['L', 'B'], help='Normalization type')
    parser.add_argument('--activation', type=str, default='relu', choices=['relu', 'gelu'], help='Activation function')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='number of workers for DataLoader')
    parser.add_argument('--train_epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    
    # Save results
    parser.add_argument('--result_path', default='result.csv', help='path to save result')
    
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    return args

def main():
    args = parse_args()
    exp_id = f"{args.data}_{args.feature_type}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_lr{args.learning_rate}" \
             f"_nt{args.norm_type}_{args.activation}_nb{args.n_block}_dp{args.dropout}_fd{args.ff_dim}"
    
    # Load datasets
    data_loader = TSFDataLoader(args.data, args.seq_len, args.pred_len, args.feature_type, args.target)
    train_data = DataLoader(data_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    # Model selection
    model_class = {
    "tsmixer": models.tsmixer.TSMixer,
    "tsmixer_rev_in": models.tsmixer_rev_in.TSMixerRevNorm
    }.get(args.model, None)
    if model_class is None:
        raise ValueError(f'Unknown model type: {args.model}')
    
    model = model_class(input_shape=(args.seq_len, data_loader.n_feature), pred_len=args.pred_len,
                         norm_type=args.norm_type, activation=args.activation, dropout=args.dropout,
                         n_block=args.n_block, ff_dim=args.ff_dim, target_slice=data_loader.target_slice)
    
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    loss_fn = nn.MSELoss()
    
    best_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    for epoch in range(args.train_epochs):
        model.train()
        epoch_loss = 0
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        epoch_loss /= len(train_data)
        print(f'Epoch {epoch + 1}/{args.train_epochs}, Loss: {epoch_loss}')
        
        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, f'{exp_id}_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print('Early stopping triggered.')
                break
    
    # Evaluate best model
    model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, f'{exp_id}_best.pth')))
    model.eval()
    
    # Save results
    results = {
        'data': args.data, 'model': args.model, 'seq_len': args.seq_len, 'pred_len': args.pred_len,
        'lr': args.learning_rate, 'best_loss': best_loss, 'train_epochs': epoch + 1
    }
    df = pd.DataFrame([results])
    df.to_csv(args.result_path, mode='a', index=False, header=not os.path.exists(args.result_path))

if __name__ == '__main__':
    main()
