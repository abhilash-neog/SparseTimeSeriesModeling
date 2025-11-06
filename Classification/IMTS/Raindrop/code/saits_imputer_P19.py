"""
SAITS Training Script with Metrics Tracking for P19 Dataset
Based on original SAITS implementation with custom training loop
"""

import pandas as pd
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import time
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import sys

# Add paths to import SAITS model and Raindrop utilities
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Fix SAITS imports - saits.py expects "modeling.layers" and "modeling.utils"
# Create a modeling module structure
import types
modeling_module = types.ModuleType('modeling')
sys.modules['modeling'] = modeling_module

# Import the actual modules and attach to modeling
from layers import *
from utils import masked_mae_cal
modeling_module.layers = sys.modules['layers']
modeling_module.utils = types.ModuleType('modeling.utils')
modeling_module.utils.masked_mae_cal = masked_mae_cal

# Now import SAITS
from saits import SAITS
from utils_rd import get_data_split, getStats, getStats_static, tensorize_normalize

parser = argparse.ArgumentParser(description='SAITS Training with Metrics')
parser.add_argument('--dataset', type=str, default='P19', choices=['P12', 'P19', 'eICU', 'PAM'])
parser.add_argument('--splittype', type=str, default='random', choices=['random', 'age', 'gender'])
parser.add_argument('--reverse', default=False, help='if True, use female/older for training')
parser.add_argument('--feature_removal_level', type=str, default='no_removal', 
                    choices=['no_removal', 'set', 'sample'])
parser.add_argument('--withmissingratio', default=False, help='if True, missing ratio ranges from 0 to 0.5')
parser.add_argument('--predictive_label', type=str, default='mortality', choices=['mortality', 'LoS'])
parser.add_argument('--device', type=int, default=4, help='CUDA device index')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
parser.add_argument('--n_splits', type=int, default=5, help='Number of data splits for cross-validation')

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# SAITS Model Parameters
n_groups = 1  # Number of groups (can be adjusted)
n_group_inner_layers = 2  # Number of inner layers per group
d_model = 256  # Model dimension
d_inner = 128  # Feed-forward dimension
n_head = 4  # Number of attention heads
d_k = 64  # Key/Query dimension
d_v = 64  # Value dimension
dropout = 0.1
param_sharing_strategy = 'between_group'  # Parameter sharing strategy

# Training parameters
learning_rate = 0.0001
num_epochs = args.epochs
batch_size = args.batch_size
n_splits = args.n_splits

# Initialize metrics arrays
avg_epoch_time_arr = np.zeros((n_splits, 1))
avg_inference_time_arr = np.zeros((n_splits, 1))
peak_memory_gb_arr = np.zeros((n_splits, 1))

# Data paths
base_path = f'../../data/{args.dataset}'
split = args.splittype
reverse = args.reverse
feature_removal_level = args.feature_removal_level
missing_ratio = 0 if not args.withmissingratio else 0.1  # Start with 0.1 if withmissingratio is True

for k in range(n_splits):
    split_idx = k + 1
    print(f'\n{"="*60}')
    print(f'Split {split_idx}/{n_splits}')
    print(f'{"="*60}\n')
    
    # Load data split
    if split == 'random':
        split_path = f'/splits/phy19_split{split_idx}_new.npy' if args.dataset == 'P19' else f'/splits/phy12_split{split_idx}.npy'
    else:
        split_path = ''  # Will be handled by get_data_split
    
    Ptrain, Pval, Ptest, ytrain, yval, ytest = get_data_split(
        base_path, split_path, split_type=split, reverse=reverse,
        baseline=False, dataset=args.dataset, predictive_label=args.predictive_label
    )
    
    print(f'Train: {len(Ptrain)}, Val: {len(Pval)}, Test: {len(Ptest)}')
    
    # Convert to tensors and normalize
    if args.dataset in ['P12', 'P19', 'eICU']:
        T, F = Ptrain[0]['arr'].shape
        D = len(Ptrain[0]['extended_static'])
        
        Ptrain_tensor = np.zeros((len(Ptrain), T, F))
        Ptrain_static_tensor = np.zeros((len(Ptrain), D))
        for i in range(len(Ptrain)):
            Ptrain_tensor[i] = Ptrain[i]['arr']
            Ptrain_static_tensor[i] = Ptrain[i]['extended_static']
        
        mf, stdf = getStats(Ptrain_tensor)
        ms, ss = getStats_static(Ptrain_static_tensor, dataset=args.dataset)
        
        Ptrain_tensor, _, Ptrain_time_tensor, _ = tensorize_normalize(Ptrain, ytrain, mf, stdf, ms, ss)
        Pval_tensor, _, Pval_time_tensor, _ = tensorize_normalize(Pval, yval, mf, stdf, ms, ss)
        Ptest_tensor, _, Ptest_time_tensor, _ = tensorize_normalize(Ptest, ytest, mf, stdf, ms, ss)
        
        # tensorize_normalize returns torch tensors in shape (N, T, 2*F) where mask_normalize concatenates [values, mask]
        # Raindrop then permutes to (T, N, 2*F) - we need (N, T, F) for SAITS
        print(f'After tensorize_normalize - Train shape: {Ptrain_tensor.shape}')
        
        # Check if already permuted (Raindrop style) or not
        # If first dimension equals T, it's likely (T, N, 2*F), otherwise (N, T, 2*F)
        if len(Ptrain_tensor.shape) == 3:
            if Ptrain_tensor.shape[0] == T and Ptrain_tensor.shape[1] != T:
                # Shape is (T, N, 2*F), permute to (N, T, 2*F)
                Ptrain_tensor = Ptrain_tensor.permute(1, 0, 2)
                Pval_tensor = Pval_tensor.permute(1, 0, 2)
                Ptest_tensor = Ptest_tensor.permute(1, 0, 2)
        
        # mask_normalize concatenates [normalized_values, mask], so shape is (N, T, 2*F)
        # Split into values and mask for SAITS
        actual_feature_dim = Ptrain_tensor.shape[2] // 2  # Original F before mask concatenation
        
        # Extract values and mask separately
        # Values are in first half, mask is in second half
        Ptrain_values = Ptrain_tensor[:, :, :actual_feature_dim]  # Shape: (N, T, F)
        Ptrain_mask_from_data = Ptrain_tensor[:, :, actual_feature_dim:]  # Shape: (N, T, F) - mask from normalization
        
        Pval_values = Pval_tensor[:, :, :actual_feature_dim]
        Pval_mask_from_data = Pval_tensor[:, :, actual_feature_dim:]
        
        Ptest_values = Ptest_tensor[:, :, :actual_feature_dim]
        Ptest_mask_from_data = Ptest_tensor[:, :, actual_feature_dim:]
        
        # Use the mask from the data (1 = observed, 0 = missing)
        # Convert to format SAITS expects: (batch, time, features)
        d_time = Ptrain_values.shape[1]  # Sequence length (T)
        d_feature = Ptrain_values.shape[2]  # Number of features (F)
        
        print(f'Final data shape - Train values: {Ptrain_values.shape}, mask: {Ptrain_mask_from_data.shape}')
        print(f'SAITS expects: d_time={d_time}, d_feature={d_feature}')
    else:
        raise ValueError(f"Dataset {args.dataset} not yet supported in this script")
    
    # Use the masks extracted from the normalized data
    # mask_normalize creates masks where 1 = observed, 0 = missing
    train_mask = Ptrain_mask_from_data  # Already extracted above
    val_mask = Pval_mask_from_data
    test_mask = Ptest_mask_from_data
    
    # SAITS expects separate X and mask (not concatenated when input_with_mask=True)
    # Create DataLoaders with (X, mask) tuples
    train_dataset = TensorDataset(Ptrain_values, train_mask)
    val_dataset = TensorDataset(Pval_values, val_mask)
    test_dataset = TensorDataset(Ptest_values, test_mask)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = SAITS(
        n_groups=n_groups,
        n_group_inner_layers=n_group_inner_layers,
        d_time=d_time,
        d_feature=d_feature,
        d_model=d_model,
        d_inner=d_inner,
        n_head=n_head,
        d_k=d_k,
        d_v=d_v,
        dropout=dropout,
        diagonal_attention_mask=False,  # Whether to use diagonal attention mask
        input_with_mask=True,  # We concatenate mask with values
        param_sharing_strategy=param_sharing_strategy,
        MIT=False,  # Masked Imputation Task (set to False for now)
        device=device
    ).to(device)
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize metrics tracking
    epoch_times = []
    
    # Reset peak GPU memory stats
    if torch.cuda.is_available():
        device_index = device.index if device.index is not None else 0
        _ = torch.zeros(1).to(device)  # Initialize device
        torch.cuda.reset_peak_memory_stats(device_index)
    
    print(f'\nStarting training for {num_epochs} epochs...\n')
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_epoch = 0
        train_batches = 0
        
        # Training loop
        for batch_idx, (batch_X, batch_mask) in enumerate(train_loader):
            batch_X = batch_X.to(device)  # Shape: (B, T, F)
            batch_mask = batch_mask.to(device)  # Shape: (B, T, F)
            
            # Create input dictionary for SAITS
            # For imputation task, we need X and missing_mask
            # missing_mask: 1 = observed, 0 = missing (opposite of what SAITS might expect)
            # SAITS uses: indicating_mask (1 = missing, 0 = observed) for reconstruction loss
            # So we need to invert the mask
            indicating_mask = 1 - batch_mask  # 1 = missing, 0 = observed
            
            inputs = {
                "X": batch_X,
                "missing_mask": batch_mask,
                "indicating_mask": indicating_mask
            }
            
            # Forward pass
            outputs = model(inputs, stage="train")
            loss = outputs["reconstruction_loss"]
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_batches += 1
        
        # Ensure GPU ops complete before ending timer
        if torch.cuda.is_available() and device.type == 'cuda':
            torch.cuda.synchronize(device)
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        avg_train_loss = train_loss_epoch / train_batches if train_batches > 0 else 0
        
        # Track epoch time (starting from epoch 2)
        if epoch >= 1:
            epoch_times.append(epoch_time)
            if len(epoch_times) >= 6:
                avg_epoch_time = sum(epoch_times[-6:]) / 6
                print(f'Epoch {epoch+1}: Time = {epoch_time:.3f}s | Avg (last 6): {avg_epoch_time:.3f}s | Loss = {avg_train_loss:.4f}')
        
        # Check peak GPU memory after epoch 3
        if epoch == 2 and torch.cuda.is_available():
            device_index = device.index if device.index is not None else 0
            peak_memory_bytes = torch.cuda.max_memory_allocated(device_index)
            peak_memory_gb = peak_memory_bytes / (1024 ** 3)
            peak_memory_gb_arr[k, 0] = peak_memory_gb
            print(f'Peak GPU Memory (after epoch {epoch+1}): {peak_memory_gb:.3f} GB')
        
        # Validation
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            model.eval()
            val_loss_epoch = 0
            val_batches = 0
            
            with torch.no_grad():
                for batch_X, batch_mask in val_loader:
                    batch_X = batch_X.to(device)
                    batch_mask = batch_mask.to(device)
                    
                    # Invert mask for indicating_mask (1 = missing, 0 = observed)
                    indicating_mask = 1 - batch_mask
                    
                    # For validation, SAITS expects X_holdout (same as X for reconstruction evaluation)
                    inputs = {
                        "X": batch_X,
                        "missing_mask": batch_mask,
                        "indicating_mask": indicating_mask,
                        "X_holdout": batch_X  # For validation loss calculation
                    }
                    
                    outputs = model(inputs, stage="val")
                    val_loss = outputs["reconstruction_loss"]
                    val_loss_epoch += val_loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss_epoch / val_batches if val_batches > 0 else 0
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), f'saits_best_model_split{split_idx}.pt')
            
            print(f'Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Calculate average training time per epoch
    if len(epoch_times) >= 6:
        avg_training_time_per_epoch = sum(epoch_times[-6:]) / 6
        avg_epoch_time_arr[k, 0] = avg_training_time_per_epoch
        print('=' * 60)
        print(f'Average Training Time per Epoch (epochs 2-7): {avg_training_time_per_epoch:.3f} seconds')
        print('=' * 60)
    else:
        avg_training_time_per_epoch = sum(epoch_times) / len(epoch_times) if epoch_times else 0
        avg_epoch_time_arr[k, 0] = avg_training_time_per_epoch
        print('=' * 60)
        print(f'Average Training Time per Epoch (available epochs): {avg_training_time_per_epoch:.3f} seconds')
        print('=' * 60)
    
    # Inference and average inference time measurement
    print('\nPerforming inference...')
    model.load_state_dict(torch.load(f'saits_best_model_split{split_idx}.pt'))
    model.eval()
    
    test_dataset_size = len(test_loader.dataset)
    # Use wall-clock timing with perf_counter for overall inference time
    inference_start = time.perf_counter()
    with torch.no_grad():
        for batch_X, batch_mask in test_loader:
            batch_X = batch_X.to(device)
            batch_mask = batch_mask.to(device)
            
            # Invert mask for indicating_mask (1 = missing, 0 = observed)
            indicating_mask = 1 - batch_mask
            
            inputs = {
                "X": batch_X,
                "missing_mask": batch_mask,
                "indicating_mask": indicating_mask
            }
            
            outputs = model(inputs, stage="test")
    # Synchronize before stopping the timer to include all GPU work
    if torch.cuda.is_available() and device.type == 'cuda':
        torch.cuda.synchronize(device)
    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start
    avg_inference_time_arr[k, 0] = inference_time
    
    print('=' * 60)
    print(f'Average Inference Time: {inference_time:.3f} seconds')
    print(f'Test Dataset Size: {test_dataset_size} samples')
    print('=' * 60)

# Print final aggregated metrics
print('\n' + '=' * 60)
print('FINAL METRICS (across all splits):')
print('=' * 60)
mean_epoch_time, std_epoch_time = np.mean(avg_epoch_time_arr), np.std(avg_epoch_time_arr)
mean_inference_time, std_inference_time = np.mean(avg_inference_time_arr), np.std(avg_inference_time_arr)
mean_memory, std_memory = np.mean(peak_memory_gb_arr), np.std(peak_memory_gb_arr)

print(f'Avg Training Time per Epoch = {mean_epoch_time:.3f} +/- {std_epoch_time:.3f} seconds')
print(f'Avg Inference Time          = {mean_inference_time:.3f} +/- {std_inference_time:.3f} seconds')
print(f'Peak GPU Memory             = {mean_memory:.3f} +/- {std_memory:.3f} GB')
print('=' * 60)

