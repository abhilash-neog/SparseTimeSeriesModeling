import pandas as pd
import numpy as np
import random
import argparse
import os
import torch
import time

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pygrinder import mcar
from pypots.data import load_specific_dataset
from pypots.imputation import SAITS
from pypots.utils.metrics import calc_mae
from thop import profile

parser = argparse.ArgumentParser(description='Imputation')
parser.add_argument('--masking_type', default="mcar", type=str)
parser.add_argument('--input_data_path', default="", type=str)
parser.add_argument('--output_data_path', default="", type=str)
parser.add_argument('--dataset', default="ETTh1", type=str)
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}')

def get_train_val_test_split(path, dataset, flag):
    
    df_raw = pd.read_csv(path)
    seq_len=336
    if dataset=='ETTh2' or dataset=='ETTh1':
        border1s = [0, 12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset=='ETTm1' or dataset=='ETTm2':
        border1s = [0, 12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
    border1 = border1s[flag]
    border2 = border2s[flag]

    # cols_data = df_raw.columns[1:]
    # date_col = df_raw.columns[:1]
    # df_data = df_raw[cols_data]
    
    data_split = df_raw[border1:border2]
    return data_split.reset_index(drop='true')

    
'''
PARAMETERS
'''
masking_type = args.masking_type
dataset = args.dataset

dataset_path = args.input_data_path
out_path = args.output_data_path
print(f"out_path = {out_path}")
filename = f"v{{}}_{masking_type}_{dataset.lower()}.csv"

out_filename = f"v{{}}_{masking_type}_{dataset.lower()}_imputed.csv"

for trial in tqdm(range(5)):
    
    t0 = time.time()
    file = filename.format(trial)
    
    train_df = get_train_val_test_split(dataset_path+file, dataset, 0)
    val_df = get_train_val_test_split(dataset_path+file, dataset, 1)
    test_df = get_train_val_test_split(dataset_path+file, dataset, 2)
    
    print(f"train_df shape = {train_df.shape}")
    print(f"val_df shape = {val_df.shape}")
    print(f"test_df shape = {test_df.shape}")
    
    train_df_interpolated = train_df.interpolate(method='spline', order=2, limit_direction='both')
    val_df_interpolated = val_df.interpolate(method='spline', order=2, limit_direction='both')
    test_df_interpolated = test_df.interpolate(method='spline', order=2, limit_direction='both')
   
    final_df=pd.concat([train_df_interpolated, val_df_interpolated, test_df_interpolated], ignore_index=True)
    print(f"final df isna() = {final_df.isna().sum()}")
    
    '''
    SAVE
    '''
    out_file = out_filename.format(trial)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    print(f"out_path = {out_path + out_file}")
    final_df.to_csv(out_path+out_file, index=False)
    
    t1 = time.time()
        
    with open("time_log.txt", "w") as f:
        print(f"time required = {t1-t0}")
        f.write(f"Time needed for version {trial} = {t1 - t0} seconds")