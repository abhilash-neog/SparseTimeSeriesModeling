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
        border1s = [0, 12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24 - seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
    elif dataset=='ETTm1' or dataset=='ETTm2':
        border1s = [0, 12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train, len(df_raw) - num_test]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        
    border1 = border1s[flag]
    border2 = border2s[flag]

    cols_data = df_raw.columns[1:]
    date_col = df_raw.columns[:1]
    df_data = df_raw[cols_data]
    
    data_split = df_data[border1:border2]
    return data_split.reset_index(drop='true'), df_raw

def perform_inference(saits, X, ss, df, rem=0):
    
    imputation = saits.impute(X)

    if rem!=0:
        imputation_a = imputation[:-1, :, :].reshape(-1, imputation.shape[-1])
        imputation_b = imputation[-1, -rem:, :].reshape(-1, imputation.shape[-1])
        imputation = np.concatenate((imputation_a, imputation_b), axis=0)
    else:
        imputation = imputation.reshape(-1, imputation.shape[-1])
        
    imputed_np = ss.inverse_transform(imputation)

    imputed_df = pd.DataFrame(imputed_np, columns=df.columns[1:])
    
    date_col = df.columns[:1]
    imputed_df[date_col] = df[date_col]
    imputed_df = imputed_df[df.columns]
    
    return imputed_df
    
def generate_overlapping_windows(data, ser_len, stride):
    num_windows = (len(data) - ser_len) // stride + 1  # Calculate the number of windows
    windows = [data[i : i + ser_len] for i in range(0, num_windows * stride, stride)]
    return np.array(windows)

# def hybrid_sampling(data, ser_len):
#     S = data.shape[0]  # Total number of samples
#     num_features = data.shape[1]
#     R = S % ser_len  # Remaining samples after disjoint windows

#     # Endpoint where disjoint sampling ends
#     X = S - R

#     # Disjoint sampling up to point X
#     disjoint_samples = [data[i:i+ser_len] for i in range(0, X, ser_len)]
    
#     # Overlapping sampling for the remaining part, stride of 1
#     overlapping_samples = []
#     start_overlapping = X - (ser_len - R)  # Start point for overlapping samples
    
#     if R > 0:
#         for i in range(start_overlapping, S - window_size + 1):
#             overlapping_samples.append(data[i:i+window_size])

#     # # Combine both disjoint and overlapping samples into a single array
#     # combined_samples = np.array(disjoint_samples + overlapping_samples)
    
#     # Ensure the output shape is (num_samples, window_size, num_features)
#     # combined_samples = combined_samples.reshape(-1, window_size, num_features)
    
#     return disjoint_samples, overlapping_samples, R
    
'''
PARAMETERS
'''
masking_type = args.masking_type
dataset = args.dataset

dataset_path = args.input_data_path
out_path = args.output_data_path
print(f"out_path = {out_path}")
filename = f"v{{}}_{masking_type}_{dataset.lower()}.csv"

out_filename = f"v{{}}_{masking_type}_{dataset.lower()}_imputed_SAITS.csv"

dataset_to_ser_len = {'ETTh1': [48, 48, 48], 
                        'ETTh2': [48, 48, 48], 
                        'ETTm1': [48, 48, 48], 
                        'ETTm2': [48, 48, 48], 
                        'weather': [48, 48, 48], 
                        'electricity': [48, 48, 48], 
                        'traffic': [48, 48, 48]}

mapper={0:'train',
       1:'val',
       2:'test'}

ser_len=48

for trial in tqdm(range(5)):
    '''
    READ DATA
    '''
    t0 = time.time()
    file = filename.format(trial)
    train_df, df_raw = get_train_val_test_split(dataset_path+file, dataset, 0)
    val_df, _ = get_train_val_test_split(dataset_path+file, dataset, 1)
    test_df, _ = get_train_val_test_split(dataset_path+file, dataset, 2)
    
    print(f"train_df shape = {train_df.shape}")
    print(f"val_df shape = {val_df.shape}")
    print(f"test_df shape = {test_df.shape}")
    
    '''
    DATA NORMALIZE AND PROCESSING
    '''
    ss = StandardScaler()
    train_X = train_df.to_numpy()
    val_X = val_df.to_numpy()
    test_X = test_df.to_numpy()
    
    ss.fit(train_X)
    
    train_X=ss.transform(train_X)
    val_X=ss.transform(val_X)
    test_X=ss.transform(test_X)
    
    '''
    We generate overlapping windows as training samples for SAITS
    But, during inference, there is no overlap, all the samples are disjoint
    '''
    stride=1
    Xtrain = generate_overlapping_windows(train_X, ser_len, stride)
    print(f"train_X = {Xtrain.shape}")
    
    Xtrain = {"X": Xtrain}
    
    '''
    prepare data for imputation
    '''
    train_s = train_X.shape[0]
    num_batches_train = train_s//ser_len
    rem_train = train_s%ser_len
    
    train_X_for_imp = train_X[:num_batches_train*ser_len]
    if rem_train!=0:
        train_X_for_imp = np.concatenate((train_X_for_imp, train_X[-ser_len:]), axis=0).reshape(-1, ser_len, train_X.shape[1])
    else:
        train_X_for_imp = train_X_for_imp.reshape(-1, ser_len, train_X.shape[1])
        
    train_X_for_imp = {"X": train_X_for_imp}
    
    val_s = val_X.shape[0]
    num_batches_val = val_s//ser_len
    rem_val = val_s%ser_len
    
    val_X_for_imp = val_X[:num_batches_val*ser_len]
    if rem_val!=0:
        val_X_for_imp = np.concatenate((val_X_for_imp, val_X[-ser_len:]), axis=0).reshape(-1, ser_len, val_X.shape[1])
    else:
        val_X_for_imp = val_X_for_imp.reshape(-1, ser_len, val_X.shape[1])
        
    val_X_for_imp = {"X": val_X_for_imp}
                                    
    test_s = test_X.shape[0]
    num_batches_test = test_s//ser_len
    rem_test = test_s%ser_len
    
    test_X_for_imp = test_X[:num_batches_test*ser_len]
    if rem_test!=0:
        test_X_for_imp = np.concatenate((test_X_for_imp, test_X[-ser_len:]), axis=0).reshape(-1, ser_len, test_X.shape[1])
    else:
        test_X_for_imp = test_X_for_imp.reshape(-1, ser_len, test_X.shape[1])
        
    test_X_for_imp = {"X": test_X_for_imp}
    
    '''
    MODEL
    '''
    saits = SAITS(n_steps=ser_len, 
                  n_features=train_X.shape[-1], 
                  n_layers=2, 
                  d_model=256, 
                  d_ffn=128, 
                  n_heads=4, 
                  d_k=64, 
                  d_v=64, 
                  dropout=0.1, 
                  epochs=100,
                  batch_size=16,
                  device=device,
                  saving_path="/raid/abhilash/saits_logs/")
    
    '''
    TRAINING
    '''
    saits.fit(Xtrain)
   
    '''
    perform inference
    '''
    imp_train=perform_inference(saits, train_X_for_imp, ss, df_raw, rem_train)
    imp_val=perform_inference(saits, val_X_for_imp, ss, df_raw, rem_val)
    imp_test=perform_inference(saits, test_X_for_imp, ss, df_raw, rem_test)
    
    final_df=pd.concat([imp_train, imp_val, imp_test], ignore_index=True)
    '''
    SAVE
    '''
    out_file = out_filename.format(trial)
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    final_df.to_csv(out_path+out_file, index=False)
    
    t1 = time.time()
        
    with open("time_log.txt", "w") as f:
        print(f"time required = {t1-t0}")
        f.write(f"Time needed for version {trial} = {t1 - t0} seconds")