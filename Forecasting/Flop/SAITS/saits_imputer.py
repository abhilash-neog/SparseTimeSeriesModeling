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
from ptflops import get_model_complexity_info

parser = argparse.ArgumentParser(description='Imputation')
parser.add_argument('--masking_type', default="mcar", type=str)
parser.add_argument('--input_data_path', default="", type=str)
parser.add_argument('--dataset', default="ETTh1", type=str)
parser.add_argument('--device', default=0, type=int)

args = parser.parse_args()

device = torch.device(f'cuda:{args.device}')

def read_data(dataset, filename):
    
    filepath = dataset_path+filename
    print(f"Filepath = {filepath}")
    if not os.path.exists(filepath):
        print(f"filepath does not exist: Aborting")
        exit(0)
    df = pd.read_csv(filepath)
    cols = df.columns[1:]
    date_col = df.iloc[:, 0]
    date_col_name = df.columns[:1]
    df = df[cols]
    return {'df':df, 'date_col':date_col, 'date_col_name':date_col_name, 'cols':cols}

'''
PARAMETERS
'''
masking_type = args.masking_type
dataset = args.dataset

dataset_path = args.input_data_path
filename = f"v{{}}_{masking_type}_{dataset.lower()}.csv"

out_filename = f"v{{}}_{masking_type}_{dataset.lower()}_imputed_SAITS.csv"
out_path = "./outputs"

# print(f"input_filename = {filename}\ninput dataset path = {dataset_path}\nout_filename = {out_filename}\nout_path = {out_path}")

dataset_to_batch_len = {'ETTh1': 52, 'ETTh2': 336, 'ETTm1': 52, 'ETTm2': 52, 'weather': 56, 'electricity': 48, 'traffic': 51}
batch_size = dataset_to_batch_len[dataset]

for trial in tqdm(range(5)):
    '''
    READ DATA
    '''
    t0 = time.time()
    file = filename.format(trial)
    data_details = read_data(dataset_path, file)
    
    df = data_details['df']
    
    '''
    DATA NORMALIZE AND PROCESSING
    '''
    ss = StandardScaler()
    X_ = df.to_numpy()
    X = ss.fit_transform(X_)
    
    num_batches = X.shape[0]//batch_size

    X = X[:num_batches*batch_size].reshape(-1, batch_size, X.shape[1])
    Xtrain = {"X": X}
    
    '''
    MODEL
    '''
    saits = SAITS(n_steps=batch_size, 
                  n_features=X.shape[-1], 
                  n_layers=2, 
                  d_model=256, 
                  d_ffn=128, 
                  n_heads=4, 
                  d_k=64, 
                  d_v=64, 
                  dropout=0.1, 
                  epochs=100,
                  device=device,
                  saving_path="./")
    # macs, params = profile(saits, inputs=(Xtrain, ))
    # print(f"macs = {macs} \nparams = {params}")
    flops, params = get_model_complexity_info(
                    model_wrapper,
                    input_res=input_shape,  # Pass the input shape as a tuple
                    as_strings=True,
                    print_per_layer_stat=False
                )
    print(f'FLOPs for this model: {flops}')
    print(f'Parameters for this model: {params}')
    exit(0)
    
    '''
    TRAINING
    '''
    saits.fit(Xtrain)

   
    
    '''
    INFERENCE
    '''
    imputation = saits.impute(Xtrain)

    imputation = imputation.reshape(-1, imputation.shape[-1])

    imputed_np = ss.inverse_transform(imputation)

    imputed_df = pd.DataFrame(imputed_np, columns=data_details['cols'])

    imputed_df[data_details['date_col_name'][0]] = data_details['date_col']
    imputed_df = imputed_df[data_details['date_col_name'].append(data_details['cols'])]

    print(imputed_df.isna().sum())

    '''
    SAVE
    '''
    out_file = out_filename.format(trial)
    imputed_df.to_csv(out_path+out_file, index=False)
    t1 = time.time()
    
    print(f"Time per trial = {(t1-t0)}")