from functools import partial

import os
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import math
import argparse
import random
import datetime

eps = 1e-6

parser = argparse.ArgumentParser(description='datagenerator')

parser.add_argument('--dataset', type=str, required=True, default='weather', help='dataset type')
parser.add_argument('--in_path', type=str, default="/projects/ml4science/time_series/ts_forecasting_datasets/", help='root path of the data, code and model files')
parser.add_argument('--source_file', type=str, default='etth1.csv')
parser.add_argument('--ntrials', type=int, default=5, help='number of trials')
parser.add_argument('--pvalues', nargs='+', help='missing value probabilities')
parser.add_argument('--masking_type', type=str, default='mcar', choices=['mcar', 'periodic', 'periodic_block', 'hybrid'], help='type of masking being applied')


args = parser.parse_args()

'''
set all parameters here
'''
pvalues = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] #args.pvalues
ntrials = args.ntrials
masking_type = args.masking_type
dataset = args.dataset
source_file = args.source_file

masking_map = {
    'mcar': 'p',
    'periodic': 'a',
    'periodic_block': 'a',
    'hybrid': 'h'
} 
if dataset=='ETT':
    out_file_name='ETT'+source_file
else:
    out_file_name=dataset

out_path = f"/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/{out_file_name}/{masking_map[masking_type]}{{}}" #+ str(p)[2:]
masked_file = f"v{{}}_{masking_type}_{out_file_name.lower()}.csv"
imputed_file = f"v{{}}_{masking_type}_{out_file_name.lower()}_imputed.csv"
in_path = os.path.join(args.in_path, dataset, out_file_name+'.csv')


def apply_mcar_with_probability(dataframe, p=0.001):
    """
    Apply MCAR to a pandas DataFrame with a specific probability for missing data.
    
    Parameters:
    - dataframe: pandas DataFrame, your dataset.
    - p: float, probability of data being marked as missing.
    
    Returns:
    - modified_df: pandas DataFrame, dataset with MCAR applied.
    """
    # Ensure the probability is between 0 and 1
    assert 0 <= p <= 1, "Probability must be between 0 and 1."
    
    # Create a mask with the same shape as the dataframe with random values
    mask = np.random.rand(*dataframe.shape) < p
    print(f"percentage = {mask.sum()/(mask.shape[0]*mask.shape[1])}")
    # Create a copy of the dataframe to apply the mask
    modified_df = dataframe.copy()
    
    # Apply the mask and set the selected values to NaN
    modified_df[mask] = np.nan
    
    return modified_df

def apply_periodic_with_probability(data, pvalue):
    """
    Apply MCAR to a pandas DataFrame with a specific probability for missing data.
    
    Parameters:
    - dataframe: pandas DataFrame, your dataset.
    - p: float, probability of data being marked as missing.
    
    Returns:
    - modified_df: pandas DataFrame, dataset with MCAR applied.
    """
    num_samples = data.shape[0]
    num_feats = data.shape[1]
    t = np.arange(num_samples)
    data_masked = data.copy()
    missing_fraction = []
        
    # perform masking for every affected feature
    for feat in range(num_feats):
        phase = np.random.uniform(0, 2 * np.pi)
        periodicity = pvalue + alpha*(np.abs(np.sin(2 * np.pi * frequencies[feat] * t + phase) + 1) - 0.5)
        mask = np.random.rand(num_samples) < periodicity
        data_masked.iloc[mask, feat] = np.nan
    
    return data_masked


'''
periodic masking parameters
'''
num_feats=7 # change this based on the dataset
frequencies = []

alpha = 0.1

for p in pvalues:
    
    print(f"\n|| STARTING FOR P VALUE = {p} ||\n")
    str_p = str(p)[2:]
    '''
    make sure directory exists
    '''
    if os.path.isdir(out_path.format(str_p)):
        print("Output directory exists")
    else:
        print("Output directory does not exists")
        os.makedirs(out_path.format(str_p))
        print(f"Created a new directory: {out_path.format(str_p)}")
    
    for trial in range(ntrials):
        
        frequencies = [round(random.uniform(0.2, 0.8),1) for _ in range(num_feats)]
        
        print(f"\nTRIAL = {trial} :: Perform masking \n")
        
        df = pd.read_csv(in_path)
        columns = df.columns[1:]
        
        if masking_type=='mcar':
            df[columns] = apply_mcar_with_probability(df[columns], p=p)
        elif masking_type=='periodic':
            df[columns] = apply_periodic_with_probability(df[columns], p=p)
        elif masking_type=='periodic_block':
            df[columns] = apply_mcar_with_probability(df[columns], p=p)
        elif masking_type=='hybrid':
            df[columns] = apply_mcar_with_probability(df[columns], p=p)
        else:
            print("Wrong option")

        print(f"df isna() = {df.isna().sum()}")
        df.to_csv(os.path.join(out_path.format(str_p), masked_file.format(trial)), index=False)

        print(f"\n Perform interpolation \n")
        df_interpolated = df.interpolate(method='spline', order=2, limit_direction='both')

        print(f"df isna() = {df_interpolated.isna().sum()}")

        df_interpolated.to_csv(os.path.join(out_path.format(str_p), imputed_file.format(trial)), index=False)
        print(f"saved interpolated dataframe as {imputed_file.format(trial)}")