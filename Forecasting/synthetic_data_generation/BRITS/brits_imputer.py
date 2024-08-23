from input_process import process
from main import run
from tqdm import tqdm

import subprocess
import torch
import os
import pandas as pd
import numpy as np
import argparse
import glob

parser = argparse.ArgumentParser(description='Imputation')
parser.add_argument('--masking_type', default="mcar", type=str)
parser.add_argument('--input_data_path', default="", type=str)
parser.add_argument('--dataset', default="ETTh1", type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--model', type=str, default='brits')
parser.add_argument('--hid_size', type=int)
parser.add_argument('--impute_weight', type=float, default=0.3)
parser.add_argument('--label_weight', type=float, default=1.0)
parser.add_argument('--pvalue', type=str)
parser.add_argument('--resumetrial', type=int, default=0)

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
    date_col = df.iloc[:, 0].values
    date_col_name = df.columns[0]
    print(f"df shape = {df.shape}")
    df = df[cols]
    return {'df':df, 'date_col':date_col, 'date_col_name':date_col_name, 'cols':cols}

'''
PARAMETERS
'''
masking_type = args.masking_type
dataset = args.dataset

resumetrial = args.resumetrial
dataset_path = args.input_data_path
filename = f"v{{}}_{masking_type}_{dataset.lower()}.csv"

out_filename = f"v{{}}_{masking_type}_{dataset.lower()}_imputed_BRITS.csv"
out_path = dataset_path

json_path = "/raid/abhilash/BRITS/json/"
result_path = "/raid/abhilash/BRITS/result/"

# print(f"input_filename = {filename}\ninput dataset path = {dataset_path}\nout_filename = {out_filename}\nout_path = {out_path}")

dataset_to_hidden_len = {'ETTh1': 108, 'ETTh2': 108, 'ETTm1': 108, 'ETTm2': 108, 'weather': 128, 'electricity':216, 'traffic':216}
dataset_to_num_feat = {'ETTh1': 7, 'ETTh2': 7, 'ETTm1': 7, 'ETTm2': 7, 'weather': 21, 'electricity':321, 'traffic':862}

args.hid_size = dataset_to_hidden_len[dataset]

for trial in tqdm(range(resumetrial, 5)):
    '''
    READ DATA
    '''
    file = filename.format(trial)
    data_path = os.path.join(dataset_path, file)
    
    jsonfilename = f'json_{dataset}_v{trial}_{args.pvalue}'
    outfilename = f'imputed_{dataset}_v{trial}_{args.pvalue}'
    
    mean, std = process(data_path, dataset, jsonfilename)
    
    data_details = read_data(dataset_path, file)
    
    '''
    TRAIN BRITS MODEL
    '''
    run(args, jsonfilename, outfilename, dataset_to_num_feat[dataset])
    
    imputations_np=np.load(result_path+outfilename+'.npy')
    
    num_feat = imputations_np.shape[2]
    print(f"imputations_np shape = {imputations_np.shape}")
    
    seq_len = 52
    num_samples = imputations_np.shape[0]
    
    imputations_reshaped = imputations_np.reshape(num_samples * seq_len, num_feat)
    print(f"imputations rehspaed = {imputations_reshaped.shape}")
    
    imputations_reshaped = imputations_reshaped*std + mean
    
    # Convert to pandas DataFrame
    imputed_df = pd.DataFrame(imputations_reshaped, columns=[data_details['cols'][i] for i in range(num_feat)])

    imputed_df.insert(0, data_details['date_col_name'], data_details['date_col'][:imputed_df.shape[0]])

    '''
    SAVE
    '''
    # out_file = out_filename.format(trial)
    # imputed_df.to_csv(out_path+out_file, index=False)
    

# Use glob to find all .npy files in the './result/' directory
# if masking_type=='periodic':
#     pattern='a'
# elif masking_type=='mcar':
#     pattern='p'
# else:
#     pattern='patch'
    
# files = glob.glob(f'./result/*ETTh2*_{pattern}*')
# jsonfiles = glob.glob(f'./json/*ETTh2*_{pattern}*')

# # Iterate over the list of files and remove each one
# for file in files:
#     try:
#         os.remove(file)
#         print(f"Removed: {file}")
#     except Exception as e:
#         print(f"Error removing {file}: {e}")
        
# for file in jsonfiles:
#     try:
#         os.remove(file)
#         print(f"Removed: {file}")
#     except Exception as e:
#         print(f"Error removing {file}: {e}")
        
# '''
# clear trash
# '''
# # Command to remove all files in the Trash directory
# command = 'rm -rf ~/.local/share/Trash/*'

# try:
#     # Execute the command
#     subprocess.run(command, shell=True, check=True)
#     print("Trash cleared successfully.")
# except subprocess.CalledProcessError as e:
#     print(f"Error occurred while clearing trash: {e}")