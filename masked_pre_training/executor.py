import argparse
import torch
import random
import numpy as np
import os
import json
import pandas as pd
import math
import datetime

from model_mae import MaskedAutoencoder
from utils import Utils
from functools import partial

'''
----------------
modification log
----------------
Date: Nov 24, 2023
Tag: initial_implementation
Description: Initially we are going with the assumptions that pre-train and fine-tune are tasks that are run separately
Author: Abhilash

'''

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='LakeMAE')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', choices=['pretrain', 'finetune', 'zeroshot'], help='task name, options:[pretrain, finetune]')
# parser.add_argument('--zero_shot', type=str, default='True', help='zero-shot evaluation perform')

# data loader
parser.add_argument('--root_path', type=str, default='./', help='root path of the data, code and model files')
parser.add_argument('--data_path', type=str, default='./data', help='path to the data dir')
parser.add_argument('--source_filename', type=str, default='TransferLearningData.csv', help='name of the data file')
parser.add_argument('--non_null_ratio', type=float, default=0.20, help='non null ratio required for considering one window')
parser.add_argument('--config_base', type=str, default='./config', help='path to the config file')
parser.add_argument('--config_name', type=str, default='config.json', help='config file')

# model loader
parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints_dir', type=str, default='./pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--pretrain_ckpt_name', type=str, default='ckpt_latest.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest.pth', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--pretrain_run_name', type=str, default='mae_pretraining_run', help='run name in wandb')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')

# pretraining task
parser.add_argument('--window', type=int, default=35, help='window for pretraining and fine-tuning')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
parser.add_argument('--num_samples', type=int, default=10, help='number of sample regions to plot for each feature during pretraining')
parser.add_argument('--num_windows', type=int, default=25, help='number of windows to generate merged plots')
parser.add_argument('--feature_wise_rmse', type=str, default='True', help='whether to plot feature-wise rmse')

# finetuning task
parser.add_argument('--lookback_window', type=int, default=14, help='past sequence length')
parser.add_argument('--freeze_encoder', type=str, default='True', help='whether to freeze encoder or not')
parser.add_argument('--n2one_ft', type=str, default='False', help='whether we want to perform n2one finetuning or not')

# model define
parser.add_argument('--encoder_embed_dim', type=int, default=64, help='encoder embedding dimension in the feature space')
parser.add_argument('--encoder_depth', type=int, default=2, help='number of encoder blocks')
parser.add_argument('--encoder_num_heads', type=int, default=4, help='number of encoder multi-attention heads')
parser.add_argument('--decoder_depth', type=int, default=2, help='number of decoder blocks')
parser.add_argument('--decoder_num_heads', type=int, default=4, help='number of decoder multi-attention heads')
parser.add_argument('--decoder_embed_dim', type=int, default=32, help='decoder embedding dimension in the feature space')
parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio for vision transformer')

# training 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--accum_iter', type=int, default=1, help='accumulation iteration for gradient accumulation')
parser.add_argument('--min_lr', type=float, default=1e-5, help='min learning rate')
parser.add_argument('--weight_decay', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--blr', type=float, default=1e-4, help='base learning rate')
parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs for learning rate')
parser.add_argument('--max_epochs', type=int, default=40)
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')


# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')

# weights and biases
parser.add_argument('--project_name', type=str, default='2dmasking', help='project name in wandb')
parser.add_argument('--run_name', type=str, required=True, default='mae_pretraining_run', help='run name in wandb')
parser.add_argument('--save_code', type=str, default='True', help='whether to log code in wandb or not')

args = parser.parse_args()

'''
update run name
'''
base_run_name = args.run_name
args.run_name = "{}_{}_{}".format(args.run_name, str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))


'''
read config file
'''
config_path = os.path.join(args.config_base, args.config_name)
with open(config_path, 'r') as json_file:
    config = json.load(json_file)

'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'

'''
read the file
'''
filepath = os.path.join(args.root_path, args.data_path, args.source_filename)
df = pd.read_csv(filepath)

# define the feature,date and flag columns
cols_to_exclude_from_features = ["Lake","Site","Depth_m","DataType","ModelRunType"]
features_col = df.columns.difference(cols_to_exclude_from_features)
features_col = [feat for feat in features_col if not 'flag' in feat.lower() and feat!='DateTime']
date_col = ['DateTime']

df.DateTime = df.DateTime.astype('datetime64[ns]')
train_df = df.copy(deep=True)

flag_cols = [col for col in train_df.columns if col.startswith('Flag')]
num_features = len(features_col)

'''
initialize utils object
'''
utils = Utils(num_features=num_features,
              inp_cols=features_col, 
              date_col=date_col, 
              window=args.window, 
              flag_cols=flag_cols,
              non_null_ratio=args.non_null_ratio,
              device=args.device,
              task_name=args.task_name,
              n2one=args.n2one_ft,
              lookback_window=args.lookback_window, 
              stride=1)

'''
create windowed dataset or load one 
'''
data_path = os.path.join(args.root_path, args.data_path)
utils.windowed_dataset_utils(df, data_path)


'''
create train and val set
'''
train_X, val_X, train_lake_names, val_lake_names = utils.split_data(data_path, config)

'''
normalize the data
'''
train_X = torch.from_numpy(train_X).type(torch.Tensor)
val_X = torch.from_numpy(val_X).type(torch.Tensor)

# print(f"before normalization = {train_X}")
train_X = utils.normalize_tensor(train_X, use_stat=False)
# print(f"after normalization = {train_X}")
val_X = utils.normalize_tensor(val_X, use_stat=True)

# train_X = train_X.transpose(1, 2)
# val_X = val_X.transpose(1, 2)

'''
model
'''
model = MaskedAutoencoder(utils, args, num_features)

if args.task_name=='pretrain':
    
    args.pretrain_checkpoints_dir = os.path.join(args.pretrain_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.pretrain_checkpoints_dir):
        os.makedirs(args.pretrain_checkpoints_dir)
    
    history = model.train_model(train_X, val_X, train_lake_names, val_lake_names, vars(args), utils=utils)
    
    model_path = os.path.join(args.pretrain_checkpoints_dir, args.ckpt_name)
    
    torch.save(model, model_path)
    
    # if args.zero_shot=='True':
    #     model.forecast_evaluate(train_X, val_X, vars(args), lookback=args.lookback_window)
        
elif args.task_name=='finetune':

    args.finetune_checkpoints_dir = os.path.join(args.finetune_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.finetune_checkpoints_dir):
        os.makedirs(args.finetune_checkpoints_dir)
    
    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    
    if os.path.exists(load_model_path) and args.load_pretrain=="True":
        '''
        fine-tune an already pretrained model
        '''
        model = torch.load(load_model_path, map_location='cpu')
    
    history = model.train_model(train_X, val_X, train_lake_names, val_lake_names, vars(args), utils=utils)
    save_model_path = os.path.join(args.finetune_checkpoints_dir, args.ckpt_name)
    torch.save(model, save_model_path)

elif args.task_name=='zeroshot':
        
    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    
    if os.path.exists(load_model_path) and args.load_pretrain=="True":
        '''
        perform zero-shot using already pretrained model
        '''
        model = torch.load(load_model_path, map_location='cpu')
        
    model.perform_zero_shot(train_X, val_X, vars(args), train_lake_names, val_lake_names, lookback=args.lookback_window, utils=utils)
    
print(f"Done with model {args.task_name} ")