import sys
sys.path.insert(1, './utils/')

import warnings
import argparse
import torch
import random
import numpy as np
import os
import json
import pandas as pd
import math
import datetime
import timefeatures

from model_mae import MaskedAutoencoder
from utils import Utils
from functools import partial


warnings.filterwarnings('ignore')
'''
----------------
modification log
----------------
Date: Nov 24, 2023
Tag: initial_implementation
Description: Initially we are going with the assumptions that pre-train and fine-tune are tasks that are run separately
Author: Abhilash

----------------
Date: Dec 6, 2023
Tag: benchmark
Description: Adding support for benchmark ETT dataset
Author: Abhilash
'''

fix_seed = 2023
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)


parser = argparse.ArgumentParser(description='benchmarktesting')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', choices=['pretrain', 'finetune'], help='task name, options:[pretrain, finetune]')

# data loader
parser.add_argument('--dataset', type=str, required=True, default='ETT', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data', help='root path of the data, code and model files')
parser.add_argument('--source_filename', type=str, default='ETTh1', help='name of the data file')
parser.add_argument('--timeenc', type=int, default=0, help='whether to time encode or not')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# model loader
parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints_dir', type=str, default='./pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--pretrain_ckpt_name', type=str, default='ckpt_latest.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest.pth', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--pretrain_run_name', type=str, default='ett_pretrain_initial', help='run name in wandb')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')

# pretraining task
parser.add_argument('--window', type=int, default=432, help='window for pretraining and fine-tuning')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
parser.add_argument('--num_samples', type=int, default=10, help='number of sample regions to plot for each feature during pretraining')
parser.add_argument('--num_windows', type=int, default=25, help='number of windows to generate merged plots')
parser.add_argument('--feature_wise_rmse', type=str, default='True', help='whether to plot feature-wise rmse')

# finetuning task
parser.add_argument('--lookback_window', type=int, default=336, help='past sequence length')
parser.add_argument('--freeze_encoder', type=str, default='True', help='whether to freeze encoder or not')
parser.add_argument('--n2one', type=bool, default=False, help='multivariate featurest to univariate target')

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
parser.add_argument('--max_epochs', type=int, default=10)
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')


# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')

# weights and biases
parser.add_argument('--project_name', type=str, default='ett', help='project name in wandb')
parser.add_argument('--run_name', type=str, required=True, default='mae_pretraining_run', help='run name in wandb')
parser.add_argument('--save_code', type=str, default='True', help='whether to log code in wandb or not')

args = parser.parse_args()

'''
update run name
'''
base_run_name = args.run_name
args.run_name = "{}_{}_{}".format(args.run_name, str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))


'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'


'''
read the file
'''

filepath = os.path.join(args.root_path, args.dataset, args.source_filename)
df = pd.read_csv(filepath+'.csv')

# define the feature,date and flag columns
features_col = df.columns[1:]
df_X = df[features_col]

date_col = df.columns[0]
df_date = df[[date_col]]


'''
time features
'''
df_date[date_col] = pd.to_datetime(df_date[date_col])

if args.timeenc==0:
    df_date['month'] = df_date.date.apply(lambda row: row.month, 1)
    df_date['day'] = df_date.date.apply(lambda row: row.day, 1)
    df_date['weekday'] = df_date.date.apply(lambda row: row.weekday(), 1)
    df_date['hour'] = df_date.date.apply(lambda row: row.hour, 1)
    df_date = df_date.drop(['date'], 1)
elif args.timeenc==1:
    df_date = time_features(pd.to_datetime(df_date['date'].values), freq=args.freq)
    df_date = df_date.transpose(1, 0)

df_X = pd.concat([df_X, df_date], axis=1)
    
'''
initialize utils object
'''
utils = Utils(inp_cols=features_col, 
              date_col=date_col, 
              args=args,
              stride=1)

'''
create train and val set
'''
ratios = {'train':0.6, 'val':0.2, 'test':0.2}

train_df, val_df, test_df = utils.split_data(df_X, ratios)

'''
create windowed dataset or load one 
'''
data_path = os.path.join(args.root_path, args.dataset)

train_X = utils.perform_windowing(train_df, data_path, name=args.source_filename, split='train')
val_X = utils.perform_windowing(val_df, data_path, name=args.source_filename, split='val')
test_X = utils.perform_windowing(test_df, data_path, name=args.source_filename, split='test')

'''
standardize the data
'''
train_X = torch.from_numpy(train_X).type(torch.Tensor)
val_X = torch.from_numpy(val_X).type(torch.Tensor)
test_X = torch.from_numpy(test_X).type(torch.Tensor)

train_X = utils.normalize_tensor(train_X, use_stat=False)
val_X = utils.normalize_tensor(val_X, use_stat=True)
test_X = utils.normalize_tensor(test_X, use_stat=True)

# print(f"utils mean = {utils.feat_mean.shape}")
# print(f"utils std = {utils.feat_std.shape}")
# print(f"train_X = {train_X[0, :10, 7:]}")

# train_X = train_X.transpose(1, 2)
# val_X = val_X.transpose(1, 2)
# test_X = test_X.transpose(1, 2)

'''
model
'''
model = MaskedAutoencoder(utils, args, num_feats=train_X.shape[-1])

if args.task_name=='pretrain':
    
    args.pretrain_checkpoints_dir = os.path.join(args.pretrain_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.pretrain_checkpoints_dir):
        os.makedirs(args.pretrain_checkpoints_dir)
    
    history = model.train_model(train_X, val_X, test_X, vars(args), utils=utils)
    
    model_path = os.path.join(args.pretrain_checkpoints_dir, args.ckpt_name)
    
    torch.save(model, model_path)
        
elif args.task_name=='finetune':
    
    args.finetune_checkpoints_dir = os.path.join(args.finetune_checkpoints_dir, base_run_name) 
    
    if not os.path.exists(args.finetune_checkpoints_dir):
        os.makedirs(args.finetune_checkpoints_dir)
    
    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    
    if os.path.exists(load_model_path) and args.load_pretrain=="True":
        '''
        fine-tune an already pretrained model
        '''
        print("Pre-trained model exists. Loading ... ")
        model = torch.load(load_model_path, map_location='cpu')
    
    history = model.train_model(train_X, val_X, test_X, vars(args), utils=utils)
    save_model_path = os.path.join(args.finetune_checkpoints_dir, args.ckpt_name)
    torch.save(model, save_model_path)
    
    print("\n|| TEST EVALUATION ||\n")
    model.forecast_evaluate(train_X, test_X, vars(args), lookback=args.lookback_window)
    
print(f"Done with model {args.task_name} ")