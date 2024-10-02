import argparse
import torch
import random
import numpy as np
import os
import json
import pandas as pd
import math
import datetime
import wandb

from utils import Utils
from functools import partial
from model import seq2seq, init_wandb

import sys


parser = argparse.ArgumentParser(description='LSTM')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='train', choices=['train', 'evaluate'], help='task name, options:[train, evaluate]')
# parser.add_argument('--zero_shot', type=str, default='True', help='zero-shot evaluation perform')

# data loader
parser.add_argument('--root_path', type=str, default='./', help='root path of the data, code and model files')
parser.add_argument('--data_path', type=str, default='../data', help='path to the data dir')
parser.add_argument('--source_filename', type=str, default='TransferLearningData.csv', help='name of the data file')
parser.add_argument('--non_null_ratio', type=float, default=0.8, help='non null ratio required for considering one window')
parser.add_argument('--config_base', type=str, default='../config', help='path to the config file')
parser.add_argument('--config_name', type=str, default='config.json', help='config file')
parser.add_argument('--horizon_csv_path', type=str, default='./horizon_csv', help='path to horizon window vs rmse csvs')

# model loader
parser.add_argument('--train_checkpoints_dir', type=str, default='./train_checkpoints/', help='location of model training checkpoints')
parser.add_argument('--train_ckpt_name', type=str, default='ckpt_latest.pth', help='checkpoints we will use for testing, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--pretrain_run_name', type=str, default='lstm_eval_test', help='run name in wandb')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')

# pretraining task
parser.add_argument('--lookback_window', type=int, default=21, help='Input window')
parser.add_argument('--horizon_window', type=int, default=14, help='Output window')
parser.add_argument('--horizon_range', nargs='+', type=int, default=[7, 14], help='List of integers')

# model define
parser.add_argument('--num_layers', type=int, default=1, help='number of lstm layers')
parser.add_argument('--hidden_feature_size', type=int, default=8, help='size of the hidden layer')
parser.add_argument('--output_size', type=int, default=1, help='number of output variables')
parser.add_argument('--model_type', type=str, default="LSTM", help='type of model: LSTM, GRU or RNN')
parser.add_argument('--embed_dim', type=int, default=8)
parser.add_argument('--num_heads', type=int, default=4, help='number of heads for multi-head cross-attention heads')

# training 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--batch_shuffle', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--max_lr', type=float, default=5e-2)
parser.add_argument('--div_factor', type=float, default=100)
parser.add_argument('--pct_start', type=float, default=0.05)
parser.add_argument('--anneal_strategy', type=str, default='cos')
parser.add_argument('--final_div_factor', type=float, default=10000.0)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--dropout', type=float, default=0.001)
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')
parser.add_argument('--early_stop', type=bool, default=False, help='Set to True if we want Early stopping')
parser.add_argument('--early_stop_thres', type=int, default=5, help='If there is no improvement for N epochs we stop the training process')
parser.add_argument('--early_stop_delta', type=float, default=0.5, help='Amount of improvement needed for early stopping criteria')
parser.add_argument('--max_epochs', type=int, default=100)
parser.add_argument('--training_prediction', type=str, default='recursive', help='teacher_forcing or recursive or mixed_teacher_forcing')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.0)
parser.add_argument('--dynamic_tf', type=bool, default=False, help='Dynamic Teacher forcing to reduce teacher forcing ratio uniformly every epoch')
def list_of_strings(arg):
    return arg.split(',')
parser.add_argument('--flag_feature', type=list_of_strings, default=[], help='features that are going to be excluded')
parser.add_argument('--flag_noise', type=str, default='', help='options: add_all, replace_all, add_drivers, replace_drivers')
parser.add_argument('--frac_std_noise', type=float, default=0.001, help='')
parser.add_argument('--ntrials', type=int, default=1, help='number of trials we want to run our model for')

# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')

# weights and biases
parser.add_argument('--project_name', type=str, default='lstm_eval', help='project name in wandb')
parser.add_argument('--run_name', type=str, required=True, default='lstm_train', help='run name in wandb')
parser.add_argument('--save_code', type=str, default='True', help='whether to log code in wandb or not')

args = parser.parse_args()
print(f'Args in experiment:{args}')


# '''
# read config file
# '''
# config_path = os.path.join(args.config_base, args.config_name)
# with open(config_path, 'r') as json_file:
#     config = json.load(json_file)

'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'

'''
read the file
'''
filepath = os.path.join(args.root_path, args.data_path, args.source_filename)
df = pd.read_csv(filepath)

'''
Define Input Features
'''
# exclude features given in args
if args.flag_feature == []:
    excluded_features = []
elif args.flag_feature == ["F"]:
    excluded_features = df.columns.difference(["Chla_ugL"]).tolist()
elif args.flag_feature == ["Ch"]:
    excluded_features = ["Chla_ugL"]
else:
    excluded_features = args.flag_feature
    
# define the feature,date and flag columns
cols_to_exclude_from_features = ["Lake","Site","Depth_m","DataType","ModelRunType"] + excluded_features
features_col = df.columns.difference(cols_to_exclude_from_features)
features_col = [feat for feat in features_col if not 'flag' in feat.lower() and feat!='DateTime']

print("included features: ", features_col)

date_col = ['DateTime']
target_col = ['Chla_ugL']

df.DateTime = df.DateTime.astype('datetime64[ns]')
train_df = df.copy(deep=True)

flag_cols = [col for col in train_df.columns if col.startswith('Flag')]
num_features = len(features_col)

'''
initialize utils object
'''
utils = Utils(num_features=num_features,
              inp_cols=features_col,
              target_col=target_col,
              date_col=date_col,
              flag_col=flag_cols,
              args=args,
              stride=1)

utils.chloro_index = df.columns.tolist().index('Chla_ugL')

'''
split and create windowed dataset or load one 
'''
data_path = os.path.join(args.root_path, args.data_path)

train_X, train_lake_names = utils.split_and_window(df, config, data_path, split='train')
test_X, test_lake_names = utils.split_and_window(df, config, data_path, split='test')

'''
normalize the data
'''
train_X = torch.from_numpy(train_X).type(torch.Tensor)
test_X = torch.from_numpy(test_X).type(torch.Tensor)

train_X = utils.normalize_tensor(train_X, use_stat=False)
test_X = utils.normalize_tensor(test_X, use_stat=True)


'''
extract the output
'''
X_train, Y_train = utils.extract_io(train_X)
X_test, Y_test = utils.extract_io(test_X)

start_seed = 2000
seeds = [start_seed + 20 for i in range(args.ntrials)]

train_dicts = []
test_dicts = []

base_run_name = args.run_name

run_name = args.run_name + "_trial_{}_{}_{}"

for trial in range(args.ntrials):
    
    '''
    update run name
    '''
    args.run_name = run_name.format(str(trial), str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))
    
    fix_seed = random.randint(0, 1000)#seeds[trial]
    
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        
    np.random.seed(fix_seed)

    '''
    model
    '''

    model = seq2seq(input_size=len(features_col),
                    utils=utils,
                    args=args)
    
    if args.task_name=='train':

        train_checkpoints_dir = os.path.join(args.train_checkpoints_dir, base_run_name) 
        print(f"ckpt dir = {train_checkpoints_dir}")

        if not os.path.exists(train_checkpoints_dir):
            os.makedirs(train_checkpoints_dir)

        train_eval_dict, test_eval_dict = model.train_model(X_train=X_train,
                                                            Y_train=Y_train,
                                                            X_test=X_test,
                                                            Y_test=Y_test,
                                                            train_lake_names=train_lake_names, 
                                                            args=vars(args), 
                                                            utils=utils)
        
        train_dicts.append(train_eval_dict)
        test_dicts.append(test_eval_dict)
        
        ckpt_name = args.ckpt_name + '_trial_' + str(trial) +'.pth'
        
        model_path = os.path.join(train_checkpoints_dir, ckpt_name)
        print(f"model_path = {model_path}")
        
        torch.save(model, model_path)

'''
plot the mean predictions with the corresponding error bars
'''
args.run_name = run_name.format("final", str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

config = init_wandb(vars(args), args.task_name)

model.evaluate_uncertainty(args=vars(args), eval_dict=train_dicts, train_or_val='train')
model.evaluate_uncertainty(args=vars(args), eval_dict=test_dicts, train_or_val='test')
wandb.finish()

print(f"Done with model {args.task_name} ")