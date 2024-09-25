import sys
sys.path.insert(1, './utils/')
sys.path.append('../')

import optuna
from optuna.trial import TrialState
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
from optuna.samplers import RandomSampler

from trainer import Trainer
from model import MaskedAutoencoder
from utils.util import Utils
from tools import transfer_weights
from functools import partial
from data_handler import DataHandler

warnings.filterwarnings('ignore')

'''
REMOVE NAME FROM THE CODE
'''
#2023



parser = argparse.ArgumentParser(description='benchmarktesting')

# basic config
parser.add_argument('--task_name', type=str, required=True, default='pretrain', choices=['pretrain', 'finetune'], help='task name, options:[pretrain, finetune]')
parser.add_argument('--seed', type=int, default=2023)

# data loader
parser.add_argument('--dataset', type=str, required=True, default='ETT', help='dataset type')
parser.add_argument('--root_path', type=str, default='/raid/abhilash/forecasting_datasets/ETT/', help='root path of the data, code and model files')
parser.add_argument('--source_filename', type=str, default='ETTh1', help='name of the data file')
parser.add_argument('--timeenc', type=int, default=2, choices=[0, 1, 2], help='0 indicates traditional time features, 1 indicates time-features , 2 indicates no time-feature creation')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')

# model loader
parser.add_argument('--finetune_checkpoints_dir', type=str, default='./finetune_checkpoints/', help='location of model fine-tuning checkpoints')
parser.add_argument('--pretrain_checkpoints_dir', type=str, default='./pretrain_checkpoints/', help='location of model pre-training checkpoints')
parser.add_argument('--pretrain_ckpt_name', type=str, default='ckpt_best.pth', help='checkpoints we will use to finetune, options:[ckpt_best.pth, ckpt10.pth, ckpt20.pth...]')
parser.add_argument('--ckpt_name', type=str, default='ckpt_latest.pth', help='name of the checkpoint to be saved for the current task')
parser.add_argument('--pretrain_run_name', type=str, default='ett_pretrain_initial')
parser.add_argument('--finetune_run_name', type=str, default='ett_finetune_initial')
parser.add_argument('--load_pretrain', type=str, default='True', help='If False will not load pretrained model')

# pretraining task
parser.add_argument('--seq_len', type=int, default=336, help='window for pretraining and fine-tuning')
parser.add_argument('--label_len', type=int, default=48, help='window for pretraining and fine-tuning')
parser.add_argument('--mask_ratio', type=float, default=0.5, help='mask ratio')
parser.add_argument('--num_samples', type=int, default=10, help='number of sample regions to plot for each feature during pretraining')
parser.add_argument('--num_windows', type=int, default=25, help='number of windows to generate merged plots')
parser.add_argument('--feature_wise_mse', type=str, default='True', help='whether to plot feature-wise mse')

# finetuning task
parser.add_argument('--pred_len', type=int, default=720, help='past sequence length')
parser.add_argument('--freeze_encoder', type=str, default='True', help='whether to freeze encoder or not')
parser.add_argument('--n2one', type=bool, default=False, help='multivariate featurest to univariate target')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--output_path', type=str, default='', help='path where all output are stored')

# model define
parser.add_argument('--encoder_embed_range', type=int, nargs='+', default=[4, 8, 16, 32, 64], help='encoder embedding dimension in the feature space')
parser.add_argument('--encoder_depth_range', type=int, nargs='+', default=[1,2,3,4], help='number of encoder blocks')
parser.add_argument('--encoder_heads_range', type=int, nargs='+', default=[1,4,8,16,32], help='number of encoder multi-attention heads')
parser.add_argument('--decoder_embed_range', type=int, nargs='+', default=[4,8, 16, 32, 64], help='decoder embedding dimension in the feature space')
parser.add_argument('--decoder_depth_range', type=int, nargs='+', default=[1,2,3,4], help='number of decoder blocks')
parser.add_argument('--decoder_heads_range', type=int, nargs='+', default=[1,4,8,16,32], help='number of decoder multi-attention heads')
parser.add_argument('--mlp_ratio', type=int, default=4, help='mlp ratio for vision transformer')

# training 
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--accum_iter', type=int, default=1, help='accumulation iteration for gradient accumulation')
parser.add_argument('--min_lr', type=float, default=1e-5, help='min learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--pretrain_lr', type=float, default=0.0001)
parser.add_argument('--finetune_lr', type=float, default=0.0001)
parser.add_argument('--blr', type=float, default=1e-4, help='base learning rate')
parser.add_argument('--lr', type=float, default=1e-4, help='filler')
parser.add_argument('--warmup_epochs', type=int, default=5, help='number of warmup epochs for learning rate')
parser.add_argument('--pretrain_epochs', type=int, default=50)
parser.add_argument('--finetune_epochs', type=int, default=10)
parser.add_argument('--eval_freq', type=int, default=1, help='frequency at which we are evaluating the model during training')
parser.add_argument('--dropout_range', type=float, nargs='+', default=[0.001, 0.2], help='dropout')
parser.add_argument('--fc_dropout_range', type=float, nargs='+', default=[0.0, 0.05], help='fine-tuning dropout')

# GPU
parser.add_argument('--device', type=str, default='3', help='cuda device')
parser.add_argument('--db', type=str, default='db.sqlite3', help='db file')
parser.add_argument('--stats_file', type=str, default='study_stats.txt', help='output stats file')
parser.add_argument('--study_name', type=str, default='h1tuning', help='name of the study')
parser.add_argument('--ntrials', type=int, default=10, help='number of trials')

# weights and biases
parser.add_argument('--project_name', type=str, default='ett', help='project name in wandb')
parser.add_argument('--run_name', type=str, default='mae_pretraining_run', help='run name in wandb')
parser.add_argument('--save_code', type=str, default='True', help='whether to log code in wandb or not')

args = parser.parse_args()

fix_seed = args.seed
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

'''
set the cuda device
'''
args.device = 'cuda:' + args.device if torch.cuda.is_available() else 'cpu'
    
'''
read and process data
'''
dh = DataHandler(args) 
utils = dh.handle()

base_pretrain_run_name = args.pretrain_run_name
base_finetune_run_name = args.finetune_run_name

def objective(trial):
    '''
    Pretrain
    '''
    args.task_name='pretrain'
    args.lr = args.pretrain_lr
    
    model = MaskedAutoencoder(utils, args, trial, num_feats=len(dh.handler.features_col))

    trainer = Trainer(args=vars(args), model=model, utils=utils)
    
    args.pretrain_run_name = "{}_{}_{}".format(base_pretrain_run_name, str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

    args.pretrain_checkpoints_dir = os.path.join(args.pretrain_checkpoints_dir, base_pretrain_run_name) 

    if not os.path.exists(args.pretrain_checkpoints_dir):
        os.makedirs(args.pretrain_checkpoints_dir)

    history, model = trainer.pretrain()

    model_path = os.path.join(args.pretrain_checkpoints_dir, args.ckpt_name)
    
    '''
    Finetune
    '''
    args.task_name='finetune'
    args.lr = args.finetune_lr
    
    args.finetune_run_name = "{}_{}_{}".format(base_finetune_run_name, str(datetime.datetime.now().date()), str(datetime.datetime.now().time()))

    args.finetune_checkpoints_dir = os.path.join(args.finetune_checkpoints_dir, base_finetune_run_name) 

    if not os.path.exists(args.finetune_checkpoints_dir):
        os.makedirs(args.finetune_checkpoints_dir)

    load_model_path = os.path.join(args.pretrain_checkpoints_dir, args.pretrain_run_name, args.pretrain_ckpt_name)
    
    model = MaskedAutoencoder(utils, args, trial, num_feats=len(dh.handler.features_col))

    if os.path.exists(load_model_path):
        '''
        fine-tune an already pretrained model
        '''
        model = transfer_weights(load_model_path, model, device=args.device)
        
    trainer = Trainer(args=vars(args), model=model, utils=utils)
    val_mse = trainer.finetune(trial)
    return val_mse
    

# Load the study if it exists, otherwise create a new one
try:
    study = optuna.load_study(study_name=args.study_name, storage="sqlite:///"+args.db)
    print(f"Previous study loaded")
    print(len(study.trials))
except:
    print(f"No previous study found")
    print(f"Creating a new study")
    study = optuna.create_study(storage="sqlite:///"+args.db, sampler=RandomSampler(seed=args.seed), study_name=args.study_name, direction="minimize")
    
study.optimize(objective, n_trials=args.ntrials)

pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

with open(args.stats_file, 'w') as f:
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    f.write("Study statistics:\n")
    f.write("  Number of finished trials: {}\n".format(len(study.trials)))
    f.write("  Number of pruned trials: {}\n".format(len(pruned_trials)))
    f.write("  Number of complete trials: {}\n".format(len(complete_trials)))

    f.write("Best trial:\n")
    trial = study.best_trial

    f.write("  Value: {}\n".format(trial.value))

    f.write("  Params:\n")
    for key, value in trial.params.items():
        f.write("    {}: {}\n".format(key, value))