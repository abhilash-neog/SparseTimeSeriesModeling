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
from utils.utils import Utils
from functools import partial


class ETT_Dataset(Dataset):
    
def read_config(args):
    config_path = os.path.join(args.config_base, args.config_name)
    with open(config_path, 'r') as json_file:
        config = json.load(json_file)

    return config

def read_data(args):
    filepath = os.path.join(args.root_path, args.data_path, args.source_filename)
    df = pd.read_csv(filepath)

    features_col = df.columns[1:]
    df_X = df[features_col]
    
    date_col = df.columns[0]
    
    df_date = df[['date']]
    df[date_col] = df[date_col].astype('datetime64[ns]')
    train_df = df.copy(deep=True)