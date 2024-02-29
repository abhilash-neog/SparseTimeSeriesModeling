import numpy as np
import torch, os
import pandas as pd
import math
import argparse
import wandb
import matplotlib.pyplot as plt
import pickle

from torch import nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class MaskEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, in_channel=14, embed_dim=64, norm_layer=None):
        
        super().__init__()
        self.proj = nn.Conv1d(in_channel, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        return x


class ActiveEmbed(nn.Module):
    """ record to mask embedding
    """
    def __init__(self, embed_dim=64, norm_layer=None):
        
        super().__init__()
        # self.rec_len = rec_len
        self.proj = nn.Conv1d(1, embed_dim, kernel_size=1, stride=1)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, _, L = x.shape
        # assert(L == self.rec_len, f"Input data width ({L}) doesn't match model ({self.rec_len}).")
        x = self.proj(x)
        x = torch.sin(x)
        x = x.transpose(1, 2)
        #   x = torch.cat((torch.sin(x), torch.cos(x + math.pi/2)), -1)
        x = self.norm(x)
        return x



def get_1d_sincos_pos_embed(embed_dim, pos, cls_token=False):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """

    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(pos)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def adjust_learning_rate(optimizer, epoch, lr, min_lr, max_epochs, warmup_epochs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    
    if epoch < warmup_epochs:
        tmp_lr = lr * epoch / warmup_epochs 
    else:
        tmp_lr = min_lr + (lr - min_lr) * 0.5 * \
            (1. + math.cos(math.pi * (epoch - warmup_epochs) / (max_epochs - warmup_epochs)))
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = tmp_lr * param_group["lr_scale"]
        else:
            param_group["lr"] = tmp_lr
    return tmp_lr


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == np.inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


class NativeScaler:

    state_dict_key = "amp_scaler"
    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)



class MAEDataset(Dataset):

    def __init__(self, X, M):
        self.X = X
        self.M = M

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx]


class Utils:
    
    def __init__(self, num_features, inp_cols, date_col, window, non_null_ratio, device, config, stride=1):
        self.num_features = num_features
        self.inp_cols = inp_cols
        self.date_col = date_col
        self.pre_train_window = window
        self.num_out_features = num_features
        self.stride = stride
        self.y_mean = None
        self.y_std = None
        self.yearly_samples = {}
        self.non_null_ratio = non_null_ratio
        self.device = device
        self.windowed_dataset_path = ""
        self.config=config
    
    def train_test_split(self, df):
        '''

        : param df:                     time array
        : para y:                       feature array
        : para split:                   percent of data to include in training set
        : return t_train, y_train:      time/feature training and test sets;
        :        t_test, y_test:        (shape: [# samples, 1])

        '''
        if split_type == 'time':
            df_train = df[df[self.date_col] <= split_date]
            df_test = df[df[self.date_col] > split_date]
            return df_train, df_test
        else:
            indx_split = int(split_ratio * df.shape[0])
            indx_train = np.arange(0, indx_split)
            indx_test = np.arange(indx_split, df.shape[0])
    
            df_train = df.iloc[indx_train]
            df_test = df.iloc[indx_test]

        return df_train.reset_index(drop='true'), df_test.reset_index(drop='true')
    
    def load_pickle(self, path, name):
        
        arr = None
        data_path = os.path.join(path, name)
        
        with open(data_path, 'rb') as pickle_file:
            arr = pickle.load(pickle_file)
        
        return arr
        
    def split_data(self, df, config):
        '''
        split time series into train/test sets
        
        The ratio is fixed for the benchmark datasets
        For eg. ETT split ratio is 6:2:2 or 12/4/4 months
        '''
        start_date = df.index.min()
    
        if config['name'] in ["m1", "m2"]:
            factor = 4  # 15-min frequency
        else:
            factor = 1  # hourly frequency
    
        train_fraction = config['train_fraction']
        dev_fraction = config['val_fraction']
        test_fraction = config['test_fraction']
        
        train_end_date_index = train_fraction * 30 * 24 * factor  # 1 year

        dev_end_date_index = train_end_date_index + dev_fraction * 30 * 24 * factor  # 1 year + 4 months
        test_end_date_index = train_end_date_index + 2*test_fraction * 30 * 24 * factor  # 1 year + 8 months
        
        train = df.iloc[:train_end_date_index,:]
        dev = df.iloc[train_end_date_index:dev_end_date_index, :]
        test = df.iloc[dev_end_date_index:test_end_date_index, :]

        return train, dev, test
    
    def normalize_tensor(self, tensor, use_stat=False):
        eps = 1e-5 # epsilon for zero std
        if not use_stat:
            self.feat_mean = tensor.mean(dim=(0, 1))[None, None, :]
            self.feat_std = tensor.std(dim=(0, 1))[None, None, :]
        tensor = (tensor-self.feat_mean)/(self.feat_std+eps)
        return tensor
        
    def normalize_pd(self, df, use_stat=False):
        '''
        Normalize data
        '''
        if use_stat:
            df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            return df
                
        # compute mean and std of target variable - to be used for unnormalizing
        self.feat_std = df[self.inp_cols].std()
        self.feat_mean = df[self.inp_cols].mean()
        
        df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            
        return df
    
    def perform_windowing(self, df, path, split='train'):
        '''
        create a windowed dataset
    
        : param y:                time series feature (array)
        : param input_window:     number of y samples to give model
        : param output_window:    number of future y samples to predict
        : param stide:            spacing between windows
        : param num_features:     number of features (i.e., 1 for us, but we could have multiple features)
        : return X, Y:            arrays with correct dimensions for LSTM
        :                         (i.e., [input/output window size # examples, # features])
        '''
        
        filename=self.config['name'] + '_' + split + '.pkl'
        save_path=os.path.join(path, filename)
        
        if os.path.exists(save_path):
            print("Window dataset already exists")
            X = self.load_pickle(path, filename)
            return X
        
        else:
            L = df.shape[0]
            num_samples = (L - self.pre_train_window) // self.stride + 1

            dfX = df[self.inp_cols]

            X = np.array([]) #np.zeros([num_samples, self.pre_train_window, self.num_features])
            # target_X = np.zeros([self.input_window, num_samples, self.num_out_features])
            # shuffled_inds = random.sample(range(num_samples),num_samples)
            for ii in tqdm(np.arange(num_samples)):
                start_x = self.stride * ii
                end_x = start_x + self.pre_train_window

                subset_dfX = dfX.iloc[start_x:end_x, :].copy(deep=True)

                if X.shape[0]==0:
                    X = np.expand_dims(subset_dfX, axis=0)
                else:
                    toAdd = np.expand_dims(subset_dfX, axis=0)
                    X = np.append(X, toAdd, axis=0)


            with open(save_path, 'wb') as pickle_file:
                pickle.dump(X, pickle_file)
                print(f"Pickled dataset {filename}")

            return X
    
    
    def numpy_to_torch(self, Xtrain, Ytrain, Xtest, Ytest):
        '''
        convert numpy array to PyTorch tensor
    
        : param Xtrain:               windowed training input data (# examples, input window size, # features); np.array
        : param Ytrain:               windowed training target data (# examples, output window size, # features); np.array
        : param Xtest:                windowed test input data (# examples, input window size, # features); np.array
        : param Ytest:                windowed test target data (# examples, output window size, # features); np.array
        : return X_train_torch, Y_train_torch,
        :        X_test_torch, Y_test_torch:      all input np.arrays converted to PyTorch tensors

        '''

        X_train_torch = torch.from_numpy(Xtrain).type(torch.Tensor)
        Y_train_torch = torch.from_numpy(Ytrain).type(torch.Tensor)

        X_test_torch = torch.from_numpy(Xtest).type(torch.Tensor)
        Y_test_torch = torch.from_numpy(Ytest).type(torch.Tensor)

        return X_train_torch, Y_train_torch, X_test_torch, Y_test_torch
    

    def plot_forecast(self, df, preds, sample_idx, plt_idx, lookback_window, epoch, train_or_val, title_prefix):
        """
        This function plots t+plt_idx horizon plots
        """
        preds = preds.detach()
        df = df.detach()
        
        sample_time_series = df[sample_idx] # GT : samples, L, feat
        predictions = preds[sample_idx]

        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()

        # Create a figure
        num_feats = predictions.shape[2]
        num_samples = len(plt_idx)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(6*num_samples, num_feats*3))
        
        for i, sample_id in enumerate(plt_idx):
            for idx in range(num_feats):
                ax = axes[idx, i]      
                sample_id = int(sample_id)
                feature_name = self.inp_cols[idx]
                dates = np.arange(predictions.shape[0])
                ax.plot(dates, predictions[:, sample_id, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, sample_time_series[:, sample_id, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                
                subtitle = f'{feature_name}: Plot at t+{sample_id-lookback_window+1}'
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        
        title = f'{title_prefix}: {train_or_val} t+N Forecasts at epoch: {epoch}'
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
        

    def plot_merged_context_windows(self, df, preds, masks, sample_index, epoch, train_or_val, title_prefix):
        """
        This function plots merged context windows for random masking. 
        It assumes that stride for windowed dataset gen is 1 and data is not shuffled.
        """
        preds = preds.detach()
        df = df.detach()
        masks = masks.detach()
        
        # Extract the selected sample from the validation set and mask
        dates = range(len(sample_index)*self.pre_train_window)

        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        mask = masks[sample_index]

        ts = sample_time_series[0]
        ps = predictions[0]
        ms = mask[0]

        for i in range(1, len(sample_index)):
            ts = torch.cat((ts, sample_time_series[i]), dim=0)
            ps = torch.cat((ps, predictions[i]), dim=0)
            ms = torch.cat((ms, mask[i]), dim=0)
        
        # print(f"ts shape ={ts.shape} \nps shape = {ps.shape} \nms shape = {ms.shape}")
        
        masked_ts = ms.unsqueeze(1)*ts
        # unmasked_ts = (1-ms).unsqueeze(1)*ts

        masked_ts = torch.where(masked_ts==0, torch.tensor(float('nan')).to(self.device), masked_ts)
        # unmasked_ts = torch.where(unmasked_ts== 0, torch.tensor(float('nan')).to(device), unmasked_ts)

        masked_ts = masked_ts.cpu().numpy()
        # unmasked_ts = unmasked_ts.cpu().numpy()

        ts = ts.cpu().numpy()                          
        ps = ps.cpu().numpy()
        ms = ms.cpu().numpy()

        # Create a figure
        num_feats = preds.shape[2]
        fig, axes = plt.subplots(num_feats, 1, figsize=(10, num_feats*3))
        for idx in range(num_feats):
            ax = axes[idx]
            
            feature_name = self.inp_cols[idx]
            
            ax.plot(dates, ps[:, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                    color='green')
            ax.plot(dates, ts[:, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                     color='blue')
            ax.plot(dates, masked_ts[:, idx], label='Masked TS', marker='o', linestyle='-', markersize=1,
                     markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)
            
            subtitle = '{}'.format(feature_name)
            ax.set_title(subtitle)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Values')
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
            ax.legend(loc="best")
        
        title = '{}: {} Merged Context-Windows at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_context_window_grid(self, df, preds, masks, sample_index, epoch, train_or_val, title_prefix):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        masks = masks.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        mask = masks[sample_index]

        masked_ts = mask.unsqueeze(-1)*sample_time_series
        masked_ts = torch.where(masked_ts==0, torch.tensor(float('nan')).to(self.device), masked_ts)

        masked_ts = masked_ts.cpu().numpy()
        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        mask = mask.cpu().numpy()

        # Create a figure
        num_feats = preds.shape[2]
        num_samples = len(sample_index)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(4*num_samples, num_feats*3))
        
        for sample_id in range(num_samples):
            for idx in range(num_feats):
                ax = axes[idx, sample_id]      

                feature_name = self.inp_cols[idx]
                dates = np.arange(predictions.shape[1])
                ax.plot(dates, predictions[sample_id, :, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, sample_time_series[sample_id, :, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                ax.plot(dates, masked_ts[sample_id, :, idx], label='Masked TS', marker='o', linestyle='-', markersize=1,
                         markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)

                subtitle = '{}'.format(feature_name)
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        
        title = '{}: {} Individual Context-Windows at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
