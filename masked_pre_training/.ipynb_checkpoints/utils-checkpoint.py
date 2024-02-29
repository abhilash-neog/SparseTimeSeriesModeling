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

class FeatEmbed(nn.Module):
    """
    Embed each feature
    """
    def __init__(self, input_dim=8, embedding_dim=8, norm_layer=None):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, embedding_dim),
                nn.LayerNorm(embedding_dim)
            )
            for _ in range(input_dim)
        ])

    def forward(self, x):
        embedded_features = [emb_layer(x[:, :, i].unsqueeze(-1)) for i, emb_layer in enumerate(self.embeddings)]
        embedded_features = torch.stack(embedded_features, dim=2)
        return embedded_features

class ActiveEmbed(nn.Module):
    """ 
    record to mask embedding
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

    def __init__(self, X, M, lake_names):
        self.X = X
        self.M = M
        self.lake_names = lake_names

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.M[idx], self.lake_names[idx]


class Utils:
    
    def __init__(self, num_features, inp_cols, date_col, window, flag_cols, non_null_ratio, device, task_name, n2one="False", lookback_window=14, stride=1):
        self.num_features = num_features
        self.inp_cols = inp_cols
        self.date_col = date_col
        self.pre_train_window = window
        self.num_out_features = num_features
        self.stride = stride
        self.y_mean = None
        self.y_std = None
        self.yearly_samples = {}
        self.flag_cols = flag_cols
        self.non_null_ratio = non_null_ratio
        self.device = device
        self.windowed_dataset_path = ""
        self.task_name = task_name
        self.n2one = n2one
        self.lookback_window = lookback_window
        
        self.chloro_index = self.inp_cols.index('Chla_ugL')
    
    def train_test_split(self, df, split_type='time', split_date=None, split_ratio=0.0):
        '''

        split time series into train/test sets

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
    
    def load_pickle(self, path, lake):
        
        lake_arr = None
        lake_path = os.path.join(path, lake) + '.pkl'
        
        with open(lake_path, 'rb') as pickle_file:
            lake_arr = pickle.load(pickle_file)
        
        return lake_arr
        
    def split_data(self, path, config):
        train_X = []
        val_X = []
    
        train_lake_names = []
        val_lake_names = []
        
        for ind, lake in enumerate(config['train_lakes']):
            lake_arr = self.load_pickle(path, lake)
            
            if len(lake_arr)==0:
                continue
            
            lake_frac = config['train_fractions'][ind]
            num_samples = lake_arr.shape[0]

            end_index = int(num_samples * lake_frac)  # Ending index for the train fraction
            train_X.append(lake_arr[:end_index, :, :])
            
            train_lake_names += [lake]*end_index
            
        for ind, lake in enumerate(config['val_lakes']):
            lake_arr = self.load_pickle(path, lake)
            if len(lake_arr)==0:
                continue
                
            lake_frac = config['val_fractions'][ind]
            num_samples = lake_arr.shape[0]

            if lake in config['train_lakes']:
                if lake_frac + config['train_fractions'][config['train_lakes'].index(lake)] <= 1:
                    end_index = num_samples - int(num_samples * lake_frac)  # Ending index for the train fraction
                    val_X.append(lake_arr[end_index:, :, :])
                else:
                    print(f"val frac + train frac for lake = {lake} is greater than 1. So skipping putting it in val set")
            else:
                end_index = num_samples - int(num_samples * lake_frac)  # Ending index for the train fraction
                val_X.append(lake_arr[end_index:, :, :])
            
            val_lake_names += [lake]*(lake_arr.shape[0]-end_index)

        return np.concatenate(train_X, axis=0), np.concatenate(val_X, axis=0), train_lake_names, val_lake_names
    
    def normalize_tensor(self, tensor, use_stat=False):
        eps = 1e-5 # epsilon for zero std
        if not use_stat:
            self.feat_mean = tensor.nanmean(dim=(0, 1))[None, None, :]
            mask = torch.isnan(tensor)
            filtered_data = tensor.clone()
            filtered_data[mask] = 0
            
            rev_mask = 1-(mask*1)
            
            sqred_values = rev_mask*((filtered_data - self.feat_mean)**2)
            sqred_sum = sqred_values.sum(dim=(0, 1))
            variance = sqred_sum/torch.sum(rev_mask, dim=(0, 1))
            
            self.feat_std = torch.sqrt(variance)[None, None, :]            
            
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
        self.feat_std = df[self.inp_cols].std(skipna=True)
        self.feat_mean = df[self.inp_cols].mean(skipna=True)
        
        df[self.inp_cols] = (df[self.inp_cols]-self.feat_mean)/self.feat_std
            
        return df
        
    def windowed_dataset(self, df, lake, include_target=True):
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

        L = df.shape[0]
        num_samples = (L - self.pre_train_window) // self.stride + 1
        
        dfX = df[self.inp_cols]
        flag_cols = df[self.flag_cols]
        
        X = np.array([]) #np.zeros([num_samples, self.pre_train_window, self.num_features])
        # target_X = np.zeros([self.input_window, num_samples, self.num_out_features])
        # shuffled_inds = random.sample(range(num_samples),num_samples)
        print(f"Lake = {lake}")
        for ii in tqdm(np.arange(num_samples)):
            start_x = self.stride * ii
            end_x = start_x + self.pre_train_window
            
            subset_df = df.iloc[start_x:end_x, :].copy(deep=True)
            subset_dfX = dfX.iloc[start_x:end_x, :].copy(deep=True)
            
            # Set interpolated cells to NULL
            for col in dfX.columns:
                
                # Get corresponding flag column
                flag_col = f'Flag_{col}'

                # Update values where flag is 1
                subset_dfX.loc[subset_df[flag_col] == 1, col] = np.nan
                
            # condition = (subset_df[self.flag_cols] != 0).any(axis=1)
            # subset_dfX.loc[condition] = None
            
            null_inds = subset_dfX[subset_dfX.isnull().all(axis=1)].index
            
            # subset_dfX.loc[null_inds, :] = None
            
            num_nulls = len(null_inds)
            # print(f"num of nulls = {num_nulls} in a subset of shape = {subset_dfX.shape}")
            # non_null_frac = (subset_dfX.shape[0] - num_nulls)/subset_dfX.shape[0]
            
            # print(f"non null frac = {non_null_frac}")
            
            # if non_null_frac >= self.non_null_ratio:
            if num_nulls < 1:
                if X.shape[0]==0:
                    X = np.expand_dims(subset_dfX, axis=0)
                else:
                    toAdd = np.expand_dims(subset_dfX, axis=0)
                    X = np.append(X, toAdd, axis=0)
            # else:
                # print(f"null fraction = {non_null_frac} with num of nulls = {num_nulls} in a shape = {subset_dfX.shape[0]} \n")
                    
        return X
    
    def windowed_dataset_utils(self, df, windowed_dataset_path, include_target=True):
        
        '''
        1. Check if we already have windowed dataset for all lakes
        2. If such window dataset exists, skip
        3. Else, performing windowing on all lakes
        '''
        
        
        lakes = df.Lake.unique()

        for lake in lakes:
            df_temp = df[df['Lake']==lake].reset_index(drop=True)
            dtypes = df_temp.DataType.unique()

            for dtype in dtypes:
                lakename = lake + "_" + dtype + '.pkl'
                lake_path = os.path.join(windowed_dataset_path, lakename)
                
                if os.path.exists(lake_path):
                    print(f"Windowed dataset already exists for lake {lakename}, skipping windowing ...")
                    continue
                
                lake_df = df_temp[df_temp['DataType']==dtype].reset_index(drop='true')
                lake_x = self.windowed_dataset(lake_df, lake + " " + dtype, include_target=include_target)
            
                with open(lake_path, 'wb') as pickle_file:
                    pickle.dump(lake_x, pickle_file)
                    print(f"Pickled lake {lake}")

                print(f"Number of windowed samples from lake {lake} = {lake_x.shape[0]}")
   

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
    
#     def plot_mae(self, sample_index, val_X, predictions, masks, epoch, train_or_val, target_col_ind):
        
#         # val_X = val_X.view(val_X.shape[0], val_X.shape[2], val_X.shape[1])
#         val_X = val_X.transpose(1, 2)
        
#         predictions = predictions.detach()
#         val_X = val_X.detach()
#         masks = masks.detach()
        
#         # Extract the selected sample from the validation set and mask
#         dates = range(len(sample_index)*self.pre_train_window)
        
#         sample_time_series = val_X[sample_index] # GT
#         ts = sample_time_series[0]
    
#         sample_predictions = predictions[sample_index] # Predictions
#         preds = sample_predictions[0]
                               
#         mask = masks[sample_index] # Masked TS
#         maskk = mask[0, :]
    
#         for i in range(1, len(sample_index)):
#             ts = torch.cat((ts, sample_time_series[i]), dim=1)
#             preds = torch.cat((preds, sample_predictions[i]), dim=0)
#             maskk = torch.cat((maskk, mask[i, :]), dim=0)
    
#         ts = ts.cpu().numpy()
#         preds = preds.cpu().numpy()
#         maskk = maskk.cpu().numpy()                           
    
#         # Create a figure
#         num_feats = val_X.shape[1]
#         fix, axes = plt.subplots(num_feats, 1, figsize=(10, num_feats*3))
#         print("NEW PLOTTING CODE.")
#         for idx in range(num_feats):
#             ax = axes[i]
            
#             feature_name = self.inp_cols[idx]
#             # Plot the masked time-series
#             unmasked_values = np.where(maskk == 1, np.nan, ts[idx, :])

#             ax.plot(dates, unmasked_values, label='Unmasked GT', linestyle='-', markersize=4, color='red')
#             ax.plot(dates, preds[:, idx], label='Predictions', marker='v', linestyle='--', markersize=4, color='green')
#             ax.plot(dates, ts[idx, :], label='Masked GT', marker='o', linestyle='--', markersize=4, color='red')

#             subtitle = '{}'.format(feature_name)
#             ax.set_title(subtitle)
#             ax.set_xlabel('Time Step')
#             ax.set_ylabel('Values')
#             ax.set_xticks(rotation=90)
#             ax.legend(loc="best")
        
#         title = '{} Masked Time-Series at epoch: {}'.format(train_or_val, epoch)
#         plt.tight_layout()
#         plt.title(title)
#         wandb.log({title: wandb.Image(plt)})
#         plt.close()
#         # plt.show()

    def plot_forecast(self, df, preds, og_masks, sample_idx, plt_idx, lookback_window, epoch, lake_names, train_or_val, title_prefix):
        """
        This function plots t+plt_idx horizon plots
        """
        preds = preds.detach()
        df = df.detach()
        og_masks = og_masks.detach()
        
        sample_time_series = df[sample_idx] # GT : samples, L, feat
        predictions = preds[sample_idx]
        
        og_mask = og_masks[sample_idx]
        lake_name = np.array(lake_names)[sample_idx]
        all_lake_names = ','.join(np.unique(lake_name).tolist())
        
        sample_time_series = sample_time_series*og_mask
        sample_time_series = torch.where(sample_time_series==0, torch.tensor(float('nan')).to(self.device), sample_time_series)

        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()

        # Create a figure
        num_feats = predictions.shape[2]
        num_samples = len(plt_idx)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(6*num_samples, num_feats*3))
        
        if self.n2one=="True":
            
            for i, sample_id in enumerate(plt_idx):      
                ax = axes[i]
                sample_id = int(sample_id)
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                dates = np.arange(predictions.shape[0])
                ax.plot(dates, predictions[:, sample_id, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, sample_time_series[:, sample_id, 0], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')

                subtitle = f'Lake {all_lake_names}: {feature_name}: Plot at t+{sample_id-lookback_window+1}'
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            
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

                    subtitle = f'Lake {all_lake_names}: {feature_name}: Plot at t+{sample_id-lookback_window+1}'
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
        

    def plot_merged_context_windows(self, df, preds, og_masks, masks, sample_index, epoch, lake_names, train_or_val, title_prefix):
        """
        This function plots merged context windows for random masking. 
        It assumes that stride for windowed dataset gen is 1 and data is not shuffled.
        """
        preds = preds.detach()
        df = df.detach()
        masks = masks.detach()
        og_masks = og_masks.detach()
        # lake_names = lake_names.detach()
        
        # Extract the selected sample from the validation set and mask
        dates = range(len(sample_index)*self.pre_train_window)

        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        mask = masks[sample_index]
        og_mask = og_masks[sample_index]
        lake_name = np.array(lake_names)[sample_index][0]

        ts = sample_time_series[0]
        ps = predictions[0]
        ms = mask[0]
        om = og_mask[0]
        
        for i in range(1, len(sample_index)):
            ts = torch.cat((ts, sample_time_series[i]), dim=0)
            ps = torch.cat((ps, predictions[i]), dim=0)
            ms = torch.cat((ms, mask[i]), dim=0)
            om = torch.cat((om, og_mask[i]), dim=0)
        
        # print(f"ts shape ={ts.shape} \nps shape = {ps.shape} \nms shape = {ms.shape}")
        
        # shape of ms = (875,)
        om = 1 - om
        masked_ts = ms.unsqueeze(-1) * torch.ones(1, ts.shape[1], device=ms.device)
        masked_ts = torch.logical_or(masked_ts, om)
        masked_ts = masked_ts*ts 
        # shape (875, 8)
        
#         print(f"ms shape = {ms.shape}")
#         print(f"masked_ts shape = {masked_ts.shape}")
                
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
            
            # if self.task_name=="finetune":
            #     feature_name = self.inp_cols[self.chloro_index]
            # else:
            feature_name = self.inp_cols[idx]
            
            ax.plot(dates, ps[:, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                    color='green')
            ax.plot(dates, ts[:, idx], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                     color='blue')
            ax.plot(dates, masked_ts[:, idx], label='Masked TS', marker='o', linestyle='-', markersize=1,
                     markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)
            
            subtitle = 'Lake {}: {}'.format(lake_name, feature_name)
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
    
    
    def plot_context_window_grid_with_original_masks(self, df, preds, og_masks, sample_index, epoch, lake_names, train_or_val, title_prefix):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        og_masks = og_masks.detach()
        # lake_names = lake_names.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        og_mask = og_masks[sample_index]
        lake_names = np.array(lake_names)[sample_index]

        og_masked_ts = og_mask*sample_time_series
        og_masked_ts = torch.where(og_masked_ts==0, torch.tensor(float('nan')).to(self.device), og_masked_ts)
        
        masked_ts = og_masked_ts.cpu().numpy()
        og_masked_ts = og_masked_ts.cpu().numpy()
        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        
        # Create a figure
        num_feats = preds.shape[2]
        num_samples = len(sample_index)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(4*num_samples, num_feats*3))
        
        if self.n2one=="True":
            
            for sample_id in range(num_samples):
                # for idx in range(num_feats):
                # ax = axes[idx, sample_id]      
                ax = axes[sample_id]
                lake = lake_names[sample_id]
                
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                dates = np.arange(predictions.shape[1])
                plt.tight_layout()
                ax.plot(dates, predictions[sample_id, :, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, masked_ts[sample_id, :, 0], label='Original TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                # ax.plot(dates, masked_ts[sample_id, :, 0], label='Original Masked TS', marker='o', linestyle='-', markersize=1,
                #          markerfacecolor='yellow', markeredgecolor='yellow', color='yellow', alpha=0.9)
                ax.axvline(x=self.lookback_window, color='black', linestyle='-', linewidth=2)
                
                subtitle = 'Lake {}: {}'.format(lake, feature_name)
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            for sample_id in range(num_samples):
                lake = lake_names[sample_id]
                
                for idx in range(num_feats):
                    ax = axes[idx, sample_id]      

                    feature_name = self.inp_cols[idx]
                    dates = np.arange(predictions.shape[1])
                    ax.plot(dates, predictions[sample_id, :, idx], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                            color='green')
                    ax.plot(dates, masked_ts[sample_id, :, idx], label='Original TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                    # ax.plot(dates, masked_ts[sample_id, :, idx], label='Original Masked TS', marker='o', linestyle='-', markersize=1,
                    #          markerfacecolor='yellow', markeredgecolor='yellow', color='yellow', alpha=0.9)
                    
                    if self.task_name=='finetune' or self.task_name=='zeroshot':
                        plt.tight_layout()
                        ax.axvline(x=self.lookback_window, color='black', linestyle='-', linewidth=2)
                        
                    subtitle = 'Lake {}: {}'.format(lake, feature_name)
                    ax.set_title(subtitle)
                    ax.set_xlabel('Time Step')
                    ax.set_ylabel('Values')
                    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                    ax.legend(loc="best")
        
        title = '{}: {} Single Windows w/Original Masks at epoch: {}'.format(title_prefix, train_or_val, epoch)
        plt.tight_layout()
        plt.suptitle(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def plot_context_window_grid(self, df, preds, masks, og_masks, sample_index, epoch, lake_names, train_or_val, title_prefix):
        """
        This function creates a grid of size Number of features X Num_Samples
        where num_samples = len(sample_index)
        """
        preds = preds.detach()
        df = df.detach()
        masks = masks.detach()
        og_masks = og_masks.detach()
        # lake_names = lake_names.detach()
        
        sample_time_series = df[sample_index] # GT
        predictions = preds[sample_index]
        mask = masks[sample_index]
        og_mask = og_masks[sample_index]
        # print(f"lake names = {lake_names[0]}")
        # print(f"lake names II = {lake_names[1]}")
        # print(f"len of lake names = {len(lake_names)}")
        lake_names = np.array(lake_names)[sample_index]
        
        og_mask = 1-og_mask
        mask = mask.unsqueeze(-1) * torch.ones(1, predictions.shape[2], device=mask.device)
        mask = torch.logical_or(mask, og_mask)
        masked_ts = mask*sample_time_series
        masked_ts = torch.where(masked_ts==0, torch.tensor(float('nan')).to(self.device), masked_ts)
        
        masked_ts = masked_ts.cpu().numpy()
        sample_time_series = sample_time_series.cpu().numpy()                          
        predictions = predictions.cpu().numpy()
        mask = mask.cpu().numpy()

        # Create a figure
        num_feats = preds.shape[2]
        num_samples = len(sample_index)
        
        fig, axes = plt.subplots(num_feats, num_samples, figsize=(4*num_samples, num_feats*3))

        if self.n2one=="True":
            
            for sample_id in range(num_samples):
                # for idx in range(num_feats):
                # ax = axes[idx, sample_id]      
                ax = axes[sample_id]
                
                feature_name = self.inp_cols[self.chloro_index]#[idx]
                lake = lake_names[sample_id]
                dates = np.arange(predictions.shape[1])
                plt.tight_layout()
                ax.plot(dates, predictions[sample_id, :, 0], label='Predictions TS', marker='o', linestyle='-', markersize=1,
                        color='green')
                ax.plot(dates, sample_time_series[sample_id, :, 0], label='Unmasked TS', marker='o', linestyle='-', markersize=1, 
                         color='blue')
                ax.plot(dates, masked_ts[sample_id, :, 0], label='Masked TS', marker='o', linestyle='-', markersize=1,
                         markerfacecolor='red', markeredgecolor='red', color='red', alpha=0.7)
                
                ax.axvline(x=self.lookback_window, color='black', linestyle='-', linewidth=2)
                
                subtitle = 'Lake {}: {}'.format(lake, feature_name)
                ax.set_title(subtitle)
                ax.set_xlabel('Time Step')
                ax.set_ylabel('Values')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
                ax.legend(loc="best")
        else:
            for sample_id in range(num_samples):
                lake = lake_names[sample_id]
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
                    
                    if self.task_name=='finetune' or self.task_name=='zeroshot':
                        plt.tight_layout()
                        ax.axvline(x=self.lookback_window, color='black', linestyle='-', linewidth=2)
                    
                    subtitle = 'Lake {}: {}'.format(lake, feature_name)
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