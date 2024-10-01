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


class Utils:
    
    def __init__(self, 
                 num_features, 
                 inp_cols, 
                 target_col, 
                 date_col,
                 flag_col,
                 args,
                 stride=1):
        
        self.num_features = num_features
        self.inp_cols = inp_cols
        self.target_cols = target_col
        self.date_col = date_col
        self.input_window = args.lookback_window
        self.output_window = args.horizon_window
        self.num_out_features = args.output_size
        self.non_null_ratio = args.non_null_ratio
        self.device = args.device
        self.task_name = args.task_name
        self.stride = stride
        self.flag_cols = flag_col
        self.horizon_range = args.horizon_range
        self.horizon_csv_path = args.horizon_csv_path
        self.window = self.input_window + self.output_window
        self.lake = args.config_name[:-5]
        self.flag_feat = args.flag_feature
        self.y_mean = None
        self.y_std = None
        self.args= args
        self.windowed_dataset_path = ""
        
        if self.target_cols[0] in self.inp_cols:
            self.all_io_cols = self.inp_cols
        else:
            self.all_io_cols = self.inp_cols + self.target_cols
        # print("all_io_cols: ", self.all_io_cols)    
        self.chloro_sub_index = self.all_io_cols.index('Chla_ugL') #index within the all_io_cols
        
        self.chloro_index = 0 #index within the original set of columns
    
#     def train_test_split(self, df, split_type='time', split_date=None, split_ratio=0.0):
#         '''

#         split time series into train/test sets

#         : param df:                     time array
#         : para y:                       feature array
#         : para split:                   percent of data to include in training set
#         : return t_train, y_train:      time/feature training and test sets;
#         :        t_test, y_test:        (shape: [# samples, 1])

#         '''
#         if split_type == 'time':
#             df_train = df[df[self.date_col] <= split_date]
#             df_test = df[df[self.date_col] > split_date]
#             return df_train, df_test
#         else:
#             indx_split = int(split_ratio * df.shape[0])
#             indx_train = np.arange(0, indx_split)
#             indx_test = np.arange(indx_split, df.shape[0])
    
#             df_train = df.iloc[indx_train]
#             df_test = df.iloc[indx_test]

#         return df_train.reset_index(drop='true'), df_test.reset_index(drop='true')
    
    def load_pickle(self, lake_path):
        
        lake_arr = None
        # lake_path = os.path.join(path, lake) + '_' + str(self.window) + '.pkl'
        
        with open(lake_path, 'rb') as pickle_file:
            lake_arr = pickle.load(pickle_file)
        
        return lake_arr
            
            
#     def split_data(self, path, config):
#         train_X = []
#         val_X = []
    
#         train_lake_names = []
#         val_lake_names = []
        
#         for ind, lake in enumerate(config['train_lakes']):
#             lake_arr = self.load_pickle(path, lake)
            
#             if len(lake_arr)==0:
#                 continue
            
#             lake_frac = config['train_fractions'][ind]
#             num_samples = lake_arr.shape[0]

#             end_index = int(num_samples * lake_frac)  # Ending index for the train fraction
#             train_X.append(lake_arr[:end_index, :, :])
            
#             train_lake_names += [lake]*end_index
            
#         for ind, lake in enumerate(config['val_lakes']):
#             lake_arr = self.load_pickle(path, lake)
#             if len(lake_arr)==0:
#                 continue
                
#             lake_frac = config['val_fractions'][ind]
#             num_samples = lake_arr.shape[0]

#             if lake in config['train_lakes']:
#                 if lake_frac + config['train_fractions'][config['train_lakes'].index(lake)] <= 1:
#                     end_index = num_samples - int(num_samples * lake_frac)  # Ending index for the train fraction
#                     val_X.append(lake_arr[end_index:, :, :])
#                 else:
#                     print(f"val frac + train frac for lake = {lake} is greater than 1. So skipping putting it in val set")
#             else:
#                 end_index = num_samples - int(num_samples * lake_frac)  # Ending index for the train fraction
#                 val_X.append(lake_arr[end_index:, :, :])
            
#             val_lake_names += [lake]*(lake_arr.shape[0]-end_index)

#         return np.concatenate(train_X, axis=0), np.concatenate(val_X, axis=0), train_lake_names, val_lake_names
    
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
            # self.feat_mean = tensor.mean(dim=(0, 1))[None, None, :]
            # self.feat_std = tensor.std(dim=(0, 1))[None, None, :]
        
        tensor = (tensor-self.feat_mean)/(self.feat_std+eps)
        return tensor
    
    def less_data(self, mask):
        for i in range(mask.shape[0]):
            window = mask[i]
            total_ = window.shape[0]*window.shape[1]
            if window.sum()/total_ < self.non_null_ratio:
                return True
        return False
        
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
    
    def extract_io(self, data):
        
        start_index = self.input_window
        
        # feat_inds = 1*(np.arange(len(self.inp_cols)) != self.chloro_sub_index)
        # print("feat_inds: ", feat_inds)
        # print("data shape: ", data.shape)
        # print("data [0]: ", data[0, 0, :])

        # print(f"\nself inp cols = {self.inp_cols}\n")
        # print(f"inds of inp cols = {np.arange(len(self.inp_cols))}")
        # Extract the portion from each sample
        X = data[:, :start_index, np.arange(len(self.inp_cols))]
        Y = data[:, start_index:, self.chloro_sub_index].unsqueeze(-1)

        # print("x shape: ", X.shape)
        # print("x [0]: ", X[0, 0, :])
        
        return X, Y
    
    def apply_noise(self, df, mean=0):
        for col in self.inp_cols:
            if col != 'Chla_ugL':
                if self.args.flag_noise == 'add':
                    df[col] += np.random.normal(mean, df[col].std(), df[col].shape)
            # if col == 'Chla_ugL':
            #     if self.args.flag_noise == 'add':
            #         df[col] += np.random.lognormal(mean, df[col].std()* self.args.frac_std_noise, df[col].shape)
            #         df[col] = df[col].clip(lower=0)
        return df
        
    def split_and_window(self, df, config, windowed_dataset_path, split, include_target=True):
        
        train_lake_names = []
        
        train_X = []
        
        train_fractions = config['train_fractions']
        val_fractions = config['val_fractions']
        test_fractions = config['test_fractions']
        
        for ind, lake in enumerate(config[split+'_lakes']):
            '''
            Assuming, we have the same set of val_lakes and test_lakes
            '''
            print(f"Splitting for {split} and for lake = {lake}")
            
            lakename, dtype = lake.split('_')
                
            df_temp = df[(df['Lake']==lakename) & (df['DataType']==dtype)].reset_index(drop=True)
            
            if dtype=='modeled':
                modelruntype="GLMAED_calibrated_observed_met"
                df_temp = df_temp[df_temp['ModelRunType']==modelruntype].reset_index(drop=True)
            
            num_samples = df_temp.shape[0]
            
            train_frac = train_fractions[ind]
            val_frac = val_fractions[ind]
            test_frac = test_fractions[ind]
            
            lakefile = lakename + "_" + dtype + '_' + str(self.window) + '_' + split +'_.pkl'
            lake_path = os.path.join(windowed_dataset_path, lakefile)
            
            train_end_index = int(train_frac*num_samples)
            val_end_index = train_end_index + int(val_frac*num_samples)
            test_end_index = val_end_index + int(test_frac*num_samples)

            if split=='train':
                train_df = df_temp.iloc[:train_end_index, :].reset_index(drop='true')
                # print("original data: ", train_df.loc[100:200, self.inp_cols])
                if self.args.flag_noise != '':
                    train_df = self.apply_noise(train_df)
                # print("noisy data: ", train_df.loc[100:200, self.inp_cols])
                self.train_size = train_df.shape[0]
                self.train_dates = train_df.DateTime.values
                
                
            elif split=='val':
                train_df = df_temp.iloc[train_end_index:val_end_index, :].reset_index(drop='true')
                self.val_size = train_df.shape[0]
                self.val_dates = train_df.DateTime.values
            else:
                train_df = df_temp.iloc[val_end_index:test_end_index, :].reset_index(drop='true')
                self.test_size = train_df.shape[0]
                self.test_dates = train_df.DateTime.values
                    
            if os.path.exists(lake_path):
                print(f"Windowed dataset already exists for lake {lakename}, skipping windowing ...")
                lake_arr = self.load_pickle(lake_path)
                train_X.append(lake_arr)
                train_lake_names += [lake]*lake_arr.shape[0]
                
            else:
                
                lake_x = self.windowed_dataset(train_df)
                print(f"Lake {lake} has got shape = {lake_x.shape}")
                
                if lake_x.shape[0]==0:
                    continue
                
                # with open(lake_path, 'wb') as pickle_file:
                #     pickle.dump(lake_x, pickle_file)
                #     print(f"Pickled lake {lake}")
                
                train_X.append(lake_x)
                train_lake_names += [lake]*lake_x.shape[0]
        
        return np.concatenate(train_X, axis=0), train_lake_names
        
        
    def windowed_dataset(self, df, include_target=True):
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
        num_samples = (L - self.window) // self.stride + 1
        
        X = np.array([]) 
        
        for ii in tqdm(np.arange(num_samples)):
            start_x = self.stride * ii
            end_x = start_x + self.window
            
            subset_dfX = df.iloc[start_x:end_x, :].copy(deep=True).reset_index(drop='true')
            
            nulls_X = subset_dfX.loc[:self.input_window, self.inp_cols].isnull().sum().sum()
            nulls_Y = subset_dfX.iloc[self.input_window:, self.chloro_index].isnull().sum().sum()
            
            # subset_dfX.loc[null_inds, :] = None
            
            non_null_frac_X = (self.input_window*len(self.inp_cols) - nulls_X)/(self.input_window*len(self.inp_cols))
            # non_null_frac_Y = (self.output_window - nulls_Y)/self.output_window
            
            # print(f"non null frac = {non_null_frac}")
            
            # if non_null_frac_X >= self.non_null_ratio and (self.output_window-nulls_Y) >= 1:
            if X.shape[0]==0:
                X = np.expand_dims(subset_dfX.loc[:, self.all_io_cols], axis=0)
            else:
                toAdd = np.expand_dims(subset_dfX.loc[:, self.all_io_cols], axis=0)
                X = np.append(X, toAdd, axis=0)
                                 
        return X
    
    def windowed_dataset_utils(self, df, windowed_dataset_path, split, include_target=True):
        
        '''
        1. Check if we already have windowed dataset for all lakes
        2. If such window dataset exists, skip
        3. Else, performing windowing on all lakes
        '''

        lake_names = []
        lakes = df.Lake.unique()

        for lake in lakes:
            df_temp = df[df['Lake']==lake].reset_index(drop=True)
            dtypes = df_temp.DataType.unique()

            for dtype in dtypes:
                lakename = lake + "_" + dtype + '_' + str(self.window) + '_' + split + '_' + '.pkl'
                lake_path = os.path.join(windowed_dataset_path, lakename)
                
                if os.path.exists(lake_path):
                    print(f"Windowed dataset already exists for lake {lakename}, skipping windowing ...")
                    continue
                
                lake_df = df_temp[df_temp['DataType']==dtype].reset_index(drop='true')
                lake_x = self.windowed_dataset(lake_df)
                
                print(f"Lake {lake} has got shape = {lake_x.shape}")
                                 
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

    def plot_samples(self, df, pred_y, true_y, idx, n_samples, xticks_spacing=False):
        '''
        params:
        
        return:
        '''
    
        input_window = self.input_window
        output_window = self.output_window
        df_un = df.copy(deep=True)
        
        df_un[self.target_cols] = df_un[self.target_cols].apply(lambda l: l*self.y_std+self.y_mean)
        df_un = df_un.reset_index(drop=True)
        
        x_axis_1 = df_un.loc[idx:idx+input_window-1,self.date_col].to_numpy().reshape(-1)
        
        start_x_axis = idx + input_window
        delta = n_samples*output_window - 1 if n_samples*output_window - 1 < (df_un.shape[0]-start_x_axis) else df_un.shape[0]-start_x_axis
        
        end_x_axis = start_x_axis + delta #n_samples*output_window - 1
        
        x_axis_2 = df_un.loc[start_x_axis:end_x_axis,self.date_col].to_numpy().reshape(-1)
            
        x_plot = df_un[self.target_cols].to_numpy()[idx:idx+input_window].reshape(-1)
        
        pred_plot = pred_y[idx:idx+n_samples*output_window:output_window].cpu().numpy().reshape(-1)
        true_plot = true_y[idx:idx+n_samples*output_window:output_window].cpu().numpy().reshape(-1)
    
        fig,ax = plt.subplots()
        
        fig.set_figheight(5)
        fig.set_figwidth(20)

        ax.grid(visible=True, alpha=0.2)
        ax.plot(x_axis_1, x_plot, linestyle='--', label='input_window')
        ax.plot(x_axis_2, pred_plot, linestyle='--', label='predictions')
        ax.plot(x_axis_2, true_plot, linestyle='--', label='true_values')
        ax.set_xlabel('Timeline')
        ax.set_ylabel('Chlorophyll')
        
        if xticks_spacing:
            every_nth = 20
            for n, label in enumerate(ax.xaxis.get_ticklabels()):
                if n % every_nth != 0:
                    label.set_visible(False)

        ax.set_xticklabels(x_axis_1, rotation=90)
        ax.set_xticklabels(x_axis_2, rotation=90)
        plt.legend()

    def plot_train_test_rmse(train_rmse, test_rmse):
        plt.figure(figsize=(5,4), dpi=150)
        plt.plot(train_rmse, lw=2.0, label='train_rmse')
        plt.plot(test_rmse, lw=2.0, label='test_rmse')
        plt.yscale("log")
        plt.grid("on", alpha=0.2)
        plt.legend()
        wandb.log({title: wandb.Image(plt)})
        
    def pred_per_step_helper(self, predictions, idx, pred_values, date):
        '''
        Compute all the predictions for a single date. i.e. as T+1, T+2, T+3, ... T+horizon timestep prediction
        '''
        c = 0
        rind = idx
        while c<self.output_window:
            rind = rind + c
            pred_values[date].append(predictions[rind].reshape(-1)[-(c+1)])
            c+=1
    
        return pred_values

    def prediction_per_step(self, df, predictions, gts, ids):
        pred_values = {}
        gts_ls = {}
        for idx in ids:
            date = df.loc[idx+self.input_window+self.output_window-1,self.date_col]
            pred_values[date] = []
            gts_ls['GT_'+date] = gts[idx].reshape(-1)[-1]
            pred_values = self.pred_per_step_helper(predictions, idx, pred_values, date)
    
        pred_values = {k:list(reversed(v)) for k,v in pred_values.items()}
        for k,v in pred_values.items():
            pred_values[k] = [i.cpu().numpy() for i in v]
            
        return pred_values, gts_ls
    
    def fillpredtable(self, r, table, pred):
        for i,k in enumerate(table.columns):
            if i-(r-1)>=0 and i-(r-1) < pred.shape[0]:
                table.loc[r, k] = pred[i-(r-1)][r-1].cpu().numpy()
        return table
            
    def predictionTable(self, pred_df, split, gt_values=None, plot=True):
        '''
        Create the prediction table
        '''
        if split=='train':
            size = self.train_size
            dates = self.train_dates[self.input_window:]
        elif split=='val':
            size = self.val_size
            dates = self.val_dates[self.input_window:]
        else:
            size = self.test_size
            dates = self.test_dates[self.input_window:]
        
        # print(f"dates = {dates[300:450]}")
        pred_table = np.zeros((self.output_window, size - self.input_window))
        pred_table = pd.DataFrame(pred_table)
        pred_table.columns = dates
        pred_table.index = range(1,self.output_window+1)
        pred_table.loc[:] = np.nan
        
        for r in range(1, self.output_window+1):
            pred_table = self.fillpredtable(r, pred_table, pred_df)
        
        if plot:    
            plot_df = pred_table.iloc[:, self.output_window-1:-self.output_window+1]
            plot_gt_values = gt_values[self.output_window-1:-self.output_window+1]
            return pred_table, plot_df, plot_gt_values
        
        return pred_table

#     def plotTable(self, plot_df, plot_gt, train_or_val):
#         '''
#         Plot the prediction table
#         '''
#         x_plot = plot_df.columns.values
#         x_plot = [str(d).split('T')[0] for d in x_plot]
#         fig,ax = plt.subplots()
        
#         fig.set_figheight(5)
#         fig.set_figwidth(20)
        
#         ax.grid(visible=True, alpha=0.2)
        
#         for t in self.horizon_range:
#             y_axis = plot_df.loc[t,:].values
#             ax.plot(x_plot, y_axis, linestyle='--', label='T+'+str(t))
        
#         # print(f"x_plot = {len(x_plot)}")
#         # print(f"plot_gt = {plot_gt.shape}")
        
#         ax.plot(x_plot, plot_gt, linestyle='--', label='Ground-truth')
#         ax.set_xlabel('Timeline')
#         ax.set_ylabel('Chlorophyll T+n predictions')
            
#         every_nth = 10
#         for n, label in enumerate(ax.xaxis.get_ticklabels()):
#             if n % every_nth != 0:
#                 label.set_visible(False)

#         ax.set_xticklabels(x_plot, rotation=90)
#         plt.legend()
        
#         title = 'T+n Prediction Performance on {} data'.format(train_or_val)
#         plt.tight_layout()
#         plt.title(title, y=1.02)
#         wandb.log({title: wandb.Image(plt)})
#         plt.close()
    
    def plotTable(self, eval_ls, train_or_val, err_std):
        '''
        Plot the prediction table
        '''
        x_plot = eval_ls[0]['plot_table'].columns.values
        
        x_plot = [str(d).split('T')[0] for d in x_plot]
        fig,ax = plt.subplots()
        
        fig.set_figheight(5)
        fig.set_figwidth(20)
        
        ax.grid(visible=True, alpha=0.2)
        
        '''
        get the mean predictions
        '''
        ntrials = len(eval_ls)
        
        stck = []
        for trial in range(ntrials):
            # print(f"plot table for {train_or_val} = {eval_ls[trial]['plot_table']}")
            stck.append(eval_ls[trial]['plot_table'])    
        
        stck = np.array([df.values for df in stck])
        mean_array = np.mean(stck, axis=0)
        std_array = np.std(stck, axis=0)
        
        plot_mean_df = pd.DataFrame(mean_array, columns=x_plot, index=eval_ls[0]['plot_table'].index)
        plot_std_df = pd.DataFrame(std_array, columns=x_plot, index=eval_ls[0]['plot_table'].index)
        
        # print(f"plot_mean df = {plot_mean_df}")
        
        for t in self.horizon_range:
            
            mean_predictions = plot_mean_df.loc[t,:].values
            std_predictions = plot_std_df.loc[t, :].values
            
            # print(f"x_plot being plot is = {x_plot[300:450]}")
            ax.plot(x_plot, mean_predictions, linestyle='--', label='T+'+str(t))
            
            lower_bounds = mean_predictions - 1.96*err_std[t-1]
            upper_bounds = mean_predictions + 1.96*err_std[t-1]
            ax.fill_between(x_plot, lower_bounds, upper_bounds, alpha=0.2, label='T+'+str(t)+' Confidence shading')
        
        # print(f"x_plot = {len(x_plot)}")
        # print(f"plot_gt = {plot_gt.shape}")
        plot_gt = eval_ls[0]['plot_gt_values']
        ax.plot(x_plot, plot_gt, linestyle='--', label='Ground-truth')
        ax.set_xlabel('Timeline')
        ax.set_ylabel('Chlorophyll T+n predictions')
        
        if len(x_plot)>6*365:
            every_nth = 200
        elif len(x_plot)>=3*365:
            every_nth = 100
        else:
            every_nth = 10
        for n, label in enumerate(ax.xaxis.get_ticklabels()):
            if n % every_nth != 0:
                label.set_visible(False)

        ax.set_xticklabels(x_plot, rotation=90)
        plt.legend()
        
        title = 'T+n Prediction Performance on {} data'.format(train_or_val)
        plt.tight_layout()
        plt.title(title, y=1.02)
        wandb.log({title: wandb.Image(plt)})
        plt.close()
    
    def compute_horizon_rmse(self, eval_ls, train_or_val):
        
        rmse_values = []
        std_values = []
        
        ntrials = len(eval_ls)
        
        # stck = []
        gt_values = eval_ls[0]['horizon_gt_values']
#         for trial in range(ntrials):
#             stck.append(eval_ls[trial]['horizon_pred_table'])    
            
#         stck = np.array([df.values for df in stck])
#         mean_array = np.nanmean(stck, axis=0)
#         std_array = np.nanstd(stck, axis=0)
        
#         T_pred_table_mean = pd.DataFrame(mean_array, columns=x_plot, index=eval_ls[0]['plot_table'].index)
#         plot_std_df = pd.DataFrame(std_array, columns=x_plot, index=eval_ls[0]['plot_table'].index)
        
        for i in range(self.output_window):
            rmse_trial = []
            for trial in range(ntrials):
                T_pred_table = eval_ls[trial]['horizon_pred_table']
                # plot_table = eval_ls[trial]['plot_table']
                rmse_trial.append(self.compute_rmse(i, T_pred_table, gt_values))
                # rmse_trial.append(self.compute_rmse(i, plot_table, gt_values))
            rmse_trial = np.array(rmse_trial)
            rmse_values.append((rmse_trial.mean(), rmse_trial.std()))
        
        rmse_values = pd.DataFrame(rmse_values, columns=['RMSE', 'STD'], index=range(1,self.output_window+1))
        if self.flag_feat==[]:
            config = 'default'
        else:
            config = '_'.join(self.flag_feat)
            
        filename = '{}_{}_days_{}_{}.csv'.format(train_or_val, self.output_window, self.lake, self.args.run_name.partition('_2')[0])
        path = os.path.join(self.horizon_csv_path, filename)
        rmse_values.to_csv(path, index=False)

        y = rmse_values.RMSE.values.tolist()
        y_std = rmse_values.STD.values.tolist()
        x = rmse_values.index.tolist()
        xlabel = ['T+'+str(i) for i in rmse_values.index]
        fig,ax = plt.subplots()
        
        fig.set_figheight(4)
        fig.set_figwidth(15)
        
        ax.grid(visible=True, alpha=0.2)
        ax.plot(x, y, linestyle='-', label='RMSE')
        ax.errorbar(x, y, yerr=y_std, label='Error Bars', fmt='o', color='green', alpha=0.5, capsize=5)
        
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel, rotation=90)
        plt.legend()
        
        title = '{} data: Varying RMSE error w.r.t horizon window'.format(train_or_val)
        plt.tight_layout()
        plt.title(title,y=1.02)
        plt.xlabel('Horizon Window')
        plt.ylabel('Root Mean Squared Error')
        
        wandb.log({title: wandb.Image(plt)})
        plt.close()
        
        return rmse_values
    
    def compute_rmse(self, i, ptable, gt_values):
    
        tk = ptable.iloc[i,:].values
        null_inds = np.where(np.isnan(tk))[0]
        mask = np.ones(gt_values.shape)
        mask[null_inds]= 0
        tk = np.nan_to_num(tk)
        
        unreduced_loss = (tk-gt_values)**2
        unreduced_loss = (unreduced_loss * mask).sum()
        
        non_zero_elements = mask.sum()
        loss = unreduced_loss / non_zero_elements
        
        rmse = loss**0.5
        # print(f"rmse = {rmse}")
        return rmse