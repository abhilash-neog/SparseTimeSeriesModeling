import random
import numpy as np
import pandas as pd
from tqdm import trange, tqdm
import torch
import torch.nn as nn
from torch import optim
from datetime import timedelta, datetime
import copy
import wandb
import math
import matplotlib.pyplot as plt

wandb.login()

from utils import Utils

    
class encoder(nn.Module):
    ''' Encodes time-series sequence '''

    def __init__(self, input_size, hidden_size, num_layers=1, model_type='LSTM', dropout=0.0):

        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.dropout = dropout

        # define LSTM/GRU/RNN layer
        f = getattr(nn, self.model_type)
        self.model = f(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True, dropout=dropout)

    def forward(self, x_input):

        '''
        : param x_input:               input of shape (# in batch, seq_len, input_size)
        : return lstm_out, hidden:     lstm_out gives all the hidden states in the sequence;
        :                              hidden gives the hidden state and cell state for the last
        :                              element in the sequence
        '''
        
        lstm_out, self.hidden = self.model(x_input.view(x_input.shape[0], x_input.shape[1], self.input_size))

        return lstm_out, self.hidden

    def init_hidden(self, batch_size):

        '''
        initialize hidden state
        : param batch_size:    x_input.shape[0]
        : return:              zeroed hidden state and cell state
        '''
        if self.model_type == 'LSTM':
            return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                    torch.zeros(self.num_layers, batch_size, self.hidden_size))
        else:
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)


class decoder(nn.Module):
    ''' Decodes hidden state output by encoder '''

    def __init__(self, input_size, hidden_size, num_layers=1, model_type='LSTM', dropout=0.0):
        '''
        : param input_size:     the number of features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of recurrent layers (i.e., 2 means there are
        :                       2 stacked LSTMs)
        '''

        super(decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model_type = model_type
        self.dropout = dropout

        # define LSTM/GRU/RNN layer
        f = getattr(nn, self.model_type)
        self.model = f(input_size=input_size, hidden_size=hidden_size,
                       num_layers=num_layers, batch_first=True, dropout=dropout)

        # TODO: predict mean and max
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self, x_input, encoder_hidden_states):
        '''
        : param x_input:                    should be 2D (batch_size, input_size)
        : param encoder_hidden_states:      hidden states
        : return output, hidden:            output gives all the hidden states in the sequence;
        :                                   hidden gives the hidden state and cell state for the last
        :                                   element in the sequence

        '''
        lstm_out, self.hidden = self.model(x_input.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(1))

        return output, self.hidden

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, features, target, Mx, My):
        'Initialization'
        self.features = features
        self.target = target
        self.mask_x = Mx
        self.mask_y = My

    def __len__(self):
        'Denotes the total number of samples'
        return self.features.__len__()

    def __getitem__(self, index):
        'Generates one sample of data'
        X = self.features[index]
        y = self.target[index]
        mask_y = self.mask_y[index]
        mask_x = self.mask_x[index]

        return X, y, mask_x, mask_y

class EarlyStopping:
    
    def __init__(self, thres=2, min_delta=0):
        
        self.thres = thres
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            
            if self.counter >= self.thres:
                return True
        else:
            self.counter -= 1
            if self.counter < 0:
                self.counter = 0

        return False


class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, out_path, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        self.out_path = out_path
        
    def __call__(
        self, current_valid_loss, model, epoch, optimizer, criterion):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save(model.state_dict(), self.out_path)

def init_wandb(args, task_name):
    wandb.init(project=args['project_name'], 
               name="_".join([task_name, args['run_name']]), 
               config=args, 
               save_code=args['save_code'])
    config = wandb.config
    return config

class seq2seq(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, input_size,
                 utils,
                 args):

        '''
        : param input_size:     the number of expected features in the input X
        : param hidden_size:    the number of features in the hidden state h
        : param num_layers:     number of lstm in both encoder and decoder
        '''

        super(seq2seq, self).__init__()

        self.input_size = input_size
        self.hidden_size = args.hidden_feature_size
        self.num_layers = args.num_layers
        self.model_type = args.model_type
        self.output_size = args.output_size
        self.device = args.device
        self.dropout = args.dropout
        self.embed_dim = args.embed_dim
        self.num_heads = args.num_heads

        self.encoder = encoder(input_size=self.input_size, hidden_size=self.hidden_size, num_layers=self.num_layers, model_type=self.model_type, dropout=self.dropout).to(self.device)
        self.decoder = decoder(input_size=self.output_size, hidden_size=self.hidden_size, num_layers=self.num_layers, model_type=self.model_type, dropout=self.dropout).to(self.device)
        
        self.encoder_init = copy.deepcopy(self.encoder)
        self.decoder_init = copy.deepcopy(self.decoder)
        
        self.utils = utils
        
        # self.var_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        # self.mhca = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)
        
        # self.mask_embed = FeatEmbed(input_dim=self.input_size,
        #                             embedding_dim=self.embed_dim)
        
#     def cross_attention(self, x, m):
        
#         batch_size, window_size, num_feat, d = x.shape
#         var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
#         x = x.view(-1, num_feat, d)
#         m_ = copy.deepcopy(m.view(-1, num_feat))
        
#         attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
#         attn_out = attn_out.view(batch_size, window_size, d)
        
#         return attn_out
    
    def forward(self, input_batch, mask, target_batch=None):
        
        # outputs tensor
        outputs = torch.zeros(input_batch.shape[0], self.target_len, self.output_size, device=self.device)

#         # Variable embedding
#         input_batch = self.mask_embed(input_batch)
        
#         # Cross attention
#         input_batch = self.cross_attention(input_batch, mask)
        
        # encoder outputs
        encoder_output, encoder_hidden = self.encoder(input_batch)

        # decoder with teacher forcing
        # TODO: first input to decoder - shape: (batch_size, input_size)
        # print(f"previous = {torch.zeros([input_batch.shape[0], self.output_size], device=self.device).shape}")
        # decoder_input = input_batch[:, -1, self.utils.chloro_sub_index].unsqueeze(-1)
        decoder_input = torch.zeros([input_batch.shape[0], self.output_size], device=self.device)  # shape: (batch_size, input_size)
        # print(f"shape = {decoder_input.shape}")
        decoder_hidden = encoder_hidden
        
        if self.training:
            if self.training_prediction == 'recursive':
                # predict recursively
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:,t,:] = decoder_output
                    decoder_input = decoder_output

            if self.training_prediction == 'teacher_forcing':
                # use teacher forcing
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[:,t,:] = decoder_output
                        decoder_input = target_batch[:, t, :]

                # predict recursively
                else:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[:,t,:] = decoder_output
                        decoder_input = decoder_output

            if self.training_prediction == 'mixed_teacher_forcing':
                # predict using mixed teacher forcing
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:,t,:] = decoder_output

                    # predict with teacher forcing
                    if random.random() < self.teacher_forcing_ratio:
                        decoder_input = target_batch[:, t, :]

                    # predict recursively
                    else:
                        decoder_input = decoder_output
                        
        else:
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:,t,:] = decoder_output
                decoder_input = decoder_output
            
        out_cloned = outputs.clone()
        out_cloned[:, :, 0] = torch.clamp(outputs[:, :, 0], min=self.alpha)
        
        return out_cloned
    
    def compute_loss(self, target, pred, mask):

        loss = (pred - target) ** 2
        loss = (loss * mask).sum() / mask.sum()
        
        return loss
        
    def train_model(self,
                    X_train, 
                    Y_train, 
                    X_test, 
                    Y_test,
                    train_lake_names,
                    args,
                    utils,
                    X_val=None,
                    Y_val=None,
                    val_lake_names=None):

        '''
        train lstm encoder-decoder

        : param X_train:              input data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param Y_train:             target data with shape (seq_len, # in batch, number features); PyTorch tensor
        : param n_epochs:                  number of epochs
        : param target_len:                number of values to predict. Time horizon
        : param batch_size:                number of samples per gradient update
        : param training_prediction:       type of prediction to make during training ('recursive', 'teacher_forcing', or
        :                                  'mixed_teacher_forcing'); default is 'recursive'
        : param teacher_forcing_ratio:     float [0, 1) indicating how much teacher forcing to use when
        :                                  training_prediction = 'teacher_forcing.' For each batch in training, we generate a random
        :                                  number. If the random number is less than teacher_forcing_ratio, we use teacher forcing.
        :                                  Otherwise, we predict recursively. If teacher_forcing_ratio = 1, we train only using
        :                                  teacher forcing.
        : param learning_rate:             float >= 0; learning rate
        : param dynamic_tf:                use dynamic teacher forcing (True/False); dynamic teacher forcing
        :                                  reduces the amount of teacher forcing for each epoch
        : return losses:                   array of loss function for each epoch
        '''
        
        project_name = args['project_name']
        run_name = args['run_name']
        self.task_name = args['task_name']
        self.training_prediction = args['training_prediction']
        self.teacher_forcing_ratio = args['teacher_forcing_ratio']
        self.learning_rate = args['learning_rate']
        self.target_len = args['horizon_window']
        self.dynamic_tf = args['dynamic_tf']
        self.batch_size = args['batch_size']
        self.batch_shuffle = args['batch_shuffle']
        self.output_size = args['output_size']
        
        config = init_wandb(args, self.task_name)
        
        n_epochs = config.max_epochs
        
        # initialize array of losses
        losses = np.full(n_epochs, np.nan)
        val_rmse = []
        train_rmse = []
        test_rmse = []
        
        # n_batches = int(math.ceil(X_train.shape[0] / config.batch_size))
        early_stop = config.early_stop
        early_stopper = EarlyStopping(thres=config.early_stop_thres, min_delta=config.early_stop_delta)
        
        params = {
                  'batch_size': config.batch_size,
                  'shuffle': config.batch_shuffle
                }
        
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        if X_val is not None:
            X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        '''
        Training generator
        '''
        M_x = 1 - (1 * (torch.isnan(X_train)))
        M_x = M_x.float().to(self.device)
        
        M_y = 1 - (1 * (torch.isnan(Y_train)))
        M_y = M_y.float().to(self.device)
        
        X_train = torch.nan_to_num(X_train)
        Y_train = torch.nan_to_num(Y_train)
        
        training_set = Dataset(X_train, Y_train, M_x, M_y)
        training_generator = torch.utils.data.DataLoader(training_set, **params)
        
        '''
        Validation generator
        '''
        if X_val is not None:
            M_x_val = 1 - (1 * (torch.isnan(X_val)))
            M_x_val = M_x_val.float().to(self.device)

            M_y_val = 1 - (1 * (torch.isnan(Y_val)))
            M_y_val = M_y_val.float().to(self.device)

            X_val = torch.nan_to_num(X_val)
            Y_val = torch.nan_to_num(Y_val)

            validation_set = Dataset(X_val, Y_val, M_x_val, M_y_val)
            validation_generator = torch.utils.data.DataLoader(validation_set, **params)

        '''
        Test generator
        '''
        M_x_test = 1 - (1 * (torch.isnan(X_test)))
        M_x_test = M_x_test.float().to(self.device)
        
        M_y_test = 1 - (1 * (torch.isnan(Y_test)))
        M_y_test = M_y_test.float().to(self.device)
        
        X_test = torch.nan_to_num(X_test)
        Y_test = torch.nan_to_num(Y_test)
        
        testing_set = Dataset(X_test, Y_test, M_x_test, M_y_test)
        testing_generator = torch.utils.data.DataLoader(testing_set, **params)
        
        n_batches = len(training_generator)
        
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=config.weight_decay)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
                                                        max_lr=config.max_lr, 
                                                        epochs=n_epochs, 
                                                        div_factor=config.div_factor, 
                                                        pct_start=config.pct_start, 
                                                        anneal_strategy=config.anneal_strategy, 
                                                        final_div_factor=config.final_div_factor,
                                                        steps_per_epoch=n_batches, 
                                                        verbose=False)
        
        # get chloro bound
        self.std_chloro = utils.feat_std[:, :, utils.chloro_sub_index].unsqueeze(1).to(self.device) 
        self.mean_chloro = utils.feat_mean[:, :, utils.chloro_sub_index].unsqueeze(1).to(self.device) 
        self.alpha = (-self.mean_chloro/self.std_chloro)[:, :, 0]
        # print(f"Mean chloro = {self.mean_chloro}")
        # print(f"Std chloro = {self.std_chloro}")
        # print(f"All std = {utils.feat_std}")
        # print(f"All mean = {utils.feat_mean}")
        # exit(0)
        
        
        wandb.watch(self.encoder)
        
        self.to(self.device)
        
        torch.autograd.set_detect_anomaly(True)
        
        
        with trange(n_epochs) as tr:
            for it in tr:

                batch_loss = 0.
                batch_loss_tf = 0.
                batch_loss_no_tf = 0.
                num_tf = 0
                num_no_tf = 0
                # batch_test_loss = np.nan
                
                encoder_hidden = self.encoder.init_hidden(config.batch_size)
                
                self.train()
                
                for input_batch, target_batch, maskX, maskY in tqdm(training_generator):
                    
                    # if all values are missing, skip it
                    if maskY.eq(0).all() or self.utils.less_data(maskX):
                        continue
                    
                    # zero the gradient
                    optimizer.zero_grad()
                    
                    outputs = self(input_batch=input_batch, mask=maskX, target_batch=target_batch)
                    
                    # compute the loss
                    loss = self.compute_loss(target_batch, outputs, maskY)
                    # loss = criterion(outputs, target_batch)
                    # print(f"loss item = {loss.item()}")
                    batch_loss += loss.item()

                    # backpropagation
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # loss for epoch
                batch_loss /= n_batches
                losses[it] = batch_loss

                # dynamic teacher forcing
                if self.dynamic_tf and self.teacher_forcing_ratio > 0:
                    self.teacher_forcing_ratio = self.teacher_forcing_ratio - 0.002

                if it % config.eval_freq == 0:
                    with torch.no_grad():
                        self.eval()
                        # print(f"ckpt 1 \n")
                        if X_val is not None:
                            val_eval_dict = self.evaluate_batch(test_generator=validation_generator)
                            batch_val_loss = val_eval_dict["rmse"].item()
                            val_rmse.append(batch_val_loss)
                        # print(f"ckpt 2 \n")
                        train_eval_dict = self.evaluate_batch(test_generator=training_generator)
                        test_eval_dict = self.evaluate_batch(test_generator=testing_generator)
                        
                        batch_train_loss = train_eval_dict["rmse"].item()
                        batch_test_loss = test_eval_dict["rmse"].item()
                        
                        train_rmse.append(batch_train_loss)
                        test_rmse.append(batch_test_loss)
                    # if early_stop and early_stopper.early_stop(batch_val_loss):
                    #     print("Early stopping")
                    #     break
                # progress bar
                if X_val is not None:
                    metrics = {
                        "loss":batch_loss,
                        "val_rmse":batch_val_loss,
                        "train_rmse":batch_train_loss
                        }
                else:
                    metrics = {
                        "loss":batch_loss,
                        "train_rmse":batch_train_loss,
                        "test_rmse":batch_test_loss
                        }
                # tr.set_postfix(loss="{0:.3e}".format(batch_loss))
                tr.set_postfix(metrics)
                wandb.log(metrics)
        
        with torch.no_grad():
            
            # print(f"ckpt 3 \n")
            if X_val is not None:
                val_eval_dict = self.evaluate_and_plot(test_generator=validation_generator, split='val')
                wandb.summary['val_rmse'] = val_eval_dict["rmse"].item()
                
            # print(f"ckpt 4 \n")
            train_eval_dict = self.evaluate_and_plot(test_generator=training_generator, split='train')
            # print(f"ckpt 5 \n")
            test_eval_dict = self.evaluate_and_plot(test_generator=testing_generator, split='test')
            
            wandb.summary['train_rmse'] = train_eval_dict["rmse"].item()
            wandb.summary['test_rmse'] = test_eval_dict["rmse"].item()
            wandb.finish()

        # if X_val is not None:
        #     return train_eval_dict, val_eval_dict, test_eval_dict
        # else:
        return train_eval_dict, test_eval_dict

    def predict_batch(self, testing_generator, target_len):
        '''
        : param input_tensor:      input data (batch, seq_len, input_size); PyTorch tensor
        : param target_len:        number of target values to predict (30)
        : return np_outputs:       np.array containing predicted values; prediction done recursively
        '''
        
#         M_x = 1 - (1*torch.isnan(input_tensor))
#         M_x = M_x.float().to(self.device)
        
#         M_y = 1 - (1*torch.isnan(output_tensor))
#         M_y = M_y.float().to(self.device)
        
#         input_tensor = torch.nan_to_num(input_tensor)
#         output_tensor = torch.nan_to_num(output_tensor)
        
#         testing_set = Dataset(input_tensor, output_tensor, M_x, M_y)
        
#         testing_generator = torch.utils.data.DataLoader(testing_set, batch_size=self.batch_size, shuffle=self.batch_shuffle)
        
        eval_outputs = []
        eval_masks = []
        target_samples = []
        
        for input_batch, target_batch, maskX, maskY in tqdm(testing_generator):
            
            with torch.cuda.amp.autocast():
                
                if not maskY.eq(0).all():
                    outputs = self(input_batch, maskX)
                else:
                    # outputs = torch.ones_like(maskY)*self.alpha
                    # outputs = torch.full_like(maskY, np.nan)
                    outputs = torch.zeros_like(maskY)
                    # target_batch = torch.full_like(target_batch, np.nan)
                
                eval_outputs.append(outputs)
                eval_masks.append(maskY)
                target_samples.append(target_batch)
                
        # np_outputs = outputs.detach()
        eval_outputs = torch.cat(eval_outputs, dim=0)
        eval_masks = torch.cat(eval_masks, dim=0)
        target_samples = torch.cat(target_samples, dim=0)
        
        return eval_outputs, eval_masks, target_samples
    
#     def predict_batch(self, input_tensor, target_len):
#         '''
#         : param input_tensor:      input data (batch, seq_len, input_size); PyTorch tensor
#         : param target_len:        number of target values to predict (30)
#         : return np_outputs:       np.array containing predicted values; prediction done recursively
#         '''
#         batch_size = input_tensor.shape[0]
        
#         encoder_output, encoder_hidden = self.encoder(input_tensor)
        
#         outputs = torch.zeros(batch_size, target_len, self.output_size, device=self.device)  # input_tensor.shape[2])
        
#         decoder_input = torch.zeros(batch_size, self.output_size, device=self.device)  # input_tensor[-1, :, :]%%!
#         decoder_hidden = encoder_hidden
        
#         for t in range(target_len):
#             decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
#             outputs[:,t,:] = decoder_output
#             decoder_input = decoder_output
    
#         np_outputs = outputs.detach()
#         return np_outputs
    
    def evaluate_batch(self, test_generator, unnorm=True):
    
        y_pred, y_masks, Y_test = self.predict_batch(test_generator, self.utils.output_window)
        
        # print(f"shape of mask = {y_masks.shape}")
        # print(f"y_test = {Y_test}")
        
        if unnorm:
            # unnormalize the data
            y_pred = y_pred*self.std_chloro + self.mean_chloro
            Y_test = Y_test*self.std_chloro + self.mean_chloro
        
        # print(f"y_pred = {y_pred}")
        # print(f"y_masks sum = {y_masks.sum()}")
        # exit(0)
        
        sqred_err = (y_pred-Y_test)**2#.mean())**0.5
        
        rmse = ((sqred_err * y_masks).sum() / y_masks.sum())**0.5
        
        evaluate_dict = {
            "y_pred":y_pred,
            "y_true":Y_test,
            "rmse":rmse,
            "mask": y_masks
        }
        return evaluate_dict
    
    def evaluate_and_plot(self, test_generator, split):
        
        eval_dict = self.evaluate_batch(test_generator=test_generator)
        
        masks = eval_dict['mask']
        gt_ = eval_dict['y_true']
        pred = eval_dict['y_pred'] # model predicted values
        
        pred_plot = copy.deepcopy(pred)
        gt_plot = copy.deepcopy(gt_)
        
        for i in range(masks.shape[0]):
            pred_plot[i][masks[i, :, 0]==0] = np.nan
            gt_plot[i][masks[i, :, 0]==0] = np.nan
        
        gt_df = pd.DataFrame(gt_plot.cpu().numpy()[:,:,0])
        
        gt_values = np.append(gt_df[0].values, gt_df.iloc[-1,1:]) # ground-truth values
        
        T_pred_table, plot_df, plot_gt_values = self.utils.predictionTable(pred_df=pred_plot, gt_values=gt_values, split=split)

        eval_dict['plot_table'] = plot_df
        eval_dict['plot_gt_values'] = plot_gt_values
        # self.utils.plotTable(plot_df=plot_df, plot_gt=plot_gt_values, train_or_val=split)
        
        # print(f"T_Pred_table = {T_pred_table}")
        
        '''
        compute rmse
        '''
        gt_df = pd.DataFrame(gt_.cpu().numpy()[:,:,0])
        gt_values = np.append(gt_df[0].values, gt_df.iloc[-1,1:]) # ground-truth values
        
        predtable, _, _ = self.utils.predictionTable(pred_df=pred, gt_values=gt_values, split=split)
        eval_dict['horizon_pred_table'] = predtable
        eval_dict['horizon_gt_values'] = gt_values
        
        # rmse_values = self.utils.compute_horizon_rmse(T_pred_table=predtable, gt_values=gt_values, train_or_val=split)
        
        return eval_dict
        
    
    def evaluate_uncertainty(self, args, eval_dict, train_or_val):
        
        rmse_values=self.utils.compute_horizon_rmse(eval_dict, train_or_val)
        err_std = rmse_values.STD.values
        self.utils.plotTable(eval_dict, train_or_val, err_std)
        
        rmses = []
        for trial in range(len(eval_dict)):
            rmses.append(eval_dict[trial]['rmse'].item())
        rmses = np.array(rmses)
        wandb.summary[train_or_val+'_rmse'] = rmses.mean()
        wandb.summary[train_or_val+'_rmse_std'] = rmses.std()
    
    def perform_evaluation(self, X_train, Y_train, X_test, Y_test, args, utils):
        
        project_name = args['project_name']
        run_name = args['run_name']
        self.task_name = args['task_name']
        self.training_prediction = args['training_prediction']
        self.teacher_forcing_ratio = args['teacher_forcing_ratio']
        self.learning_rate = args['learning_rate']
        self.target_len = args['horizon_window']
        self.dynamic_tf = args['dynamic_tf']
        self.batch_size = args['batch_size']
        self.batch_shuffle = args['batch_shuffle']
        self.output_size = args['output_size']
        self.utils = utils
        
        config = init_wandb(args, self.task_name)
        
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_test, Y_test = X_test.to(self.device), Y_test.to(self.device)
        
        self.std_chloro = utils.feat_std[:, :, utils.chloro_sub_index].unsqueeze(1).to(self.device) 
        self.mean_chloro = utils.feat_mean[:, :, utils.chloro_sub_index].unsqueeze(1).to(self.device) 
        self.alpha = (-self.mean_chloro/self.std_chloro)[:, :, 0]
        
        params = {
                  'batch_size': config.batch_size,
                  'shuffle': config.batch_shuffle
                }
        
        self.to(self.device)
        
        torch.autograd.set_detect_anomaly(True)
        
        with torch.no_grad():
                
            self.eval()
            test_eval_dict = self.evaluate_batch(X_test=X_test, Y_test=Y_test)
            train_eval_dict = self.evaluate_batch(X_test=X_train, Y_test=Y_train)

            self.utils.plot_train_test_rmse(train_eval_dict["rmse"].item(), test_eval_dict["rmse"].item())
            
            wandb.summary['val_rmse'] = test_eval_dict["rmse"].item()
            wandb.summary['train_rmse'] = train_eval_dict["rmse"].item()
            wandb.finish()

        
    def plot_err_win(self, X_test=None, Y_test=None, unnorm=True):
    
        y_pred = self.predict_batch(X_test, self.utils.output_window)
        
        if unnorm:
            y_pred = y_pred*self.utils.y_std + self.utils.y_mean
            Y_test = Y_test*self.utils.y_std + self.utils.y_mean
        
        rmse = (((y_pred-Y_test)**2).mean())**0.5
        
        err_vs_ws = []
    
        for ws in range(1, self.utils.output_window+1):
            #err_vs_ws.append((((y_pred[:, :ws, :]-Y_test[:, :ws, :])**2).mean())**0.5)
            err_vs_ws.append(((((y_pred[:, :ws, :]-Y_test[:, :ws, :])**2).mean(axis=1))**0.5).mean())
        
        plt.figure(figsize=(20,4), dpi=150)
        plt.grid("on", alpha=0.5)
        plt.plot(list(range(1,self.utils.output_window+1)), [i.cpu().numpy() for i in err_vs_ws])
        plt.xlabel("Window size")
        plt.ylabel("RMSE")
        plt.show()