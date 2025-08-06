import sys
# sys.path.insert(1, './utils/.')

import math
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import wandb
import sys
import os
import copy
import time
import matplotlib.pyplot as plt

from model import MaskedAutoencoder
from collections import OrderedDict
from functools import partial
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler
from torch.optim import lr_scheduler
import torch.nn.functional as F
from timm.models.vision_transformer import Block
from utils import MaskEmbed, MAEDataset_PT, MAEDataset_FT, NativeScaler, get_1d_sincos_pos_embed, ActiveEmbed, FeatEmbed, adjust_learning_rate, cal_classification_metrics, get_class_weights
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D
from tools import EarlyStopping, adjust_learning_rate, transfer_weights

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score, balanced_accuracy_score
from data_provider.clf_dataloader import data_generator


# from ptflops import get_model_complexity_info

eps = 1e-6


def init_wandb(args, task_name):
    wandb.init(project=args['project_name'], 
               name="_".join([task_name, args['run_name']]), 
               config=args, 
               save_code=args['save_code'])
    config = wandb.config
    return config

class ModelPlugins():
    
    def __init__(self, 
                 window_len, 
                 enc_embed_dim,
                 dec_embed_dim,
                 task_name,
                 num_feats,
                 batch_size,
                 device):
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.window_len = window_len 
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.window_len + 1, self.enc_embed_dim), requires_grad=False).to(self.device)
        self.pos_embed = PositionalEncoding2D(enc_embed_dim).to(self.device)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.window_len + 1, self.dec_embed_dim), requires_grad=False).to(self.device)
        # self.decoder_pos_embed = PositionalEncoding2D(dec_embed_dim).to(self.device)
        
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        
        enc_z = torch.rand((1, self.window_len + 1, self.num_feats, self.enc_embed_dim)).to(self.device) # +1 for the cls token
        self.pos_embed = self.pos_embed(enc_z)
        
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.window_len, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.window_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

class ModelWrapper(nn.Module):
    def __init__(self, model, mask_original, mpl):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.mask_original = mask_original
        self.mpl = mpl

    def forward(self, input_tensor):
        # Forward pass through the original model with additional arguments
        return self.model(input_tensor, self.mask_original, self.mpl)

class Trainer():
    
    def __init__(self, args, model):
        self.args = args
        self.model = model.to(args.device)
        self.batch_size = self.args.batch_size
        self.accum_iter = self.args.accum_iter
        self.weight_decay = self.args.weight_decay
        self.lr = self.args.lr
        # self.train_epochs = self.args.train_epochs
        self.pct_start = self.args.pct_start
        self.device = self.args.device
        self.eval_freq = self.args.eval_freq
        self.seq_len = self.args.seq_len
        self.task_name = self.args.task_name
        self.window_len = self.seq_len
        self.finetune_checkpoints_dir = args.finetune_checkpoints_dir
        # self.configs = self.model.data_config
        self.trial = args.trial
        
        # Load datasets
        self.sourcedata_path = os.path.join(args.clf_data_path, args.pretrain_dataset)
        self.targetdata_path = os.path.join(args.clf_data_path, args.target_dataset)
        
        self.mpl = ModelPlugins(window_len=self.window_len, 
                                enc_embed_dim=self.model.embed_dim, 
                                dec_embed_dim=self.model.decoder_embed_dim,
                                num_feats=self.model.num_feats, 
                                task_name=self.task_name, 
                                batch_size=self.batch_size,
                                device=self.device)
        
    def freeze_encoder_model(self):
        
        for param in self.model.encoder_blocks.parameters():
            param.requires_grad = False
        for param in self.model.mask_embed.parameters():
            param.requires_grad = False
        for param in self.model.mhca.parameters():
            param.requires_grad = False
                
        print(f"Encoder Blocks Frozen!")
        
    def select_optimizer_(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)#, betas=(0.9, 0.95), weight_decay=self.weight_decay)
        return optimizer
        
    def select_scaler_(self):
        loss_scaler = NativeScaler()
        return loss_scaler
    
    def get_data(self, X, Y, split_flag):
        
        M = 1 - (1 * (torch.isnan(X)))
        M = M.float()
        
        X = torch.nan_to_num(X)
        
        '''
        Dataloader
        '''
        if split_flag=='test':
            dataset = MAEDataset_FT(X, Y, M)
            dataloader = DataLoader(
                dataset,
                shuffle=False,
                batch_size=self.batch_size
            )
        else:
            dataset = MAEDataset_FT(X, Y, M)
            dataloader = DataLoader(
                dataset, sampler=RandomSampler(dataset),
                batch_size=self.batch_size,
            )
        return dataloader

    def get_data_PT(self, X, split_flag):
        
        M = 1 - (1 * (torch.isnan(X)))
        M = M.float()
        
        X = torch.nan_to_num(X)
        
        '''
        Dataloader
        '''
        dataset = MAEDataset_PT(X, M)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size)
        return dataloader

    def get_data_FT(self, X, Y, split_flag):
        
        M = 1 - (1 * (torch.isnan(X)))
        M = M.float()
        
        X = torch.nan_to_num(X)
        
        '''
        Dataloader
        '''
        dataset = MAEDataset_FT(X, Y, M)
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.batch_size)
        return dataloader
    
    def val_one_epoch(self, dataloader, split, masked_penalize):
        
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        predictions = []
        gt = []
        self.model.eval()
        for iteration, (X, mask_original) in tqdm(enumerate(dataloader)):
            
            mask_original = 1 - (1 * (torch.isnan(X)))
            mask_original = mask_original.float().to(self.device)
            X = torch.nan_to_num(X).float().to(self.device)

            pred, mask, nask = self.model(X, mask_original, self.mpl)

            loss, masked_loss, unmasked_loss = self.model.forward_loss(data=X, 
                                                                       pred=pred, 
                                                                       mask=mask, 
                                                                       nask=nask, 
                                                                       miss_idx=mask_original, 
                                                                       masked_penalize=masked_penalize)

            if masked_loss is not None:
                masked_loss_value = masked_loss.item()
                unmasked_loss_value = unmasked_loss.item()

                masked_batch_loss += masked_loss_value
                unmasked_batch_loss += unmasked_loss_value

            loss_value = loss.item()
            batch_loss += loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            
            predictions.append(pred.detach().cpu().numpy())
            gt.append(X.detach().cpu().numpy())
        
        batch_loss /= len(dataloader)
        predictions = np.concatenate(predictions, axis=0)
        original = np.concatenate(gt, axis=0)

        if masked_loss is not None:
            masked_batch_loss /= len(dataloader)
            unmasked_batch_loss /= len(dataloader)
        
        self.model.train()
        return batch_loss, masked_batch_loss, unmasked_batch_loss, predictions, original
        
    def train_one_epoch(self, dataloader, split, masked_penalize, optimizer, scheduler, computed_flops):
        
        batch_loss, masked_batch_loss, unmasked_batch_loss = 0, 0, 0
        
        optimizer.zero_grad()
        
        self.model.train()

        for iteration, (X, mask_original) in tqdm(enumerate(dataloader)):
            
            mask_original = 1 - (1 * (torch.isnan(X)))
            mask_original = mask_original.float().to(self.device)
            X = torch.nan_to_num(X).float().to(self.device)
            # print(f"shape of X = {X.shape}")

#             if not computed_flops:
#                 # Create a wrapped model instance
#                 model_wrapper = ModelWrapper(self.model, mask_original, self.mpl)
                
#                 # Compute FLOPs for the first batch
#                 input_tensor = X  # Use the first batch input for FLOPs calculation
#                 input_shape = tuple(input_tensor.shape[1:])  # Ensure it's explicitly a tuple (channels, seq_len)

#                 # Now pass the wrapped model to get_model_complexity_info
#                 print("input_shape: ", input_shape)
#                 flops, params = get_model_complexity_info(
#                     model_wrapper,
#                     input_res=input_shape,  # Pass the input shape as a tuple
#                     as_strings=True,
#                     print_per_layer_stat=False
#                 )
#                 print(f'FLOPs for this model: {flops}')
#                 print(f'Parameters for this model: {params}')
#                 computed_flops = True

            pred, mask, nask = self.model(X, mask_original, self.mpl)
            
            loss, masked_loss, unmasked_loss = self.model.forward_loss(data=X, 
                                                                       pred=pred, 
                                                                       miss_idx=mask_original,
                                                                       mask=mask, 
                                                                       nask=nask,
                                                                       masked_penalize=masked_penalize)
            if masked_loss is not None:
                masked_loss_value = masked_loss.item()
                unmasked_loss_value = unmasked_loss.item()

                masked_batch_loss += masked_loss_value
                unmasked_batch_loss += unmasked_loss_value

            loss_value = loss.item()
            batch_loss += loss_value

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            loss /= self.accum_iter
            
            loss.backward()
            optimizer.step()

            # we use a per iteration (instead of per epoch) lr scheduler
            if (iteration + 1) % self.accum_iter == 0:
                optimizer.zero_grad()

        scheduler.step()
        batch_loss /= len(dataloader)

        if masked_loss is not None:
            masked_batch_loss /= len(dataloader)
            unmasked_batch_loss /= len(dataloader)

        return batch_loss, masked_batch_loss, unmasked_batch_loss
        
    def pretrain(self, data_splits):
        
        # train_loader, vali_loader, _ = data_generator(self.sourcedata_path, 
        #                                              self.targetdata_path, 
        #                                              self.configs, 
        #                                               self.trial,
        #                                              training_mode='pre_train', 
                                                      
        #                                              subset=False,
        #                                              fraction=self.args.fraction)
        
        path = os.path.join(self.args.pretrain_checkpoints_dir, self.args.pretrain_dataset + "_v" + str(self.args.trial))
        if not os.path.exists(path):
            os.makedirs(path)
        
        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                                     T_max=self.args.pretrain_epochs)
        
        torch.autograd.set_detect_anomaly(True)
        min_vali_loss=None
        masked_penalize=False
        losses = []
        
        with trange(self.args.pretrain_epochs) as tr:
 
            # torch.cuda.empty_cache()
            computed_flops = False
            for it in tr:
                
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                
                train_loss, masked_train_loss, unmasked_train_loss = self.train_one_epoch(dataloader=self.get_data_PT(data_splits["train_X"], "train"), 
                                                                                          split='train', 
                                                                                          optimizer=optimizer,
                                                                                          scheduler=model_scheduler,
                                                                                          masked_penalize=masked_penalize,
                                                                                          computed_flops = computed_flops)

                if it == 0:
                    computed_flops = True


                val_loss, masked_val_loss, unmasked_val_loss, preds, gt = self.val_one_epoch(dataloader=self.get_data_PT(data_splits["val_X"], "val"),
                                                                                  split='val', 
                                                                                  masked_penalize=masked_penalize)
                
                if it==self.args.pretrain_epochs-1:
                    preds=np.squeeze(preds)
                    gt=np.squeeze(gt)

                    fig, axes = plt.subplots(10,1)
                    for idx, i in enumerate(np.random.choice(len(preds), 10, replace=False)):
                        axes[idx].plot(preds[i], label='predictions')
                        axes[idx].plot(gt[i], label='ground-truth')

                    plt.legend()
                    plt.savefig('./plots.pdf')

                
                losses.append(train_loss)
                

                if self.model.mask_ratio<1 and masked_penalize==False:
                    metrics = {
                        "train_loss": train_loss,
                        "train_masked_loss":masked_train_loss,
                        "train_unmasked_loss":unmasked_train_loss,
                        
                        "val_loss": val_loss,
                        "val_masked_loss":masked_val_loss,
                        "val_unmasked_loss":unmasked_val_loss,
                    }
                else:
                    metrics = {
                        "train_loss":train_loss,
                         "val_loss": val_loss,
                       }
                
                tr.set_postfix(metrics)
                
                # checkpoint saving
                if not min_vali_loss or val_loss <= min_vali_loss:
                    if it == 0:
                        min_vali_loss = val_loss

                    print(
                        "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, val_loss, it))

                    min_vali_loss = val_loss
                    model_ckpt = {'epoch': it, 'model_state_dict': self.model}
                    torch.save(self.model, os.path.join(path, "ckpt_best_"+self.args.fraction+".pth"))

                if (it + 1) % 10 == 0:
                    print("Saving model at epoch {}...".format(it + 1))
                    model_ckpt = {'epoch': it, 'model_state_dict': self.model}
                    torch.save(self.model, os.path.join(path, f'ckpt{it + 1}_' + self.args.fraction+'.pth'))
            
        return losses, self.model
        
    def finetune(self, data_splits, args):
        
        # train_loader, vali_loader, test_loader = data_generator(self.sourcedata_path,
        #                                                         self.targetdata_path, 
        #                                                         # self.configs, 
        #                                                          self.trial,
        #                                                         training_mode='fine_tune', 
                                                               
        #                                                         subset=False,
        #                                                        fraction=self.args.fraction)
        
        path = os.path.join(self.args.finetune_checkpoints_dir, self.args.target_dataset + "_v" + str(self.args.trial))
        if not os.path.exists(path):
            os.makedirs(path)
    
        torch.autograd.set_detect_anomaly(True)
        
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, args=self.args)

        train_loader = self.get_data_FT(data_splits["train_X"], data_splits["train_Y"], "train")
        val_loader = self.get_data_FT(data_splits["val_X"], data_splits["val_Y"], "val")
        test_loader = self.get_data_FT(data_splits["test_X"], data_splits["test_Y"], "test")

        train_steps = len(train_loader)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        criterion = nn.CrossEntropyLoss()
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.finetune_epochs,
                                            max_lr=self.args.lr)

        computed_flops = False
        
        with trange(self.args.finetune_epochs) as tr:
            
            for it in tr:
                
                iter_count = 0
                train_loss = []

                total_loss = []
                total_acc = []
                total_auc = []
                total_prc = []

                outs = np.array([])
                trgs = np.array([])

                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                optimizer.zero_grad()

                self.model.train()
                for iteration, (X, labels, mask_original) in tqdm(enumerate(train_loader)):
                    
                    mask_original = 1 - (1 * (torch.isnan(X)))
                    mask_original = mask_original.float().to(self.device)
                    X = torch.nan_to_num(X).float().to(self.device)
                    labels = labels.long().to(self.device)
                    
                    # Debug: Print shapes to understand the issue
                    if iteration == 0:
                        print(f"Debug - Labels shape: {labels.shape}")
                        print(f"Debug - Labels dtype: {labels.dtype}")
                        print(f"Debug - First few labels: {labels[:5] if len(labels) > 0 else labels}")
                    
                    # Fix: Ensure labels are class indices for CrossEntropyLoss
                    if labels.dim() > 1:
                        # If labels are one-hot encoded, convert to class indices
                        labels = labels.argmax(dim=1)
                        if iteration == 0:
                            print(f"Debug - Converted labels shape: {labels.shape}")
                            print(f"Debug - Converted first few labels: {labels[:5] if len(labels) > 0 else labels}")

                    # Compute FLOPs for the first batch
                    # if not computed_flops:
                    #     model_wrapper = ModelWrapper(self.model, mask_original, self.mpl)
                    #     input_shape = tuple(X.shape[1:])  # Exclude batch size
                    #     flops, params = get_model_complexity_info(
                    #         model_wrapper,
                    #         input_res=input_shape,
                    #         as_strings=True,
                    #         print_per_layer_stat=False
                    #     )
                    #     print(f'FLOPs for this model: {flops}')
                    #     print(f'Parameters for this model: {params}')
                    #     computed_flops = True  # Set the flag to prevent further FLOP computations

                    predictions = self.model(X, mask_original, self.mpl)
                    
                    # Debug: Print prediction shape
                    if iteration == 0:
                        print(f"Debug - Predictions shape: {predictions.shape}")

                    loss = criterion(predictions, labels)
                    loss_value = loss.item()
                    
                    if not math.isfinite(loss_value):
                        print("Loss is {}, stopping training".format(loss_value))
                        sys.exit(1)

                    loss /= self.accum_iter
                    
                    acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
                    
                    # Skip per-batch AUC calculation to avoid single-class batch issues
                    # Instead, accumulate predictions and labels for epoch-level calculation
                    pred_probs = torch.softmax(predictions, dim=1).detach().cpu().numpy()
                    labels_np = labels.detach().cpu().numpy()
                    
                    # Store for epoch-level AUC calculation
                    if iteration == 0:
                        # Initialize epoch-level accumulation arrays
                        epoch_pred_probs = pred_probs
                        epoch_labels = labels_np
                    else:
                        epoch_pred_probs = np.vstack([epoch_pred_probs, pred_probs])
                        epoch_labels = np.concatenate([epoch_labels, labels_np])

                    total_acc.append(acc_bs)
                    
                    # Skip per-batch AUC calculations that cause single-class issues
                    # total_auc.append(auc_bs)  # Will calculate at epoch level
                    # total_prc.append(prc_bs)  # Will calculate at epoch level
                    total_loss.append(loss.item())
                
                    loss.backward()
                    optimizer.step()
                    
                    pred = predictions.max(1, keepdim=True)[1]
                    outs = np.append(outs, pred.cpu().numpy())
                    trgs = np.append(trgs, labels.data.cpu().numpy())
                
                    # we use a per iteration (instead of per epoch) lr scheduler
                    if (iteration + 1) % self.accum_iter == 0:
                        optimizer.zero_grad()
                
                # Calculate epoch-level AUC metrics using accumulated data
                try:
                    # Check if we have both classes in the epoch
                    unique_labels = np.unique(epoch_labels)
                    if len(unique_labels) > 1:
                        if epoch_pred_probs.shape[1] == 2:
                            # Binary classification
                            epoch_auc = roc_auc_score(epoch_labels, epoch_pred_probs[:, 1])
                            epoch_prc = average_precision_score(epoch_labels, epoch_pred_probs[:, 1])
                        else:
                            # Multi-class
                            epoch_auc = roc_auc_score(epoch_labels, epoch_pred_probs, average="weighted", multi_class="ovr")
                            epoch_prc = average_precision_score(epoch_labels, epoch_pred_probs, average="weighted")
                        
                        print(f"Epoch {it+1} AUC Metrics - ROC-AUC: {epoch_auc:.4f}, PR-AUC: {epoch_prc:.4f}")
                        
                        # DEBUGGING: Check training patterns
                        epoch_predictions = np.argmax(epoch_pred_probs, axis=1)
                        train_accuracy = np.mean(epoch_labels == epoch_predictions)
                        print(f"  Training accuracy: {train_accuracy*100:.2f}%")
                        print(f"  True label distribution: {np.bincount(epoch_labels.astype(int))}")
                        print(f"  Predicted label distribution: {np.bincount(epoch_predictions.astype(int))}")
                        
                        # Check if model is converging to one class
                        unique_train_preds = np.unique(epoch_predictions)
                        if len(unique_train_preds) == 1:
                            print(f"  ⚠️  WARNING: Model only predicts class {unique_train_preds[0]} in training!")
                            
                    else:
                        print(f"Epoch {it+1} - Only one class present in entire epoch (class {unique_labels[0]})")
                        epoch_auc = 1.0  # Perfect separation if only one class
                        epoch_prc = 1.0
                        
                except Exception as e:
                    print(f"Epoch-level AUC calculation failed: {e}")
                    epoch_auc = 0.0
                    epoch_prc = 0.0
                
                # Calculate F1 using epoch-level predictions
                epoch_predictions = np.argmax(epoch_pred_probs, axis=1)
                F1 = f1_score(epoch_labels, epoch_predictions, average='macro')

                total_loss = torch.tensor(total_loss).mean()  # average loss
                total_acc = torch.tensor(total_acc).mean()  # average acc
                
                # Use epoch-level AUC metrics instead of empty lists
                total_auc = torch.tensor(epoch_auc)
                total_prc = torch.tensor(epoch_prc)

                scheduler.step()
                
                vali_loss, _, _, _, _, vali_F1 = self.vali(val_loader, criterion)
                test_loss, _, _, _, _, test_F1 = self.vali(test_loader, criterion)
            
                print(
                "Epoch: {0} | Train F1: {1:.7f} Vali F1: {2:.7f} Test F1: {3:.7f}".format(
                    it + 1, F1, vali_F1, test_F1))
                early_stopping(vali_loss, self.model, path)
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
                
                adjust_learning_rate(optimizer, scheduler, it + 1, args)
                
        best_model_path = path + '/' + 'checkpoint_' + self.args.fraction + '.pth'
        self.model = torch.load(best_model_path, map_location='cpu')

        self.lr = optimizer.param_groups[0]['lr']

        return self.model
    
    def vali(self, vali_loader, criterion):
        total_loss = []
        total_acc = []
        total_auc = []
        total_prc = []
        
        outs = np.array([])
        trgs = np.array([])
        
        # Accumulate for epoch-level AUC calculation
        vali_all_pred_probs = []
        vali_all_labels = []
        
        print(f"Length of vali loader = {len(vali_loader)}")
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, labels, mask_original) in enumerate(vali_loader):
                
                mask_original = 1 - (1 * (torch.isnan(batch_x)))
                mask_original = mask_original.float().to(self.device)
                batch_x = torch.nan_to_num(batch_x).float().to(self.device)
                labels = labels.long().to(self.device)
                
                # Fix: Ensure labels are class indices for CrossEntropyLoss
                if labels.dim() > 1:
                    # If labels are one-hot encoded, convert to class indices
                    labels = labels.argmax(dim=1)

                # encoder
                outputs = self.model(batch_x, mask_original, self.mpl)

                loss = criterion(outputs, labels)
                
                acc_bs = labels.eq(outputs.detach().argmax(dim=1)).float().mean()
                
                # Accumulate predictions and labels for epoch-level AUC calculation
                pred_probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                labels_np = labels.detach().cpu().numpy()
                
                if i == 0:
                    vali_all_pred_probs = pred_probs
                    vali_all_labels = labels_np
                else:
                    vali_all_pred_probs = np.vstack([vali_all_pred_probs, pred_probs])
                    vali_all_labels = np.concatenate([vali_all_labels, labels_np])

                total_acc.append(acc_bs)
                total_loss.append(loss.item())
                
                pred = outputs.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                
                # record
                total_loss.append(loss)
                
            # Calculate validation-level AUC metrics
            try:
                unique_labels = np.unique(vali_all_labels)
                if len(unique_labels) > 1:
                    if vali_all_pred_probs.shape[1] == 2:
                        # Binary classification
                        vali_auc = roc_auc_score(vali_all_labels, vali_all_pred_probs[:, 1])
                        vali_prc = average_precision_score(vali_all_labels, vali_all_pred_probs[:, 1])
                    else:
                        # Multi-class
                        vali_auc = roc_auc_score(vali_all_labels, vali_all_pred_probs, average="weighted", multi_class="ovr")
                        vali_prc = average_precision_score(vali_all_labels, vali_all_pred_probs, average="weighted")
                else:
                    vali_auc = 1.0  # Perfect separation if only one class
                    vali_prc = 1.0
                    
            except Exception as e:
                print(f"Validation AUC calculation failed: {e}")
                vali_auc = 0.0
                vali_prc = 0.0
             
            # Calculate F1 using validation-level predictions
            vali_predictions = np.argmax(vali_all_pred_probs, axis=1)
            F1 = f1_score(vali_all_labels, vali_predictions, average='macro')

            total_loss = torch.tensor(total_loss).mean()  # average loss
            total_acc = torch.tensor(total_acc).mean()  # average acc
            total_auc = torch.tensor(vali_auc)  # epoch-level auc
            total_prc = torch.tensor(vali_prc)  # epoch-level prc
                

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, total_acc, total_auc, total_prc, trgs, F1
    
    def test(self, data_splits):
        preds = []
        trues = []
        
        total_loss = []
        total_acc = []
        total_auc = []
        total_prc = []
        total_precision, total_recall, total_f1 = [], [], []
        
        outs = np.array([])
        trgs = np.array([])
        
        # Accumulate for epoch-level AUC calculation
        test_all_pred_probs = []
        test_all_labels = []

        self.model.eval().to(self.device)
        with torch.no_grad():
            
            labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
            
            for i, (data, labels, mask_original) in enumerate(self.get_data_FT(data_splits["test_X"], data_splits["test_Y"], "test")):
                
                mask_original = 1 - (1 * (torch.isnan(data)))
                mask_original = mask_original.float().to(self.device)
                data = torch.nan_to_num(data).float().to(self.device)
                labels = labels.long().to(self.device)
                
                # Fix: Ensure labels are class indices
                if labels.dim() > 1:
                    # If labels are one-hot encoded, convert to class indices
                    labels = labels.argmax(dim=1)
                
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)

                # encoder
                outputs = self.model(data, mask_original, self.mpl)
                
                acc_bs = labels.eq(outputs.detach().argmax(dim=1)).float().mean()
                
                # Accumulate predictions and labels for epoch-level AUC calculation
                pred_probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()
                
                if i == 0:
                    test_all_pred_probs = pred_probs
                    test_all_labels = labels_numpy
                else:
                    test_all_pred_probs = np.vstack([test_all_pred_probs, pred_probs])
                    test_all_labels = np.concatenate([test_all_labels, labels_numpy])

                # Calculate per-batch metrics (non-AUC)
                pred_numpy = np.argmax(pred_probs, axis=1)
                precision = precision_score(labels_numpy, pred_numpy, average='macro')
                recall = recall_score(labels_numpy, pred_numpy, average='macro')
                F1 = f1_score(labels_numpy, pred_numpy, average='macro')

                total_acc.append(acc_bs)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(F1)

                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

                labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
                pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
            
        # Calculate test-level AUC metrics
        try:
            unique_labels = np.unique(test_all_labels)
            if len(unique_labels) > 1:
                if test_all_pred_probs.shape[1] == 2:
                    # Binary classification
                    test_auc = roc_auc_score(test_all_labels, test_all_pred_probs[:, 1])
                    test_prc = average_precision_score(test_all_labels, test_all_pred_probs[:, 1])
                else:
                    # Multi-class
                    test_auc = roc_auc_score(test_all_labels, test_all_pred_probs, average="weighted", multi_class="ovr")
                    test_prc = average_precision_score(test_all_labels, test_all_pred_probs, average="weighted")
            else:
                test_auc = 1.0  # Perfect separation if only one class
                test_prc = 1.0
                
        except Exception as e:
            print(f"Test AUC calculation failed: {e}")
            test_auc = 0.0
            test_prc = 0.0
        
        labels_numpy_all = labels_numpy_all[1:]
        pred_numpy_all = pred_numpy_all[1:]

        precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
        recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
        F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
        acc = accuracy_score(labels_numpy_all, pred_numpy_all, )
        
        # Balanced accuracy (more appropriate for imbalanced datasets)
        balanced_acc = balanced_accuracy_score(labels_numpy_all, pred_numpy_all)
        
        # Per-class metrics for medical datasets
        precision_per_class = precision_score(labels_numpy_all, pred_numpy_all, average=None)
        recall_per_class = recall_score(labels_numpy_all, pred_numpy_all, average=None)
        f1_per_class = f1_score(labels_numpy_all, pred_numpy_all, average=None)

        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(test_auc)  # Use epoch-level AUC
        total_prc = torch.tensor(test_prc)  # Use epoch-level PRC

        performance = {'acc': acc * 100, 
                       'balanced_acc': balanced_acc * 100,  # Better for imbalanced data
                       'precision':precision * 100, 
                       'recall':recall * 100, 
                       'F1':F1 * 100, 
                       'total_auc':total_auc * 100, 
                       'total_prc':total_prc * 100,
                       'precision_per_class': precision_per_class * 100,
                       'recall_per_class': recall_per_class * 100,
                       'f1_per_class': f1_per_class * 100}
                       
        # Print detailed metrics for imbalanced datasets
        print("\n=== IMBALANCED DATASET EVALUATION RESULTS ===")
        print(f"Standard Accuracy: {acc*100:.2f}%")
        print(f"Balanced Accuracy: {balanced_acc*100:.2f}% (Better for imbalanced data)")
        print(f"ROC-AUC: {total_auc*100:.2f}%")
        print(f"PR-AUC: {total_prc*100:.2f}% (More informative for imbalanced data)")
        print(f"Macro F1: {F1*100:.2f}%")
        print("Per-class Performance:")
        for i, (p, r, f) in enumerate(zip(precision_per_class, recall_per_class, f1_per_class)):
            print(f"  Class {i}: Precision={p*100:.2f}%, Recall={r*100:.2f}%, F1={f*100:.2f}%")
        
        # DEBUGGING: Check for suspicious perfect scores
        print("\n=== DEBUGGING PERFECT SCORES ===")
        print(f"Total test samples: {len(test_all_labels)}")
        print(f"True label distribution: {np.bincount(test_all_labels.astype(int))}")
        
        test_predictions = np.argmax(test_all_pred_probs, axis=1)
        print(f"Predicted label distribution: {np.bincount(test_predictions.astype(int))}")
        
        # Check if model is just predicting one class
        unique_preds = np.unique(test_predictions)
        if len(unique_preds) == 1:
            print(f"⚠️  WARNING: Model only predicts class {unique_preds[0]} - likely majority class bias!")
        
        # Show prediction confidence
        max_probs = np.max(test_all_pred_probs, axis=1)
        print(f"Prediction confidence - Mean: {np.mean(max_probs):.4f}, Std: {np.std(max_probs):.4f}")
        print(f"Min confidence: {np.min(max_probs):.4f}, Max confidence: {np.max(max_probs):.4f}")
        
        # Confusion Matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(test_all_labels, test_predictions)
        print(f"Confusion Matrix:\n{cm}")
        
        # Check for perfect predictions (potential data leakage)
        perfect_matches = np.sum(test_all_labels == test_predictions)
        print(f"Perfect predictions: {perfect_matches}/{len(test_all_labels)} ({perfect_matches/len(test_all_labels)*100:.2f}%)")
        
        if perfect_matches == len(test_all_labels):
            print("🚨 ALERT: Perfect classification detected!")
            print("Possible causes:")
            print("  1. Data leakage (test data seen during training)")
            print("  2. Dataset too easy/small")
            print("  3. Extreme overfitting")
            print("  4. Model memorization")
            
        # Check class balance
        class_balance = np.bincount(test_all_labels.astype(int)) / len(test_all_labels)
        print(f"Class balance: {class_balance}")
        if np.max(class_balance) > 0.95:
            print(f"⚠️  WARNING: Extremely imbalanced dataset ({np.max(class_balance)*100:.1f}% majority class)")
            
        print("===============================================\n")
        
        return total_loss, total_acc, total_auc, total_prc, trgs, performance