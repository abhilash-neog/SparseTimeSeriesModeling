from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, transfer_weights, show_series, show_matrix
from utils.augmentations import masked_data
from utils.metrics import metric
from torch.optim import lr_scheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import os
import time
import warnings
import numpy as np
from collections import OrderedDict
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import random
from data_provider.clf_dataloader import data_generator
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score
from config_files.Epilepsy_Configs import Config as EConfigs
from config_files.SleepEEG_Configs import Config as SConfigs
from config_files.EMG_Configs import Config as EMConfigs
from config_files.Gesture_Configs import Config as GConfigs
from config_files.FDB_Configs import Config as FDBConfigs

warnings.filterwarnings('ignore')


class Exp_SimMTM(Exp_Basic):
    def __init__(self, args):
        # self.args = args
        if args.pretrain_dataset == 'SleepEEG':
            args.seq_len = 178
            self.configs=SConfigs()
        elif args.pretrain_dataset=='Epilepsy':
            args.seq_len = 178
            self.configs=EConfigs()
        elif args.pretrain_dataset=='Gesture':
            args.seq_len = 178
            self.configs=GConfigs()
        elif args.pretrain_dataset=='EMG':
            args.seq_len = 178
            self.configs=EMConfigs()
        elif args.pretrain_dataset=='FD-B':
            args.seq_len = 178
            self.configs=FDBConfigs()
        else:
            print("Wrong dataset")
        
        '''
        set num of target classes
        '''
        if args.target_dataset == 'SleepEEG':
            self.configs.num_classes_target=5
        elif args.target_dataset=='Epilepsy':
            self.configs.num_classes_target=2
        elif args.target_dataset=='Gesture':
            self.configs.num_classes_target=8
        elif args.target_dataset=='EMG':
            self.configs.num_classes_target=3
        elif args.target_dataset=='FD-B':
            self.configs.num_classes_target=3
        else:
            print("Wrong dataset")
            
        super(Exp_SimMTM, self).__init__(args)
        self.writer = SummaryWriter(f"./outputs/logs")
        
        # Load datasets
        self.sourcedata_path = os.path.join(args.clf_data_path, args.pretrain_dataset)#f"./dataset/{sourcedata}"  # './data/Epilepsy'
        self.targetdata_path = os.path.join(args.clf_data_path, args.target_dataset)#f"./dataset/{targetdata}"
        self.args = args
        
        
    def _build_model(self):
        
        # epilepsy = 178
        # sleepEEG = 3000
    
        model = self.model_dict[self.args.model].Model(self.args, data_config=self.configs).float()
        
        if self.args.load_checkpoints:
            print("Loading ckpt: {}".format(self.args.load_checkpoints))

            model, weights = transfer_weights(self.args.load_checkpoints, model, device=self.device)
            self.weights = weights
            
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!", self.args.device_ids)
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
            
        # print out the model size
        print('number of model params', sum(p.numel() for p in model.parameters() if p.requires_grad))
        
        return model

#     def _get_data(self, flag, gt=None):
#         data_set, data_loader = data_provider(self.args, flag, gt=None)
#         return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        # criterion = nn.MSELoss()
        criterion = nn.CrossEntropyLoss()
        return criterion

    def pretrain(self):

        # data preparation
        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        
        # subset = self.args.subset  # if subset= true, use a subset for debugging.
        train_loader, vali_loader, _ = data_generator(self.sourcedata_path, 
                                                     self.targetdata_path, 
                                                     self.configs,
                                                     self.args.trial,
                                                     self.args.imputation,
                                                     training_mode='pre_train', 
                                                     subset=False,
                                                     fraction=self.args.fraction)
        
        # # show cases
        # self.train_show = next(iter(train_loader))
        # self.valid_show = next(iter(vali_loader))

        path = os.path.join(self.args.pretrain_checkpoints, self.args.pretrain_dataset + "_v" + str(self.args.trial))
        if not os.path.exists(path):
            os.makedirs(path)

        # optimizer
        model_optim = self._select_optimizer()
        #model_optim.add_param_group({'params': self.awl.parameters(), 'weight_decay': 0})
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optim,
                                                                     T_max=self.args.train_epochs)

        # pre-training
        min_vali_loss = None
        for epoch in range(self.args.train_epochs):
            start_time = time.time()

            train_loss, train_cl_loss, train_rb_loss = self.pretrain_one_epoch(train_loader, model_optim, model_scheduler)
            vali_loss, valid_cl_loss, valid_rb_loss, preds, gt = self.valid_one_epoch(vali_loader)
            
#             if epoch==self.args.train_epochs-1:
#                 preds=np.squeeze(preds)
#                 gt=np.squeeze(gt)

#                 fig, axes = plt.subplots(10,1)
#                 for idx, i in enumerate(np.random.choice(len(preds), 10, replace=False)):
#                     axes[idx].plot(preds[i], label='predictions')
#                     axes[idx].plot(gt[i], label='ground-truth')
                    
#                 plt.legend()
#                 plt.savefig('./plots.pdf')
                
            # log and Loss
            end_time = time.time()
            print(
                "Epoch: {0}, Lr: {1:.7f}, Time: {2:.2f}s | Train Loss: {3:.4f}/{4:.4f}/{5:.4f} Val Loss: {6:.4f}/{7:.4f}/{8:.4f}"
                .format(epoch, model_scheduler.get_lr()[0], end_time - start_time, train_loss, train_cl_loss,
                        train_rb_loss, vali_loss, valid_cl_loss, valid_rb_loss))

            loss_scalar_dict = {
                'train_loss': train_loss,
                'train_cl_loss': train_cl_loss,
                'train_rb_loss': train_rb_loss,
                'vali_loss': vali_loss,
                'valid_cl_loss': valid_cl_loss,
                'valid_rb_loss': valid_rb_loss,
            }

            self.writer.add_scalars(f"/pretrain_loss", loss_scalar_dict, epoch)

            # checkpoint saving
            if not min_vali_loss or vali_loss <= min_vali_loss:
                if epoch == 0:
                    min_vali_loss = vali_loss

                print(
                    "Validation loss decreased ({0:.4f} --> {1:.4f}).  Saving model epoch{2} ...".format(min_vali_loss, vali_loss, epoch))

                min_vali_loss = vali_loss
                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')  # multi-gpu
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best_{self.args.fraction}.pth"))
                # torch.save(encoder_ckpt, os.path.join(path, f"ckpt_best_{self.args.root_path.split('/')[-1]}.pth"))

            if (epoch + 1) % 10 == 0:
                print("Saving model at epoch {}...".format(epoch + 1))

                self.encoder_state_dict = OrderedDict()
                for k, v in self.model.state_dict().items():
                    if 'encoder' in k or 'enc_embedding' in k:
                        if 'module.' in k:
                            k = k.replace('module.', '')
                        self.encoder_state_dict[k] = v
                encoder_ckpt = {'epoch': epoch, 'model_state_dict': self.encoder_state_dict}
                torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch+1}_{self.args.fraction}.pth"))
                # torch.save(encoder_ckpt, os.path.join(path, f"ckpt{epoch + 1}_{self.args.root_path.split('/')[-1]}.pth"))

                # self.show(5, epoch + 1, 'train')
                # self.show(5, epoch + 1, 'valid')

    def pretrain_one_epoch(self, train_loader, model_optim, model_scheduler):

        train_loss = []
        train_cl_loss = []
        train_rb_loss = []

        self.model.train()
        for i, (batch_x, batch_y) in enumerate(train_loader):
            model_optim.zero_grad()

            if self.args.select_channels < 1:

                # random select channels
                B, S, C = batch_x.shape
                random_c = int(C * self.args.select_channels)
                if random_c < 1:
                    random_c = 1

                index = torch.LongTensor(random.sample(range(C), random_c))
                batch_x = torch.index_select(batch_x, 2, index)
            
            # data augumentation
            batch_x_m, mask = masked_data(batch_x, self.args.mask_rate, self.args.lm, self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            batch_x = batch_x.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)
            
            # print(f"batch_x_om = {batch_x_om.shape}")
            # print(f"batch_x = {batch_x.shape}")
            # print(f"mask_om = {mask_om.shape}")
            
            # encoder
            loss, loss_cl, loss_rb, _, _, _, _ = self.model(x_enc=batch_x_om, batch_x=batch_x, mask=mask_om)

            if torch.numel(loss) != 1:
                # print("Loss is not a scalar")
                # print(f"torch.numel(loss) = {torch.numel(loss)}")
                loss=loss.sum()
                loss_cl=loss_cl.sum()
                loss_rb=loss_rb.sum()
            
            # backward
            loss.backward()
            model_optim.step()

            # record
            train_loss.append(loss.item())
            train_cl_loss.append(loss_cl.item())
            train_rb_loss.append(loss_rb.item())

        model_scheduler.step()

        train_loss = np.average(train_loss)
        train_cl_loss = np.average(train_cl_loss)
        train_rb_loss = np.average(train_rb_loss)

        return train_loss, train_cl_loss, train_rb_loss

    def valid_one_epoch(self, vali_loader):
        valid_loss = []
        valid_cl_loss = []
        valid_rb_loss = []

        predictions = []
        gt = []
        
        self.model.eval()
        for i, (batch_x, batch_y) in enumerate(vali_loader):
            # data augumentation
            batch_x_m, mask = masked_data(batch_x, 
                                          self.args.mask_rate, 
                                          self.args.lm,
                                          self.args.positive_nums)
            batch_x_om = torch.cat([batch_x, batch_x_m], 0)

            # masking matrix
            mask = mask.to(self.device)
            mask_o = torch.ones(size=batch_x.shape).to(self.device)
            mask_om = torch.cat([mask_o, mask], 0).to(self.device)

            # to device
            batch_x = batch_x.float().to(self.device)
            batch_x_om = batch_x_om.float().to(self.device)
            # batch_x_mark = batch_x_mark.float().to(self.device)

            # encoder
            loss, loss_cl, loss_rb, _, _, _, preds = self.model(x_enc=batch_x_om, batch_x=batch_x, mask=mask_om)
            
            # Record
            valid_loss.append(loss.item())
            valid_cl_loss.append(loss_cl.item())
            valid_rb_loss.append(loss_rb.item())
            
            predictions.append(preds.detach().cpu().numpy())
            gt.append(batch_x.detach().cpu().numpy())
            
        predictions = np.concatenate(predictions, axis=0)
        original = np.concatenate(gt, axis=0)
        
        vali_loss = np.average(valid_loss)
        valid_cl_loss = np.average(valid_cl_loss)
        valid_rb_loss = np.average(valid_rb_loss)

        self.model.train()
        return vali_loss, valid_cl_loss, valid_rb_loss, predictions, original

    def train(self, setting):
        
        train_loader, vali_loader, test_loader = data_generator(self.sourcedata_path,
                                                                self.targetdata_path, 
                                                                self.configs, 
                                                                self.args.trial,
                                                                self.args.imputation,
                                                                training_mode='fine_tune', 
                                                                subset=False,
                                                                fraction=self.args.fraction)
        
        # data preparation
        # train_data, train_loader = self._get_data(flag='train')
        # vali_data, vali_loader = self._get_data(flag='val')
        # test_data, test_loader = self._get_data(flag='test')
        path = os.path.join(self.args.checkpoints, self.args.target_dataset + "_v" + str(self.args.trial))
        if not os.path.exists(path):
            os.makedirs(path)
            
        # path = os.path.join(self.args.checkpoints, self.args.target_dataset + "_" + str(self.args.pred_len))
        # if not os.path.exists(path):
        #     os.makedirs(path)
            
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True, args=self.args)#, root_path=self.args.pretrain_dataset)

        # Optimizer
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        
    
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
#             stdev_ls = []
#             means_ls = []
            
            total_loss = []
            total_acc = []
            total_auc = []
            total_prc = []

            outs = np.array([])
            trgs = np.array([])
        
            start_time = time.time()
            for i, (X, labels) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                if self.args.select_channels < 1:
                    # Random select channels
                    B, S, C = labels.shape
                    random_c = int(C * self.args.select_channels)
                    if random_c < 1:
                        random_c = 1

                    index = torch.LongTensor(random.sample(range(C), random_c))
                    X = torch.index_select(X, 2, index)
                    labels = torch.index_select(labels, 2, index)

                # to device
                X = X.float().to(self.device)
                labels = labels.long().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                predictions = self.model(X)#, batch_x_mark)
                # stdev_ls.append(stdev.squeeze())
                # means_ls.append(means.squeeze())
                loss = criterion(predictions, labels)
                # f_dim = -1 if self.args.features == 'MS' else 0
                
                
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                
                acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
                onehot_label = F.one_hot(labels)
                pred_numpy = predictions.detach().cpu().numpy()
                
                try:
                    auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
                except:
                    auc_bs = 0.0

                try:
                    prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
                except:
                    prc_bs = 0.0

                total_acc.append(acc_bs)

                if auc_bs != 0:
                    total_auc.append(auc_bs)
                if prc_bs != 0:
                    total_prc.append(prc_bs)
                total_loss.append(loss.item())
                
                # loss
                loss.backward()
                model_optim.step()
                
                pred = predictions.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

            labels_numpy = labels.detach().cpu().numpy()
            pred_numpy = np.argmax(pred_numpy, axis=1)
            F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

            total_loss = torch.tensor(total_loss).mean()  # average loss
            total_acc = torch.tensor(total_acc).mean()  # average acc
            total_auc = torch.tensor(total_auc).mean()  # average auc
            total_prc = torch.tensor(total_prc).mean()

            scheduler.step()
                
            # train_loss = np.average(train_loss)
            # vali_loss = self.vali(vali_loader, criterion)
            # test_loss = self.vali(test_loader, criterion)
            vali_loss, _, _, _, _, vali_F1 = self.vali(vali_loader, criterion)
            test_loss, _, _, _, _, test_F1 = self.vali(test_loader, criterion)

            end_time = time.time()
            print(
            "Epoch: {0}, Steps: {1}, Time: {2:.2f}s | Train F1: {3:.7f} Vali F1: {4:.7f} Test F1: {5:.7f}".format(
                epoch + 1, train_steps, end_time - start_time, F1, vali_F1, test_F1))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                # break

            # adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        # best_model_path = path + '/' + 'checkpoint_'+self.args.root_path.split('/')[-1]+'.pth'
        best_model_path = path + '/' + 'checkpoint_' + self.args.fraction + '.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        self.lr = model_optim.param_groups[0]['lr']

        return self.model

    def vali(self, vali_loader, criterion):
        total_loss = []
        total_acc = []
        total_auc = []
        total_prc = []
        
        outs = np.array([])
        trgs = np.array([])
    
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, labels) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                labels = labels.long().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(batch_x)#, batch_x_mark)

                # loss
                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # pred = outputs.detach().cpu()
                # true = batch_y.detach().cpu()
                loss = criterion(outputs, labels)
                
                acc_bs = labels.eq(outputs.detach().argmax(dim=1)).float().mean()
                onehot_label = F.one_hot(labels)
                pred_numpy = outputs.detach().cpu().numpy()
                
                try:
                    auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
                except:
                    auc_bs = 0.0

                try:
                    prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
                except:
                    prc_bs = 0.0

                total_acc.append(acc_bs)

                if auc_bs != 0:
                    total_auc.append(auc_bs)
                if prc_bs != 0:
                    total_prc.append(prc_bs)
                total_loss.append(loss.item())
                
                pred = outputs.max(1, keepdim=True)[1]
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())
                
                # record
                total_loss.append(loss)
                
            labels_numpy = labels.detach().cpu().numpy()
            pred_numpy = np.argmax(pred_numpy, axis=1)
            F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

            total_loss = torch.tensor(total_loss).mean()  # average loss
            total_acc = torch.tensor(total_acc).mean()  # average acc
            total_auc = torch.tensor(total_auc).mean()  # average auc
            total_prc = torch.tensor(total_prc).mean()
                

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss, total_acc, total_auc, total_prc, trgs, F1

    def test(self):
        _, _, test_loader = data_generator(self.sourcedata_path,
                                            self.targetdata_path, 
                                            self.configs, 
                                           self.args.trial,
                                           self.args.imputation,
                                            training_mode='fine_tune', 
                                            subset=False)
        
        # test_data, test_loader = self._get_data(flag='test')
        # test_data_gt, test_loader_gt = self._get_data(flag='test', gt=True)
        
        preds = []
        trues = []
        
        total_loss = []
        total_acc = []
        total_auc = []
        total_prc = []
        total_precision, total_recall, total_f1 = [], [], []
        
        outs = np.array([])
        trgs = np.array([])
        # emb_test_all = []

        self.model.eval()
        with torch.no_grad():
            
            labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
            
            for i, (data, labels) in enumerate(test_loader):
                data = data.float().to(self.device)
                labels = labels.long().to(self.device)
                # batch_x_mark = batch_x_mark.float().to(self.device)

                # encoder
                outputs = self.model(data)#, batch_x_mark)

                # f_dim = -1 if self.args.features == 'MS' else 0
                # outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                # pred = outputs.detach().cpu().numpy()
                # true = labels.detach().cpu().numpy()
                
                # acc_bs = batch_y.eq(pred.detach().argmax(dim=1)).float().mean()
                # onehot_label = F.one_hot(true)
                # pred_numpy = pred.detach().cpu().numpy()
                
                acc_bs = labels.eq(outputs.detach().argmax(dim=1)).float().mean()
                onehot_label = F.one_hot(labels)
                pred_numpy = outputs.detach().cpu().numpy()
                labels_numpy = labels.detach().cpu().numpy()
                
                try:
                    auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
                except:
                    auc_bs = 0.0

                try:
                    prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
                except:
                    prc_bs = 0.0

                pred_numpy = np.argmax(pred_numpy, axis=1)
                precision = precision_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
                recall = recall_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
                F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

                total_acc.append(acc_bs)

                if auc_bs != 0:
                    total_auc.append(auc_bs)
                if prc_bs != 0:
                    total_prc.append(prc_bs)
                total_precision.append(precision)
                total_recall.append(recall)
                total_f1.append(F1)

                # total_loss.append(loss.item())

                pred = outputs.max(1, keepdim=True)[1]  # get the index of the max log-probability
                outs = np.append(outs, pred.cpu().numpy())
                trgs = np.append(trgs, labels.data.cpu().numpy())

                labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
                pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
            
            
        labels_numpy_all = labels_numpy_all[1:]
        pred_numpy_all = pred_numpy_all[1:]

        precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
        recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
        F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
        acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

        total_loss = torch.tensor(total_loss).mean()
        total_acc = torch.tensor(total_acc).mean()
        total_auc = torch.tensor(total_auc).mean()
        total_prc = torch.tensor(total_prc).mean()

        performance = {'acc': acc * 100, 
                       'precision':precision * 100, 
                       'recall':recall * 100, 
                       'F1':F1 * 100, 
                       'total_auc':total_auc * 100, 
                       'total_prc':total_prc * 100}

        # emb_test_all = torch.cat(tuple(emb_test_all))
        return total_loss, total_acc, total_auc, total_prc, trgs, performance
                
#             for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader_gt):
#                 batch_x = batch_x.float().to(self.device)
#                 batch_y = batch_y.float().to(self.device)
#                 batch_x_mark = batch_x_mark.float().to(self.device)
                
#                 f_dim = -1 if self.args.features == 'MS' else 0
#                 batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
#                 true = batch_y.detach().cpu().numpy()
                
#                 trues.append(true)

#         preds = np.array(preds)
#         trues = np.array(trues)
#         # preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
#         # trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

#         mae, mse, rmse, mape, mspe = metric(preds, trues)
#         print('{0}->{1}, mse:{2:.3f}, mae:{3:.3f}'.format(self.args.seq_len, self.args.pred_len, mse, mae))
#         # f = open("./outputs/score_"+self.args.root_path.split('/')[-1]+".txt", 'a')
#         f = open(folder_path+"score_"+self.args.root_path.split('/')[-1]+".txt", 'a')
#         f.write('{0}->{1}, {2:.3f}, {3:.3f} \n'.format(self.args.seq_len, self.args.pred_len, mse, mae))
#         f.close()

    def show(self, num, epoch, type='valid'):

        # show cases
        if type == 'valid':
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.valid_show
        else:
            batch_x, batch_y, batch_x_mark, batch_y_mark = self.train_show

        # data augumentation
        batch_x_m, batch_x_mark_m, mask = masked_data(batch_x, batch_x_mark, self.args.mask_rate, self.args.lm,
                                                      self.args.positive_nums)
        batch_x_om = torch.cat([batch_x, batch_x_m], 0)

        # masking matrix
        mask = mask.to(self.device)
        mask_o = torch.ones(size=batch_x.shape).to(self.device)
        mask_om = torch.cat([mask_o, mask], 0).to(self.device)

        # to device
        batch_x = batch_x.float().to(self.device)
        batch_x_om = batch_x_om.float().to(self.device)
        batch_x_mark = batch_x_mark.float().to(self.device)

        # Encoder
        with torch.no_grad():
            loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x = self.model(batch_x_om, batch_x_mark, batch_x, mask=mask_om)

        for i in range(num):

            if i >= batch_x.shape[0]:
                continue

        fig_logits, fig_positive_matrix, fig_rebuild_weight_matrix = show_matrix(logits, positives_mask, rebuild_weight_matrix)
        self.writer.add_figure(f"/{type} show logits_matrix", fig_logits, global_step=epoch)
        self.writer.add_figure(f"/{type} show positive_matrix", fig_positive_matrix, global_step=epoch)
        self.writer.add_figure(f"/{type} show rebuild_weight_matrix", fig_rebuild_weight_matrix, global_step=epoch)