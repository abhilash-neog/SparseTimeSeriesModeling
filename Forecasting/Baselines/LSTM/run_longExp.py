import argparse
import os
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='Autoformer & Transformer family for Time Series Forecasting')

# basic config
parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
parser.add_argument('--model', type=str, required=True, default='Autoformer',
                    help='model name, options: [Autoformer, Informer, Transformer]')

# data loader
parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

parser.add_argument('--gt_root_path', type=str, default='', help='root path of the ground-truth file')
parser.add_argument('--gt_data_path', type=str, default='', help='gt data file')
parser.add_argument('--output_path', type=str, required=True, default='')
parser.add_argument('--trial', type=int, default=0)
    
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./model_checkpoints/', help='location of model checkpoints')

parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

# LSTM
parser.add_argument('--num_layers', type=int, default=2, help='number of lstm layers')
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--hidden_size', type=int, default=64, help='hidden size')
parser.add_argument('--model_type', type=str, default='LSTM', help='model type')
parser.add_argument('--dropout', type=float, default=0.0005, help='dropout')
parser.add_argument('--training_prediction', type=str, default='recursive', help='training prediction')
parser.add_argument('--teacher_forcing_ratio', type=float, default=0.5, help='teacher forcing ratio')
parser.add_argument('--alpha', type=float, default=0.0, help='alpha')

# misstsm
parser.add_argument('--misstsm', type=int, default=1, help="whether to apply misstsm layer")
parser.add_argument('--q_dim', type=int, default=128, help='dimension of model')
parser.add_argument('--k_dim', type=int, default=128, help='dimension of model')
parser.add_argument('--v_dim', type=int, default=128, help='dimension of model')
parser.add_argument('--layernorm', type=int, default=1, help='whether to apply layernorm after misstsm layer')
parser.add_argument('--mtsm_norm', type=int, default=1, help='perform denorm misstsm')
parser.add_argument('--mtsm_embed', type=str, default="linear", help='type of TFI embedding to apply')
parser.add_argument('--skip_connection', type=int, default=0, help='add skipconnection to misstsm')
parser.add_argument('--misstsm_heads', type=int, default=1, help='number of heads in misstsm layer')
parser.add_argument('--backbone_revin', type=int, default=0, help='apply RevIN to LSTM backbone (before encoder, after decoder)')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='mse', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

# GPU
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=1, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
parser.add_argument('--test_flop', action='store_true', default=False, help='See utils/tools for usage')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

if args.use_gpu and args.use_multi_gpu:
    args.dvices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

print('Args in experiment:')
print(args)

Exp = Exp_Main

if args.is_training:
    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_hs{}_nl{}_dp{}_tp{}_tf{}_al{}_dt{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.hidden_size,
            args.num_layers,
            args.dropout,
            args.training_prediction,
            args.teacher_forcing_ratio,
            args.alpha,
            args.des, ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)

        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        # exp.test(setting)
        if args.gt_root_path == '':
            exp.test(setting)#, test=1)
        else:
            print(f"calling test masked")
            exp.test_masked(setting)#, test=1)

        # if args.do_predict:
        #     print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        #     exp.predict(setting, True)

        torch.cuda.empty_cache()
else:
    ii = 0
    setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(args.model_id,
                                                                                                  args.model,
                                                                                                  args.data,
                                                                                                  args.features,
                                                                                                  args.seq_len,
                                                                                                  args.label_len,
                                                                                                  args.pred_len,
                                                                                                  args.d_model,
                                                                                                  args.n_heads,
                                                                                                  args.e_layers,
                                                                                                  args.d_layers,
                                                                                                  args.d_ff,
                                                                                                  args.factor,
                                                                                                  args.embed,
                                                                                                  args.distil,
                                                                                                  args.des, ii)

    exp = Exp(args)  # set experiments
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    # exp.test(setting, test=1)
    
    if args.gt_root_path == '':
        exp.test(setting, test=1)
    else:
        exp.test_masked(setting, test=1)
        
    torch.cuda.empty_cache()
