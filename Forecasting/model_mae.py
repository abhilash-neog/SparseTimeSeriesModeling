import sys
sys.path.insert(1, './utils/.')

import math
import torch
import numpy as np
import torch.nn as nn
import pandas as pd
import wandb
import sys
import copy
import time

from functools import partial
from tqdm import trange, tqdm
from torch.utils.data import DataLoader, RandomSampler
from timm.models.vision_transformer import Block
from utils import MaskEmbed, MAEDataset, NativeScaler, get_1d_sincos_pos_embed, ActiveEmbed, FeatEmbed, adjust_learning_rate
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D

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
                 seq_len, 
                 enc_embed_dim,
                 dec_embed_dim,
                 task_name,
                 num_feats,
                 n2one,
                 batch_size,
                 device):
        
        self.enc_embed_dim = enc_embed_dim
        self.dec_embed_dim = dec_embed_dim
        self.seq_len = seq_len 
        self.device = device
        self.batch_size = batch_size
        self.num_feats = num_feats
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.enc_embed_dim), requires_grad=False).to(self.device)
        self.pos_embed = PositionalEncoding2D(enc_embed_dim).to(self.device)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.seq_len + 1, self.dec_embed_dim), requires_grad=False).to(self.device)
        # self.decoder_pos_embed = PositionalEncoding2D(dec_embed_dim).to(self.device)
        
        if n2one==True:
            self.decoder_pred = nn.Linear(self.dec_embed_dim, 1, bias=True).to(self.device)  # decoder to patch
        else:
            self.decoder_pred = nn.Linear(self.dec_embed_dim, num_feats, bias=True).to(self.device)  # decoder to patch
        
        self.initialize_embeddings()
    
    def initialize_embeddings(self):
        
        enc_z = torch.rand((1, self.seq_len + 1, self.num_feats, self.enc_embed_dim)).to(self.device) # +1 for the cls token
        self.pos_embed = self.pos_embed(enc_z)
        
#         pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], self.seq_len, cls_token=True)
#         self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_1d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.seq_len, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        

class MaskedAutoencoder(nn.Module):
    
    """ 
    Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self,
                 utils,
                 args,
                 num_feats,
                 norm_layer=nn.LayerNorm, 
                 norm_field_loss=False,
                 encode_func='linear'):
        
        '''
        depth: refers to the number of encoder transformer blocks
        decoder_depth: refers to decoder transformer blocks
        mlp_ratio: is w.r.t ViT Block -> number of hidden layers = mlp_ratio*inp_size
        
        '''
        super().__init__()
        
        self.utils = utils
        
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.embed_dim = args.encoder_embed_dim
        self.depth = args.encoder_depth
        self.num_heads= args.encoder_num_heads
        self.mlp_ratio = args.mlp_ratio
        self.decoder_embed_dim = args.decoder_embed_dim
        self.decoder_num_heads = args.decoder_num_heads
        self.decoder_depth = args.decoder_depth
        self.mask_ratio = args.mask_ratio
        task_name = args.task_name

        self.num_feats = num_feats
        self.norm_layer = norm_layer
        self.encode_func = encode_func
        self.norm_field_loss = norm_field_loss
        
        self.var_query = nn.Parameter(torch.zeros(1, 1, self.embed_dim), requires_grad=True)
        self.mhca = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=self.num_heads, batch_first=True)

        self.mask_embed = FeatEmbed(input_dim=self.num_feats,
                                    embedding_dim=self.embed_dim,
                                    norm_layer=self.norm_layer)
        
        # if self.encode_func == 'active':
        #     self.mask_embed = ActiveEmbed(self.embed_dim)
        # else:
        #     self.mask_embed = MaskEmbed(in_channel=self.num_feats, 
        #                                 embed_dim=self.embed_dim, 
        #                                 norm_layer=self.norm_layer)
        
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 1, self.embed_dim))
        
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.rec_len + 1, self.embed_dim), requires_grad=False)  
        
        self.encoder_blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.depth)])
        
        self.norm = self.norm_layer(self.embed_dim)


        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(self.embed_dim, self.decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.decoder_embed_dim))

        # self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.rec_len + 1, self.decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(self.decoder_embed_dim, self.decoder_num_heads, self.mlp_ratio, qkv_bias=True, norm_layer=self.norm_layer)
            for i in range(self.decoder_depth)])

        self.decoder_norm = self.norm_layer(self.decoder_embed_dim)
            
        # --------------------------------------------------------------------------

        self.norm_field_loss = self.norm_field_loss
        self.initialize_weights()
        self.set_masking_mode()
        self.lookback_window = None 
        


    def initialize_weights(self):

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        # w = self.mask_embed.proj.weight.data
        # torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        w = [self.mask_embed.embeddings[i][0].weight.data for i in range(self.num_feats)]
        for i in range(self.num_feats):
            torch.nn.init.xavier_uniform_(w[i].view([w[i].shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def set_lookbackwindow(self, lookback):
        self.lookback_window = lookback
        print(f"Lookback window has been set to {self.lookback_window }, forecast window has been set to {self.rec_len-self.lookback_window}")
    
    def set_masking_mode(self, masking_mode=None):
        if masking_mode is None:
            masking_mode = "random_masking"
        assert masking_mode in ["continuous_masking", "random_masking"]
        self.masking_mode = masking_mode
        print(f"Masking Mode has been set to {self.masking_mode}")
        
    def masking(self, x, m):
        if self.masking_mode=="random_masking":
            return self.random_masking(x, m)
        elif self.masking_mode=="continuous_masking":
            return self.continous_masking(x, m)
        else:
            print("Masking Error.")
    
    def continous_masking(self, x, m):
        N, L, D = x.shape  # batch, length, dim
        
        #uncomment this part when we infer
        len_keep = self.lookback_window

        noise = torch.linspace(0, 1, L, device=x.device).repeat(N, 1)  # predictable noise
        # noise[m[:,0,:] < eps] = 1
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask

#         if self.training:
#             mask[m[:, 0, :] < eps] = 0
        
        return x_masked, mask, nask, ids_restore
    
    def random_masking(self, x, m):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        
        #uncomment this part when we infer
        len_keep = int(L * (1 - self.mask_ratio))
        # if self.training:
        #     len_keep = int(L * (1 - self.mask_ratio))
        # else:
        #     len_keep = int(torch.min(torch.sum(m, dim=2)))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # noise[m[:,0,:] < eps] = 1
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        nask = torch.ones([N, L], device=x.device) - mask
        
        return x_masked, mask, nask, ids_restore

    def cross_attention(self, x, m):
        
        batch_size, window_size, num_feat, d = x.shape
        var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
        x = x.view(-1, num_feat, d)
        m_ = copy.deepcopy(m.view(-1, num_feat))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.view(batch_size, window_size, d)
        
        return attn_out
    
    def forward_encoder(self, x, m):
        
        # embed patches
        x = self.mask_embed(x)
        
        # add pos embed w/o cls token
        x = x + self.mpl.pos_embed[:, 1:, :, :]
        
        # perform cross-attention
        x = self.cross_attention(x, m)
        
        # masking: length -> length * mask_ratio
        x, mask, nask, ids_restore = self.masking(x, m)
        
        # append cls token
        cls_token = self.cls_token + self.mpl.pos_embed[:, :1, :, :]
        
        cls_tokens = cls_token.expand(x.shape[0], -1, -1, -1)
        
        cls_tokens = cls_tokens[:, :, 0, :]
        
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.encoder_blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, nask, ids_restore


    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)
        
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token
        
        # add pos embed
        x = x + self.mpl.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        
        # predictor projection
        # x = torch.tanh(self.decoder_pred(x))/2 + 0.5
        x = self.mpl.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]
        
        return x


    def forward_loss(self, data, pred, mask, nask, miss_idx, masked_penalize=False):
        """
        data: [N, 1, L]
        pred: [N, L]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        
        '''
        create a 2D mask for loss computation
        
        Mask for loss computation = Original mask * Expanded 1D mask
        '''
        
        # data = data.transpose(1, 2)
        target = data
        
        mask = mask.unsqueeze(-1) * torch.ones(1, pred.shape[2], device=mask.device)
        mask = mask*miss_idx
        nask = torch.ones([pred.shape[0], pred.shape[1], pred.shape[2]], device=mask.device) - mask
        
        if self.norm_field_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + eps)**.5
        
        loss = (pred - target) ** 2
        
        masked_loss = None
        unmasked_loss = None
        
        if masked_penalize:
            loss = (loss * mask).sum() / mask.sum()
        else:
            if mask.sum()==0:
                loss = (loss * nask).sum() / nask.sum()
            else:
                masked_loss = (loss * mask).sum() / mask.sum()
                unmasked_loss = (loss * nask).sum() / nask.sum() 
                loss = masked_loss + unmasked_loss
                
        return loss, masked_loss, unmasked_loss


    def forward(self, data, miss_idx):
        
        latent, mask, nask, ids_restore = self.forward_encoder(data, miss_idx)
        pred = self.forward_decoder(latent, ids_restore) 
        
        return pred, mask, nask
    
    def freeze_encoder_model(self):
        for param in self.encoder_blocks.parameters():
            param.requires_grad = False
        print(f"Encoder Blocks Frozen!")
    
    
    def train_model(self, Xtrain, Xval, Xtest, args, utils=None, masked_penalize=False):
        
        self.batch_size = args['batch_size']
        self.accum_iter = args['accum_iter']
        self.min_lr = args['min_lr']
        self.weight_decay = args['weight_decay']
        self.lr = args['lr']
        self.blr = args['blr']
        self.warmup_epochs = args['warmup_epochs']
        self.model = None
        self.norm_parameters = None
        self.max_epochs = args['max_epochs']
        self.device = args['device']
        self.eval_freq = args['eval_freq']
        self.feature_wise_rmse = args['feature_wise_rmse']
        self.rec_len = args['window']
        self.task_name = args['task_name']
        self.n2one_ft = args['n2one']
        self.utils = utils
        
        num_windows = args["num_windows"]
        num_samples = args["num_samples"]
        
        
        self.mpl = ModelPlugins(seq_len=self.rec_len, enc_embed_dim=self.embed_dim, dec_embed_dim=self.decoder_embed_dim,
                                num_feats=self.num_feats, task_name=self.task_name, n2one=self.n2one_ft, batch_size=self.batch_size,
                                device=self.device)
        
        print(f"self task name = {self.task_name}")
        
        if self.task_name=="finetune":
            masked_penalize=True
            print("Mask Penalize Has been set to True.")
            lookback = args['lookback_window']
            self.set_lookbackwindow(lookback)
            self.set_masking_mode(masking_mode="continuous_masking")
            if args['freeze_encoder']=='True':
                self.freeze_encoder_model()
            
        
        config = init_wandb(args, self.task_name)
        
        # Set missing
        M = 1 - (1 * (torch.isnan(Xtrain)))
        M = M.float().to(self.device)
        
        Xtrain = torch.nan_to_num(Xtrain)
        Xtrain = Xtrain.to(self.device)
        
        # Xval = torch.nan_to_num(Xval)
        Xval = Xval.to(self.device)
        Xtest = Xtest.to(self.device)
        
        self.to(self.device)
        
        n_batches = int(math.ceil(Xtrain.shape[0] / self.batch_size))
        
        eff_batch_size = self.batch_size * self.accum_iter
        
        if self.lr is None:  # only base_lr is specified
            self.lr = self.blr * eff_batch_size / 64
            
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, betas=(0.9, 0.95))
        loss_scaler = NativeScaler()

        dataset = MAEDataset(Xtrain, M)
        dataloader = DataLoader(
            dataset, sampler=RandomSampler(dataset),
            batch_size=self.batch_size,
        )

        losses = np.full(self.max_epochs, np.nan)
        val_rmse = []
        train_rmse = []
        
        if self.n2one_ft==True:
            self.std = self.utils.feat_std[:, :, self.utils.target_index].unsqueeze(1).to(self.device)
            self.mean = self.utils.feat_mean[:, :, self.utils.target_index].unsqueeze(1).to(self.device) 
        else:
            self.std = self.utils.feat_std.to(self.device)
            self.mean = self.utils.feat_mean.to(self.device)
        
        with trange(self.max_epochs) as tr:
            '''
            Do we need gradient accumulation here? For maybe large models, or for fine-tuning a pre-trained model,
            gradient accumulation maybe useful
            '''
            for it in tr:
                self.train() # setting model to train mode
                
                total_loss = 0
                batch_loss = 0
                masked_batch_loss = 0
                unmasked_batch_loss = 0
                batch_train_loss = 0 
                batch_val_loss = 0
                
                adjust_learning_rate(optimizer, epoch=it+1, lr=self.lr, min_lr=self.min_lr, 
                                     max_epochs=self.max_epochs, warmup_epochs=self.warmup_epochs)
                
                learning_rate = optimizer.param_groups[0]['lr']
                print(f"learning rate in epoch {it} = {learning_rate} ")
                
                for iteration, (samples, masks) in enumerate(dataloader):
                    
                    # we use a per iteration (instead of per epoch) lr scheduler
                    if (iteration + 1) % self.accum_iter == 0:
                        optimizer.zero_grad()
                    
                    with torch.cuda.amp.autocast():
                        
                        pred, mask, nask = self(samples, masks) # we get de-normalized predictions
                        
                        if self.n2one_ft==True:
                            samples = samples[:, :, self.utils.target_index].unsqueeze(2)
                        
                        loss, masked_loss, unmasked_loss = self.forward_loss(samples, pred, mask, nask, masks, masked_penalize)
                        
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
                    loss_scaler(loss, optimizer, parameters=self.parameters(), update_grad=(iteration + 1) % self.accum_iter == 0)
                
                # batch_loss = batch_loss/n_batches
                batch_loss /= len(dataloader)
                
                if masked_loss is not None:
                    masked_batch_loss /= len(dataloader)
                    unmasked_batch_loss /= len(dataloader)
                    
                losses[it] = batch_loss
                
                
                if it % self.eval_freq == 0:
                    t0 = time.time()
                    
                    with torch.no_grad():
                        batch_val_loss_dict = self.evaluate_and_plot(Xval,
                                                                     args=args, 
                                                                     train_or_test='val',
                                                                     num_windows=num_windows,
                                                                     num_samples=num_samples,
                                                                     it=it)
                                                                                            
                        batch_train_loss_dict = self.evaluate_and_plot(Xtrain,
                                                                       args=args,
                                                                       train_or_test='train',
                                                                       num_windows=num_windows,
                                                                       num_samples=num_samples,
                                                                       it=it)

                if self.mask_ratio<1 and masked_penalize==False:
                    metrics = {
                        "mse_loss": batch_loss,
                        "masked_mse_loss":masked_batch_loss,
                        "unmasked_mse_loss":unmasked_batch_loss,
                        "train_rmse":batch_train_loss_dict["RMSE"],
                        "val_rmse":batch_val_loss_dict["RMSE"]
                    }
                else:
                    metrics = {
                        "mse_loss":batch_loss,
                        "train_rmse":batch_train_loss_dict["RMSE"],
                        "val_rmse": batch_val_loss_dict["RMSE"]
                       }
                
                tr.set_postfix(metrics)
                
                if self.feature_wise_rmse=='True':
                    batch_train_loss_dict.pop("RMSE")
                    batch_val_loss_dict.pop("RMSE")
                    
                    metrics = {**metrics, **batch_train_loss_dict, **batch_val_loss_dict}
                
                wandb.log(metrics)
                
        
        with torch.no_grad():
            val_eval_dict = self.evaluate(val_X=Xval, args=args, train_or_val='val')
            train_eval_dict = self.evaluate(val_X=Xtrain, args=args, train_or_val='train')
            
            test_eval_dict = self.evaluate_and_plot(Xval=Xtest, args=args, train_or_test='test')
            
            self.wandb_summarize(val_eval_dict["rmse_dict"], train_or_test='val')
            self.wandb_summarize(train_eval_dict["rmse_dict"], train_or_test='train')
            self.wandb_summarize(test_eval_dict, train_or_test='test')
            wandb.finish()
            
        return losses
    
    def evaluate_and_plot(self, Xval, train_or_test, args, it=0, num_windows=25, num_samples=8):

        val_eval_dict = self.evaluate(val_X=Xval, args=args, train_or_val=train_or_test)
        
        batch_val_loss = val_eval_dict["rmse_dict"]

        print(f"{train_or_test} avg loss = {val_eval_dict['avg_loss']}")

        val_predictions = val_eval_dict["preds"]

        val_masks = val_eval_dict["mask"]

        val_x_gt = val_eval_dict["gt"]
        
        val_og_masks = val_eval_dict["og_masks"]
        
        # Plotting the N_samples for each features
        val_start = np.random.choice(len(val_x_gt), num_samples)
        
#         self.utils.plot_context_window_grid(val_x_gt, val_predictions, val_masks, val_og_masks,
#                                             val_start, it, train_or_test, title_prefix=self.masking_mode)
        
        
        # self.utils.plot_context_window_grid_with_original_masks(val_x_gt, val_predictions, val_og_masks, 
        #                                                         val_start, it, 'Val', title_prefix=self.masking_mode)
        # self.utils.plot_context_window_grid_with_original_masks(train_x_gt, train_predictions, train_og_masks, 
        #                                                         train_start, it, 'Train', title_prefix=self.masking_mode)
        
        
        # Plotting merged context-windows for visualizing seasonal trends
#         if self.masking_mode == "random_masking":
#             idx = np.array([self.rec_len*i for i in range(num_windows)]) #creating a continous time-window indices
            
#             print(f"for {train_or_test}, we have len(val_x_gt) = {len(val_x_gt)} \nidx = {idx}")
#             val_start = idx + np.random.choice(len(val_x_gt)-idx[-1]-1, 1)
        
#             self.utils.plot_merged_context_windows(val_x_gt, val_predictions, val_og_masks, val_masks, 
#                                                    val_start, it, train_or_test, title_prefix=self.masking_mode)
            
#         if self.masking_mode == "continuous_masking":
            
#             if num_windows*self.rec_len >= len(val_x_gt):
#                 num_windows = (len(val_x_gt)//self.rec_len)//2
                
#             len_context_window = self.rec_len*num_windows
            
#             val_start = np.random.choice(len(val_x_gt)-len_context_window, 1)
#             val_idx = np.arange(val_start, val_start+len_context_window+1, 1)
            
            
#             plt_idx = np.floor(np.linspace(self.lookback_window, self.rec_len-1, 4))
            
#             self.utils.plot_forecast(val_x_gt, val_predictions, val_og_masks,
#                                   val_idx, plt_idx, self.lookback_window ,it, train_or_test, title_prefix=self.masking_mode)
            
        return batch_val_loss
    
    
    def wandb_summarize(self, val_eval_dict, train_or_test):
        
        rmse=train_or_test+'_rmse'
        if self.feature_wise_rmse=='True':
            for k,v in val_eval_dict.items():
                if k=='RMSE':
                    wandb.summary[rmse] = v
                else:
                    wandb.summary[k] = v

        else:
            wandb.summary[rmse] = val_eval_dict['RMSE']
    
    def perform_zero_shot(self, Xtrain, Xval, args, lookback, utils):
        
        self.task_name = args['task_name']
        self.batch_size = args['batch_size']
        self.device = args['device']
        self.feature_wise_rmse = args['feature_wise_rmse']
        self.rec_len = args['window']
        self.n2one_ft = args['n2one']
        
        self.utils = utils
        num_windows = args["num_windows"]
        num_samples = args["num_samples"]
        
        
        self.mpl = ModelPlugins(seq_len=self.rec_len, enc_embed_dim=self.embed_dim, dec_embed_dim=self.decoder_embed_dim,
                                num_feats=self.num_feats, task_name=self.task_name, n2one=self.n2one_ft, batch_size=self.batch_size,
                                device=self.device)
        
        
        _ = init_wandb(args, self.task_name)
        
        self.set_lookbackwindow(lookback)
        self.set_masking_mode(masking_mode="continuous_masking")
        
        Xtrain = Xtrain.to(self.device)
        
        Xval = Xval.to(self.device)
        
        self.to(self.device)
        
        if self.n2one_ft==True:
            self.std = self.utils.feat_std[:, :, self.utils.target_index].unsqueeze(1).to(self.device)
            self.mean = self.utils.feat_mean[:, :, self.utils.target_index].unsqueeze(1).to(self.device) 
        else:
            self.std = self.utils.feat_std.to(self.device)
            self.mean = self.utils.feat_mean.to(self.device)

        batch_train_loss, batch_val_loss = self.evaluate_and_plot(Xtrain, 
                                                                  Xval, 
                                                                  args, 
                                                                  num_windows=num_windows,
                                                                  num_samples=num_samples)
        self.wandb_summarize(batch_val_loss, batch_train_loss)
        
            
    def predict(self, X, args):
        self.eval() # setting model to eval mode
        
        self.batch_size = args['batch_size']
        self.accum_iter = args['accum_iter']
        self.min_lr = args['min_lr']
        self.weight_decay = args['weight_decay']

        self.blr = args['blr']
        self.warmup_epochs = args['warmup_epochs']
        self.model = None
        self.norm_parameters = None
        self.max_epochs = args['max_epochs']
        self.device = args['device']
        
        Xval = X.clone()

        # Set missing
        M = 1 - (1 * (torch.isnan(Xval)))
        M = M.float().to(self.device)
        
        Xval = torch.nan_to_num(Xval)
        n_batches = int(math.ceil(Xval.shape[0] / self.batch_size))
        
        # set optimizers
        # param_groups = optim_factory.add_weight_decay(model, args.weight_decay)
        # eff_batch_size = self.batch_size * self.accum_iter
    
        dataset = MAEDataset(Xval, M)
        dataloader = DataLoader(dataset,
                                batch_size=self.batch_size,
                                shuffle=False)

        samples_list = []
        preds_list = []
        masks_list = []
        nasks_list = []
        batch_loss = 0
        og_masks_list = []
        
        for it, (samples, masks) in tqdm(enumerate(dataloader)):
                
            with torch.cuda.amp.autocast():
                pred, mask, nask = self(samples, masks)
                
                if self.n2one_ft==True:
                    samples = samples[:, :, self.utils.target_index].unsqueeze(2)
                
                # samples_list.append(samples.transpose(1, 2).detach())
                samples_list.append(samples.detach())
                preds_list.append(pred.detach())
                masks_list.append(mask.detach())
                nasks_list.append(nask.detach())
                og_masks_list.append(masks.detach())
                    
                loss, _, _ = self.forward_loss(samples, pred, mask, nask, masks)
                
                loss_value = loss.item()
                batch_loss += loss_value
        
        samples_list = torch.cat(samples_list, dim=0)
        preds_list = torch.cat(preds_list, dim=0)
        masks_list = torch.cat(masks_list, dim=0)
        nasks_list = torch.cat(nasks_list, dim=0)
        og_masks_list = torch.cat(og_masks_list, dim=0)
        
            # Convert to torch
        # batch_loss = batch_loss/n_batches
        batch_loss = batch_loss/len(dataloader)
        
        metrics = {
            "batch_loss":batch_loss,
            "pred":preds_list,
            "samples":samples_list,
            "masks":masks_list,
            "nasks":nasks_list,
            "og_masks":og_masks_list
        }
        return metrics
    
    def evaluate(self, val_X, args, train_or_val):
        
        metrics = self.predict(val_X, args)
        predictions = metrics['pred']
        masks = metrics['masks']
        val_X = metrics["samples"]
        loss = metrics['batch_loss']
        og_masks = metrics['og_masks']
        
        '''
        rmse
        '''
        predictions = predictions.to(self.device)
        val_X = val_X.to(self.device)
        
        # if self.n2one_ft=="True":
        #     std = self.utils.feat_std[:, :, self.utils.chloro_index].unsqueeze(1)
        #     mean = self.utils.feat_mean[:, :, self.utils.chloro_index].unsqueeze(1)
        # else:
        #     std = self.utils.feat_std
        #     mean = self.utils.feat_mean
        # print(f"predictions shape = {predictions.shape}")
        # print(f"val_X shape = {val_X.shape}")
        # print(f"self std = {self.std.shape}")
        # print(f"self mean = {self.mean.shape}")
        
        predictions = predictions*self.std + self.mean
        
        val_X = val_X*self.std + self.mean
        
        # print(f"masks = {masks.shape}")
        # print(f"og_masks = {og_masks.shape}")
        
        twodmasks = masks.unsqueeze(-1) * torch.ones(1, predictions.shape[2], device=masks.device)
        twodmasks = twodmasks*og_masks
        
        RMSE_dict = {}
        if self.task_name == "pretrain":
            if self.feature_wise_rmse=='True':
                sqred_err = (predictions-val_X)**2
                sum_sqred_err = (sqred_err*twodmasks).sum((0,1))
                feature_wise_rmse = (sum_sqred_err/twodmasks.sum((0,1)))**0.5
                RMSE_dict = {train_or_val+"_"+self.utils.inp_cols[idx]:feature_wise_rmse[idx].item() for idx in range(self.num_feats)}
            
            RMSE = (((predictions-val_X)**2).mean())**0.5
            RMSE_dict["RMSE"] = RMSE.item()
            
        elif self.task_name == "finetune" or self.task_name == "zeroshot":
            if self.feature_wise_rmse=='True':
                sqred_err = (predictions-val_X)**2
                sum_sqred_err = (sqred_err*twodmasks).sum((0,1))
                feature_wise_rmse = (sum_sqred_err/twodmasks.sum((0,1)))**0.5
                RMSE_dict = {train_or_val+"_"+self.utils.inp_cols[idx]:feature_wise_rmse[idx].item() for idx in range(self.num_feats)}
            
            sqred_err = (predictions-val_X)**2
            RMSE = ((sqred_err*twodmasks).sum() / twodmasks.sum())**0.5
            RMSE_dict["RMSE"] = RMSE.item()

        return {'avg_loss':loss, 'rmse_dict':RMSE_dict, 'preds':predictions, 'gt': val_X, 'mask': masks, 'og_masks':og_masks}
    

def mae_base(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=4,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=2., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_medium(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=32, depth=4, num_heads=4,
        decoder_embed_dim=32, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


def mae_large(**kwargs):
    model = MaskedAutoencoder(
        embed_dim=64, depth=8, num_heads=4,
        decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps), **kwargs)
    return model


if __name__ == '__main__':

    model = MaskedAutoencoder(
        rec_len=4, embed_dim=8, depth=1, num_heads=1,
        decoder_embed_dim=8, decoder_depth=1, decoder_num_heads=1,
        mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=eps)
    )
    
    X = pd.DataFrame([[np.nan, 0.5, np.nan, 0.8]])
    
    X = torch.tensor(X.values, dtype=torch.float32)
    M = 1 - (1 * (np.isnan(X)))
    X = torch.nan_to_num(X)
    
    X = X.unsqueeze(dim=1)
    print(model.forward(X, M, 0.75))