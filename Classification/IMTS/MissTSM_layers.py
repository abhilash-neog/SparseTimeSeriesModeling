__all__ = ['Transpose', 'get_activation_fn', 'moving_avg', 'series_decomp', 'PositionalEncoding', 'SinCosPosEncoding', 'Coord2dPosEncoding', 'Coord1dPosEncoding', 'positional_encoding', 'TFI', 'LinearEmbed', 'MissTSM']           

import torch
from torch import nn
import math
import copy
import time
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D


class TFI(nn.Module):
    """
    Tensor Feature-wise Embedding (TFI)
    Embed each feature separately
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


class LinearEmbed(nn.Module):
    """
    Linear Embedding
    Embed each feature
    """
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, embedding_dim), nn.LayerNorm(embedding_dim))
        

    def forward(self, x):
        embedded_features = self.embedding(x.unsqueeze(-1))
        return embedded_features


# class SingleHeadAttention(nn.Module):
#     def __init__(self, embed_dim, dropout=0.0):
#         super().__init__()
#         self.scale = embed_dim ** -0.5
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, q, k, v, key_padding_mask=None):
#         # q: (B*W, 1, D)
#         # k,v: (B*W, N, D)
#         attn_weights = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B*W, 1, N)

#         if key_padding_mask is not None:
#             # key_padding_mask: (B*W, N) with True for pad
#             invalid_mask = (key_padding_mask.bool()).unsqueeze(1)
#             attn_weights = attn_weights.masked_fill(invalid_mask, float('-inf'))

#         attn_probs = torch.softmax(attn_weights, dim=-1)
#         attn_probs = self.dropout(attn_probs)
#         attn_out = torch.bmm(attn_probs, v)  # (B*W, 1, D)
#         return attn_out, attn_probs



class MissTSM(nn.Module):
    
    """ 
    Masked Autoencoder with Transformer backbone
    """
    
    def __init__(self, q_dim=8,
                 k_dim=8,
                 v_dim=8, 
                 num_feats=8, 
                 num_heads=1, 
                 out_dim=None, 
                 embed="linear", 
                 mtsm_norm=False,
                 layernorm=True):
        
        '''
        depth: refers to the number of encoder transformer blocks
        decoder_depth: refers to decoder transformer blocks
        mlp_ratio: is w.r.t ViT Block -> number of hidden layers = mlp_ratio*inp_size
        
        '''
        super().__init__()
        
        # TODO: Query dimension should be greater than key, value dimension
        self.q_dim = q_dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.num_heads = num_heads
        self.var_query = nn.Parameter(torch.zeros(1, 1, self.q_dim), requires_grad=True)
        self.num_feats = num_feats
        self.mtsm_norm = mtsm_norm
        self.embed = embed

        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = num_feats
            
        ## Do we really need Multi-head attention?
        ## Grouped query-attention similar to llama3
        
        self.mhca = nn.MultiheadAttention(embed_dim=self.q_dim, num_heads=self.num_heads, batch_first=True)
        # self.mhca = SingleHeadAttention(embed_dim=self.q_dim)
        # self.mask_embed = LinearEmbed(embedding_dim=self.embed_dim)
        if self.embed=="linear":
            self.mask_embed = LinearEmbed(embedding_dim=self.q_dim)
        else:
            self.mask_embed = TFI(input_dim=self.num_feats, embedding_dim=self.q_dim)

        self.pos_embed = PositionalEncoding2D(self.q_dim)
        self.projection = nn.Linear(self.q_dim, self.out_dim)
        
        if layernorm:
            self.layernorm = nn.LayerNorm(self.q_dim)
        else:
            self.layernorm = None
        
    def RevIN(self, x, m):
        '''
        Perform Reversible instance normalization
        '''
        means = torch.sum(x, dim=1) / torch.sum(m == 1, dim=1)
        means = means.unsqueeze(1)
        x = x - means
        
        stdev = torch.sqrt(torch.sum(x * x, dim=1) / torch.sum(m == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1)
        x /= stdev

        return x, means, stdev

    def cross_attention(self, x, m):
        
        batch_size, window_size, num_feat, d = x.shape
        # var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        var_query = self.var_query.expand(batch_size*window_size, -1, -1)

        x = x.view(-1, num_feat, d)
        
        m_ = m.reshape(-1, self.num_feats).clone()
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, window_size, d)
        
        return attn_out
    
    def forward(self, x, m, track_timing=False):
        """
        Forward pass through MissTSM layers.
        
        Args:
            x: Input tensor of shape (batch_size, window_size, num_feats)
            m: Mask tensor of shape (batch_size, window_size, num_feats)
            track_timing: If True, return timing and memory metrics for the entire MissTSM layer
        
        Returns:
            If track_timing=False:
                x: Output tensor
            If track_timing=True:
                x, misstsm_time, misstsm_memory
        """
        # Track timing and memory for the entire MissTSM layer
        if track_timing:
            if torch.cuda.is_available() and x.is_cuda:
                device = x.device
                # Ensure event creation/recording happens on the same current device
                with torch.cuda.device(device):
                    start_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    # Ensure all prior ops complete, then reset peak mem and capture baseline
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats(device)
                    mem_before_alloc = torch.cuda.memory_allocated(device)
                    mem_before_reserved = torch.cuda.memory_reserved(device)
                    start_event.record()
            else:
                misstsm_time_start = time.time()
        
        # perform rev instance norm
        if self.mtsm_norm:
            x, means, std = self.RevIN(x, m)
        else:
            means, std = None, None
        
        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed(x)
        
        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)
            
        # perform cross-attention
        x = self.cross_attention(x, m)
            
        # linear projection
        x = self.projection(x)
        
        if self.mtsm_norm:
            x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
        
        if track_timing:
            if torch.cuda.is_available() and x.is_cuda:
                with torch.cuda.device(device):
                    end_event.record()
                    torch.cuda.synchronize()
                    misstsm_time_ms = start_event.elapsed_time(end_event)
                    misstsm_time = misstsm_time_ms / 1000.0
                    peak_mem_alloc_bytes = torch.cuda.max_memory_allocated(device)
                    peak_mem_reserved_bytes = torch.cuda.max_memory_reserved(device)
                    misstsm_memory_alloc = (peak_mem_alloc_bytes - mem_before_alloc) / (1024 ** 3)
                    misstsm_memory_reserved = (peak_mem_reserved_bytes - mem_before_reserved) / (1024 ** 3)
            else:
                misstsm_time = time.time() - misstsm_time_start
                misstsm_memory_alloc = 0.0
                misstsm_memory_reserved = 0.0
            return x, misstsm_time, misstsm_memory_alloc, misstsm_memory_reserved
        else:
            return x