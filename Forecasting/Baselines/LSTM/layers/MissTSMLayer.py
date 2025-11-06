import torch
from torch import nn
import math
import copy
from positional_encodings.torch_encodings import PositionalEncoding1D, PositionalEncoding2D

class TFI(nn.Module):
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

class LinearEmbed(nn.Module):
    """
    Embed each feature
    """
    def __init__(self, embedding_dim=8):
        super().__init__()
        self.embedding = nn.Sequential(nn.Linear(1, embedding_dim), nn.LayerNorm(embedding_dim))
        

    def forward(self, x):
        embedded_features = self.embedding(x.unsqueeze(-1))
        return embedded_features
    
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
        var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
        x = x.view(-1, num_feat, d)
        
        m_ = copy.deepcopy(m.reshape(-1, num_feat))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, window_size, d)
        
        return attn_out
    
    def forward(self, x, m):
        
        # perform rev instance norm
        if self.mtsm_norm:
            print(f"Applying RevIN to MissTSM")
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
        
        return x
    
class MissTSMSkip(nn.Module):
    
    """ 
    MissTSM with skip connections
    """
    
    def __init__(self, q_dim=8,
                 k_dim=8,
                 v_dim=8, 
                 num_feats=8, 
                 num_heads=1, 
                 out_dim=None, 
                 norm=False, 
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
        self.norm = norm
        self.mtsm_norm = mtsm_norm
        self.embed = embed
        
        if out_dim:
            self.out_dim = out_dim
        else:
            self.out_dim = num_feats
            
        ## Do we really need Multi-head attention?
        ## Grouped query-attention similar to llama3
        
        self.mhca = nn.MultiheadAttention(embed_dim=self.q_dim, num_heads=self.num_heads, batch_first=True)

        # self.mask_embed = TFI(input_dim=self.num_feats, embedding_dim=self.embed_dim)
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
        x = x / stdev

        return x, means, stdev

    def cross_attention(self, x, m):
        
        batch_size, window_size, num_feat, d = x.shape
        var_query = self.var_query.repeat_interleave(batch_size*window_size, dim=0)
        
        x = x.view(-1, num_feat, d)
        
        m_ = copy.deepcopy(m.reshape(-1, num_feat))
        
        attn_out, _ = self.mhca(var_query, x, x, key_padding_mask=m_)
        
        attn_out = attn_out.reshape(batch_size, window_size, d)
        
        return attn_out
    
    def forward(self, x, m):
        
        # perform rev instance norm
        if self.mtsm_norm:
            # print(f"Applying RevIN to MissTSM")
            x, means, std = self.RevIN(x, m)
        else:
            means, std = None, None
        
        x_inp = x

        # embed patches
        x = self.mask_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed(x)

        # print(f"shape after positional embedding = {x.shape}")
        
        # apply layernorm
        if self.layernorm:
            x = self.layernorm(x)
            
        # perform cross-attention
        x = self.cross_attention(x, m)
        
        # linear projection
        x = self.projection(x)
        
        x = m*x_inp + (1-m)*x

        if self.mtsm_norm:
            x = x * (std[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))
            x = x + (means[:, 0, :].unsqueeze(1).repeat(1, x.shape[1], 1))

        return x