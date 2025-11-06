import random
import numpy as np
import torch
import torch.nn as nn
from layers.MissTSMLayer import MissTSM, MissTSMSkip

    
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

class Model(nn.Module):
    ''' train LSTM encoder-decoder and make predictions '''

    def __init__(self, configs):
        '''
        : param configs:     configuration object containing all hyperparameters
        '''
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_size = configs.enc_in  # Number of input features
        self.output_size = configs.c_out  # Number of output features
        
        # For multivariate forecasting, typically enc_in == c_out (predict all variables)
        # If they differ, ensure decoder is configured correctly
        if self.input_size != self.output_size:
            print(f"Warning: enc_in ({self.input_size}) != c_out ({self.output_size}). "
                  f"Decoder will produce {self.output_size} features.")
        
        self.hidden_size = getattr(configs, 'hidden_size', 64)
        self.num_layers = getattr(configs, 'num_layers', 2)
        self.model_type = getattr(configs, 'model_type', 'LSTM')
        self.dropout = getattr(configs, 'dropout', 0.0)
        self.device = torch.device('cuda' if torch.cuda.is_available() and configs.use_gpu else 'cpu')
        
        # MissTSM related parameters
        self.use_misstsm = getattr(configs, 'misstsm', False)
        self.skipconnection = getattr(configs, 'skip_connection', False)
        
        # Training parameters
        self.target_len = self.pred_len
        self.training_prediction = getattr(configs, 'training_prediction', 'recursive')
        self.teacher_forcing_ratio = getattr(configs, 'teacher_forcing_ratio', 0.5)
        self.alpha = getattr(configs, 'alpha', 0.0)  # Minimum value for output clamping
        
        # RevIN for backbone (LSTM encoder-decoder) - separate from MissTSM's internal RevIN
        self.backbone_revin = getattr(configs, 'backbone_revin', False)
        if self.backbone_revin:
            print("\nApplying RevIN to LSTM backbone (before encoder, after decoder)\n")

        # Initialize MissTSM layer if enabled
        if self.use_misstsm:
            print("\nApplying MissTSM layer to LSTM\n")
            if self.skipconnection:
                self.MTSMLayer = MissTSMSkip(q_dim=getattr(configs, 'q_dim', 64),
                                             k_dim=getattr(configs, 'k_dim', 64),
                                             v_dim=getattr(configs, 'v_dim', 64),
                                             num_feats=self.input_size,
                                             num_heads=getattr(configs, 'misstsm_heads', 1),
                                             out_dim=self.input_size,
                                             embed=getattr(configs, 'mtsm_embed', 'linear'),
                                             mtsm_norm=getattr(configs, 'mtsm_norm', False),
                                             layernorm=getattr(configs, 'layernorm', True))
            else:
                self.MTSMLayer = MissTSM(q_dim=getattr(configs, 'q_dim', 64),
                                         k_dim=getattr(configs, 'k_dim', 64),
                                         v_dim=getattr(configs, 'v_dim', 64),
                                         num_feats=self.input_size,
                                         num_heads=getattr(configs, 'misstsm_heads', 1),
                                         out_dim=self.input_size,
                                         embed=getattr(configs, 'mtsm_embed', 'linear'),
                                         mtsm_norm=getattr(configs, 'mtsm_norm', False),
                                         layernorm=getattr(configs, 'layernorm', True))
            # After MissTSM, output size is still input_size (out_dim=input_size)
            encoder_input_size = self.input_size
        else:
            encoder_input_size = self.input_size

        # Encoder and Decoder
        self.encoder = encoder(input_size=encoder_input_size, 
                              hidden_size=self.hidden_size, 
                              num_layers=self.num_layers, 
                              model_type=self.model_type, 
                              dropout=self.dropout).to(self.device)
        
        self.decoder = decoder(input_size=self.output_size, 
                              hidden_size=self.hidden_size, 
                              num_layers=self.num_layers, 
                              model_type=self.model_type, 
                              dropout=self.dropout).to(self.device)
    
    def backbone_revin_normalize(self, x, m=None):
        """
        Perform Reversible Instance Normalization for backbone (LSTM encoder-decoder).
        Normalizes per instance (per sample in batch) along the sequence dimension.
        x: [batch, seq_len, num_features]
        m: [batch, seq_len, num_features] mask (1 = observed, 0 = missing)
        """
        if m is None:
            # If no mask provided, assume all values are observed
            m = torch.ones_like(x)
        
        # Compute mean per instance (per sample) along sequence dimension
        # Sum over sequence, divide by number of observed values
        means = torch.sum(x * m, dim=1, keepdim=True) / (torch.sum(m, dim=1, keepdim=True) + 1e-5)
        x_norm = x - means
        
        # Compute std per instance
        stdev = torch.sqrt(torch.sum(x_norm * x_norm * m, dim=1, keepdim=True) / 
                          (torch.sum(m, dim=1, keepdim=True) + 1e-5) + 1e-5)
        x_norm = x_norm / stdev
        
        return x_norm, means, stdev
    
    def backbone_revin_denormalize(self, x, means, stdev):
        """
        Reverse the normalization for backbone outputs.
        x: [batch, seq_len, num_features] normalized output
        means: [batch, 1, num_features] mean values
        stdev: [batch, 1, num_features] std values
        """
        # Expand means and stdev to match output sequence length
        means_expanded = means.expand(-1, x.shape[1], -1)
        stdev_expanded = stdev.expand(-1, x.shape[1], -1)
        
        x_denorm = x * stdev_expanded + means_expanded
        return x_denorm
    
    def forward(self, x, m=None, target_batch=None):
        """
        Forward pass for LSTM encoder-decoder.
        x: [batch, seq_len, num_features] input sequence
        m: [batch, seq_len, num_features] mask (optional, for MissTSM)
        target_batch: [batch, target_len, num_features] target sequence (for teacher forcing)
        """
        batch_size = x.shape[0]
        device = x.device
        
        # Apply MissTSM if enabled (has its own RevIN if mtsm_norm=True)
        if self.use_misstsm and m is not None:
            x = self.MTSMLayer(x, m)  # [batch, seq_len, input_size] -> [batch, seq_len, input_size]
        
        # Apply RevIN for backbone (before encoder) if enabled
        # This is separate from MissTSM's internal RevIN
        if self.backbone_revin:
            # Create mask for backbone if not provided (use all ones if MissTSM was applied)
            backbone_mask = m if m is not None else torch.ones_like(x)
            x, backbone_revin_means, backbone_revin_stdev = self.backbone_revin_normalize(x, backbone_mask)
        
        # Encoder outputs: x should be [batch, seq_len, input_size]
        encoder_output, encoder_hidden = self.encoder(x)
        
        # Initialize outputs tensor
        outputs = torch.zeros(batch_size, self.target_len, self.output_size, device=device)
        
        # Decoder input: start with zeros or last encoder output
        decoder_input = torch.zeros([batch_size, self.output_size], device=device)
        decoder_hidden = encoder_hidden

        
        # Decoder with teacher forcing (training) or recursive prediction (eval)
        if self.training and target_batch is not None:
            if self.training_prediction == 'recursive':
                # Predict recursively
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output
                    decoder_input = decoder_output

            elif self.training_prediction == 'teacher_forcing':
                # Use teacher forcing based on probability
                if random.random() < self.teacher_forcing_ratio:
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[:, t, :] = decoder_output
                        decoder_input = target_batch[:, t, :]
                else:
                    # Predict recursively
                    for t in range(self.target_len):
                        decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                        outputs[:, t, :] = decoder_output
                        decoder_input = decoder_output

            elif self.training_prediction == 'mixed_teacher_forcing':
                # Mixed teacher forcing: decide per timestep
                for t in range(self.target_len):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    outputs[:, t, :] = decoder_output

                    # Predict with teacher forcing or recursively
                    if random.random() < self.teacher_forcing_ratio:
                        decoder_input = target_batch[:, t, :]
                    else:
                        decoder_input = decoder_output
        else:
            # Evaluation: always predict recursively
            for t in range(self.target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                outputs[:, t, :] = decoder_output
                decoder_input = decoder_output
        
        # Apply output clamping if alpha is set
        if self.alpha > 0:
            outputs = outputs.clone()
            outputs[:, :, 0] = torch.clamp(outputs[:, :, 0], min=self.alpha)
        
        # Denormalize outputs if backbone RevIN was applied
        if self.backbone_revin:
            outputs = self.backbone_revin_denormalize(outputs, backbone_revin_means, backbone_revin_stdev)
        
        return outputs