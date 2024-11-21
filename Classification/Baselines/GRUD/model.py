import torch.utils.data as utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math
import numpy as np
import pandas as pd
import time


class FilterLinear(nn.Module):
    def __init__(self, in_features, out_features, filter_square_matrix, bias=True):
        '''
        filter_square_matrix : filter square matrix, whose each elements is 0 or 1.
        '''
        super(FilterLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        use_gpu = torch.cuda.is_available()
        self.filter_square_matrix = None
        if use_gpu:
            self.filter_square_matrix = Variable(filter_square_matrix.cuda(), requires_grad=False)
        else:
            self.filter_square_matrix = Variable(filter_square_matrix, requires_grad=False)
        
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
#         print(self.weight.data)
#         print(self.bias.data)

    def forward(self, input):
#         print(self.filter_square_matrix.mul(self.weight))
        return F.linear(input, self.filter_square_matrix.mul(self.weight), self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'
            
class GRUD(nn.Module):
    def __init__(self, input_size, cell_size, hidden_size, X_mean, output_last=False, num_classes=2):
        super(GRUD, self).__init__()

        self.hidden_size = hidden_size
        self.delta_size = input_size
        self.mask_size = input_size
        self.output_last = output_last

        use_gpu = torch.cuda.is_available()
        self.identity = torch.eye(input_size).cuda() if use_gpu else torch.eye(input_size)
        self.zeros = Variable(torch.zeros(input_size).cuda() if use_gpu else torch.zeros(input_size))
        self.X_mean = Variable(torch.Tensor(X_mean).cuda() if use_gpu else torch.Tensor(X_mean))

        self.zl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.rl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)
        self.hl = nn.Linear(input_size + hidden_size + self.mask_size, hidden_size)

        self.gamma_x_l = FilterLinear(self.delta_size, self.delta_size, self.identity)
        self.gamma_h_l = nn.Linear(self.delta_size, self.delta_size)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_size, num_classes)

        # Activation for binary or multi-class classification
        if num_classes == 1:  # Binary classification
            self.activation = nn.Sigmoid()
        else:  # Multi-class classification
            self.activation = nn.Softmax(dim=1)

    def step(self, x, x_last_obsv, x_mean, h, mask, delta):
        delta_x = torch.exp(-torch.max(self.zeros, self.gamma_x_l(delta)))
        delta_h = torch.exp(-torch.max(self.zeros, self.gamma_h_l(delta)))

        x = mask * x + (1 - mask) * (delta_x * x_last_obsv + (1 - delta_x) * x_mean)
        h = delta_h * h

        combined = torch.cat((x, h, mask), 1)
        z = torch.sigmoid(self.zl(combined))
        r = torch.sigmoid(self.rl(combined))
        combined_r = torch.cat((x, r * h, mask), 1)
        h_tilde = torch.tanh(self.hl(combined_r))
        h = (1 - z) * h + z * h_tilde

        return h

    def forward(self, input):
        batch_size = input.size(0)
        step_size = input.size(2)

        Hidden_State = self.initHidden(batch_size)
        X = torch.squeeze(input[:, 0, :, :])
        X_last_obsv = torch.squeeze(input[:, 1, :, :])
        Mask = torch.squeeze(input[:, 2, :, :])
        Delta = torch.squeeze(input[:, 3, :, :])

        outputs = None
        for i in range(step_size):
            Hidden_State = self.step(
                X[:, i, :],
                X_last_obsv[:, i, :],
                self.X_mean,
                Hidden_State,
                Mask[:, i, :],
                Delta[:, i, :]
            )
            if outputs is None:
                outputs = Hidden_State.unsqueeze(1)
            else:
                outputs = torch.cat((outputs, Hidden_State.unsqueeze(1)), 1)

        # Use last hidden state for classification
        logits = self.fc(outputs[:, -1, :])  # Only the last hidden state
        return self.activation(logits)

    def initHidden(self, batch_size):
        use_gpu = torch.cuda.is_available()
        Hidden_State = Variable(torch.zeros(batch_size, self.hidden_size).cuda()) if use_gpu else Variable(torch.zeros(batch_size, self.hidden_size))
        return Hidden_State
