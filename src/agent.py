import math
import random
import numpy as np
from collections import deque
import warnings; warnings.simplefilter('ignore')
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# device = torch.device('cpu')

class ParamRNN(nn.Module):
    def __init__(self, env, args):
        super(ParamRNN, self).__init__()
        
        # set sizes
        max_len = env.dataloader.get_max_len()
        embedding_size = env.dataloader.get_embed_size()
        parameter_shape = env.tagger.get_para_shape()
        trellis_shape = env.tagger.get_trellis_shape()
    
        rnn_hidden = args.rnn_hidden
        n_filters = args.cnn_flt_n
        filter_size = args.cnn_flt_size
        stride = args.cnn_stride
        
        # set random seed
        self.random = random.Random(args.seed_agent)
        
        para_h, para_w = parameter_shape
        # CNN for CRF parameters
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=filter_size,              
                stride=stride,        # input&output (batch, channel, height, width)
            )
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, filter_size, stride)
        self.bn2 = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*2, filter_size, stride)
        self.bn3 = nn.BatchNorm2d(n_filters*2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = filter_size, stride = stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(para_w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(para_h)))
        linear_input_size = convw * convh * n_filters * 2
        
        self.fc1 = nn.Linear(linear_input_size, rnn_hidden)
        
        # LSTM for w sequence
        self.rnn = nn.LSTM(
            input_size=embedding_size ,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1)
    
    def forward(self, tagger_x, seq_x):
        # CNN
        x1 = F.relu(self.bn1(self.conv1(tagger_x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        # h_c shape (n_layers, batch, hidden_size)
        r_out,_ = self.rnn(seq_x, None) 
        x2 = r_out[:, -1, :]
        x = x1 + x2
        return self.fc(x) # flatten the output
    
    
    def get_action(self, state, device):
#         self.eval()
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]
        
        # use TE to explore
        max_idx = candidates[np.argmax([seq_confidence[i][0] for i in candidates])]
        
        para_ts = torch.from_numpy(tagger_para).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(para_ts, embed_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = 0.3
    
        if self.random.random() < eps_threshold:
            return (0, max_idx, max_qvalue)

        for i in candidates:
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0)
            qvalue = self.forward(para_ts, embed_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)


