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

EPS = 0.3

class TE(nn.Module):
    def __init__(self, env, args):
        super(TE, self).__init__()
        
    def get_action(self, state, device, mode):
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        # note: index in training set
        #       index in data should be: scope[i] for i in candidates
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]
        
        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]
        
        return (1, max_idx, 0)

class Rand(nn.Module):
    def __init__(self, env, args):
        super(Rand, self).__init__()
        # set random seed
        self.random = random.Random(args.seed_agent*2)
        
    def get_action(self, state, device, mode):
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        # note: index in training set
        #       index in data should be: scope[i] for i in candidates
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]
        
        # use TE to explore
        max_idx = self.random.choice(candidates)
        
        return (1, max_idx, 0)
        
class ParamRNN(nn.Module):
    def __init__(self, env, args):
        super(ParamRNN, self).__init__()
        
        # set sizes
        embedding_size = env.dataloader.get_embed_size()
        parameter_shape = env.tagger.get_para_shape()
    
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
    
    
    def get_action(self, state, device, mode):
        self.eval()
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        # note: index in training set
        #       index in data should be: scope[i] for i in candidates
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]
        
        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]
        
        para_ts = torch.from_numpy(tagger_para).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(para_ts, embed_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = EPS
    
        if self.random.random() < eps_threshold and mode != 'online':
            return (0, max_idx, max_qvalue)

        for i in candidates:
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            qvalue = self.forward(para_ts, embed_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)

class ParamRNNBudget(nn.Module):
    def __init__(self, env, args):
        super(ParamRNNBudget, self).__init__()
        
        # set sizes
        embedding_size = env.dataloader.get_embed_size()
        parameter_shape = env.tagger.get_para_shape()
    
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
        self.fc2 = nn.Linear(1, rnn_hidden)
        
        # LSTM for w sequence
        self.rnn = nn.LSTM(
            input_size=embedding_size ,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1)
    
    def forward(self, tagger_x, seq_x, budget):
        # CNN for tagger
        x1 = F.relu(self.bn1(self.conv1(tagger_x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        
        # RNN for sequence
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        # h_c shape (n_layers, batch, hidden_size)
        r_out,_ = self.rnn(seq_x, None) 
        x2 = r_out[:, -1, :]
        
        # MLP for time
        x3 = F.relu(self.fc2(budget))
        x = x1 + x2 + x3
        return self.fc(x) # flatten the output
    
    def get_action(self, state, device, mode):
        self.eval()
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]
        
        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]
        
        para_ts = torch.from_numpy(tagger_para).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        budget_ts = torch.from_numpy(np.array([budget])).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(para_ts, embed_ts, budget_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = EPS
    
        if self.random.random() < eps_threshold and mode != 'online':
            return (0, max_idx, max_qvalue)

        for i in candidates:
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            qvalue = self.forward(para_ts, embed_ts, budget_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)

class TrellisCNN(nn.Module): # all
    def __init__(self, env, args):
        super(TrellisCNN, self).__init__()
        
        # set sizes
        embedding_size = env.dataloader.get_embed_size()
        trellis_shape = env.tagger.get_trellis_shape()
    
        rnn_hidden = args.rnn_hidden
        n_filters = args.cnn_flt_n
        filter_size = args.cnn_flt_size
        stride = args.cnn_stride
        
        # set random seed
        self.random = random.Random(args.seed_agent)
        
        para_h, para_w = trellis_shape
        
        # CNN for CRF trellis
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=(filter_size, para_w),              
                stride=stride,          # input&output (batch, channel, height, width)
            )
        self.bn1 = nn.BatchNorm2d(n_filters)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_w_out(size, kernel_w_size = para_w, stride = stride):
            return (size - (kernel_w_size - 1) - 1) // stride  + 1
        def conv2d_h_out(size, kernel_h_size = filter_size, stride = stride):
            return (size - (kernel_h_size - 1) - 1) // stride  + 1
        
        convw = conv2d_w_out(para_w)
        convh = conv2d_h_out(para_h)
        linear_input_size = convw * convh * n_filters
        
        self.fc11 = nn.Linear(linear_input_size, 3 * rnn_hidden)
        self.fc12 = nn.Linear(3 * rnn_hidden, rnn_hidden)
        self.fc2 = nn.Linear(rnn_hidden, rnn_hidden)
        self.fc3 = nn.Linear(1, rnn_hidden)
        
        # LSTM for w sequence
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden * 3, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x, seq_x, conf_x):
        # CNN for trellis
        x1 = F.relu(self.bn1(self.conv1(trellis_x)))
        x1 = F.relu(self.fc11(x1.view(x1.size(0), -1)))
        x1 = F.relu(self.fc12(x1)).squeeze()  
        # RNN for sequence
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        # h_c shape (n_layers, batch, hidden_size)
        #print(seq_x.size())
        #print(trellis_x.size())
        #x2 = torch.cat((seq_x, trellis_x.squeeze(1)), -1)
        r_out,_ = self.rnn(seq_x, None) 
        # output of last time step
        x2 = r_out[:, -1, :]
        x2 = F.relu(self.fc2(x2)).squeeze()
        # MLP for confidence
        x3 = F.relu(self.fc3(conf_x))

        norm = x1.norm(dim=-1, p=2, keepdim = True)
        x1 = (x1-torch.mean(x1)).div(norm.expand_as(x1))
        norm = x2.norm(dim=-1, p=2, keepdim = True)
        x2 = (x2-torch.mean(x2)).div(norm.expand_as(x2))
        norm = x3.norm(dim=-1, p=2, keepdim = True)
        x3 = (x3-torch.mean(x3)).div(norm.expand_as(x3))
        #x = x1 + x2 + x3
        x = torch.cat((x1, x2, x3), -1)
        return self.fc(x) # flatten the output
    
    def get_action(self, state, device, mode):
        self.eval()
        # observation = [seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]

        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]

        trellis_ts = torch.from_numpy(seq_trellis[max_idx]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        conf_ts = torch.from_numpy(np.array(seq_confidence[max_idx])).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(trellis_ts, embed_ts, conf_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = EPS

        if self.random.random() < eps_threshold and mode != 'online':
            return (0, max_idx, max_qvalue)

        for i in candidates:
            trellis_ts = torch.from_numpy(seq_trellis[i]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            conf_ts = torch.from_numpy(np.array(seq_confidence[i])).type(torch.FloatTensor).unsqueeze(0).to(device)
            qvalue = self.forward(trellis_ts, embed_ts, conf_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)



class TrellisBudget(nn.Module): # all
    def __init__(self, env, args):
        super(TrellisBudget, self).__init__()
        
        # set sizes
        embedding_size = env.dataloader.get_embed_size()
        trellis_shape = env.tagger.get_trellis_shape()
    
        rnn_hidden = args.rnn_hidden
        n_filters = args.cnn_flt_n
        filter_size = args.cnn_flt_size
        stride = args.cnn_stride
        
        # set random seed
        self.random = random.Random(args.seed_agent)
        
        para_h, para_w = trellis_shape
        
        # CNN for CRF trellis
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=(filter_size, para_w),              
                stride=stride,          # input&output (batch, channel, height, width)
            )
        self.bn1 = nn.BatchNorm2d(n_filters)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_w_out(size, kernel_w_size = para_w, stride = stride):
            return (size - (kernel_w_size - 1) - 1) // stride  + 1
        def conv2d_h_out(size, kernel_h_size = filter_size, stride = stride):
            return (size - (kernel_h_size - 1) - 1) // stride  + 1
        
        convw = conv2d_w_out(para_w)
        convh = conv2d_h_out(para_h)
        linear_input_size = convw * convh * n_filters
        
        self.fc11 = nn.Linear(linear_input_size, 3 * rnn_hidden)
        self.fc12 = nn.Linear(3 * rnn_hidden, rnn_hidden)
        self.fc2 = nn.Linear(rnn_hidden, rnn_hidden)
        self.fc3 = nn.Linear(1, rnn_hidden)
        self.fc4 = nn.Linear(1, rnn_hidden)
        
        # LSTM for w sequence
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x, seq_x, conf_x, budget):
        # CNN for trellis
        x1 = F.relu(self.bn1(self.conv1(trellis_x)))
        x1 = F.relu(self.fc11(x1.view(x1.size(0), -1)))
        x1 = F.relu(self.fc12(x1))
        
        # RNN for sequence
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        # h_c shape (n_layers, batch, hidden_size)
        #print(seq_x.size())
        #print(trellis_x.size())
        #x2 = torch.cat((seq_x, trellis_x.squeeze(1)), -1)
        r_out,_ = self.rnn(seq_x, None) 
        # output of last time step
        x2 = r_out[:, -1, :]
        x2 = F.relu(self.fc2(x2))
        
        # MLP for confidence
        x3 = F.relu(self.fc3(conf_x))
        
        # MLP for budget
        x4 = F.relu(self.fc4(budget))
    
#         x3 = self.fc2(conf_test)
#         x = torch.cat((x1, x2, x3), 1)
#         x = torch.cat((x1, x2), 1)
        x = x1 + x2 + x3 + x4
        
        return self.fc(x) # flatten the output
    
    def get_action(self, state, device, mode):
        self.eval()
        # observation = [seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]

        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]

        trellis_ts = torch.from_numpy(seq_trellis[max_idx]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        conf_ts = torch.from_numpy(np.array(seq_confidence[max_idx])).type(torch.FloatTensor).unsqueeze(0).to(device)
        budget_ts = torch.from_numpy(np.array([budget])).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(trellis_ts, embed_ts, conf_ts, budget_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = EPS

        if self.random.random() < eps_threshold and mode != 'online':
            return (0, max_idx, max_qvalue)

        for i in candidates:
            trellis_ts = torch.from_numpy(seq_trellis[i]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            conf_ts = torch.from_numpy(np.array(seq_confidence[i])).type(torch.FloatTensor).unsqueeze(0).to(device)
            qvalue = self.forward(trellis_ts, embed_ts, conf_ts, budget_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)
    
class PAL(nn.Module): # all
    def __init__(self, env, args):
        super(PAL, self).__init__()
        
        # set sizes
        embedding_size = env.dataloader.get_embed_size()
        trellis_shape = env.tagger.get_trellis_shape()
    
        rnn_hidden = 256
        n_filters = 20
        filter_size = args.cnn_flt_size
        
        # set random seed
        self.random = random.Random(args.seed_agent)
        
        para_h, para_w = trellis_shape
        
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=(3, para_w),              
                stride=1,         
            )

        self.pool = nn.AvgPool2d((para_h - 3 + 1, 1))
        self.fc1 = nn.Linear(n_filters, rnn_hidden)

        
        self.conv21 = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (3, embedding_size), stride = 1),
                nn.ReLU(),
                nn.MaxPool2d((para_h - 3 + 1, 1))
            )
        self.conv22 = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (4, embedding_size), stride = 1),
                nn.ReLU(),
                nn.MaxPool2d((para_h - 4 + 1, 1))
            )
        self.conv23 = nn.Sequential(
                nn.Conv2d(in_channels = 1, out_channels = 128, kernel_size = (5, embedding_size), stride = 1),
                nn.ReLU(),
                nn.MaxPool2d((para_h - 5 + 1, 1))
            )
        self.fc2 = nn.Linear(384, rnn_hidden)
        self.fc3 = nn.Linear(1, rnn_hidden)

        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x, seq_x, conf_x):
        x1 = self.pool(self.conv1(trellis_x))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        seq_x = seq_x.unsqueeze(1)
        x21 = self.conv21(seq_x)
        x22 = self.conv22(seq_x)
        x23 = self.conv23(seq_x)
        x2 = torch.cat((x21, x22, x23), 1)
        x2 = F.relu(self.fc2(x2.view(x2.size(0), -1)))
        
        x3 = F.relu(self.fc3(conf_x))
        x = x1 + x2 + x3
        
        return self.fc(x) # flatten the output

    def get_action(self, state, device, mode):
        self.eval()
        # observation = [seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]

        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]

        trellis_ts = torch.from_numpy(seq_trellis[max_idx]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        conf_ts = torch.from_numpy(np.array(seq_confidence[max_idx])).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(trellis_ts, embed_ts, conf_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = EPS

        if self.random.random() < eps_threshold and mode != 'online':
            return (0, max_idx, max_qvalue)

        for i in candidates:
            trellis_ts = torch.from_numpy(seq_trellis[i]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            conf_ts = torch.from_numpy(np.array(seq_confidence[i])).type(torch.FloatTensor).unsqueeze(0).to(device)
            qvalue = self.forward(trellis_ts, embed_ts, conf_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)

class SepRNN(nn.Module):
    def __init__(self, env, args):
        super(SepRNN, self).__init__()
        
        # set sizes
        embedding_size = env.dataloader.get_embed_size()

        label_size, para_h, para_w = env.tagger.get_sep_para_shape()
    
        rnn_hidden = args.rnn_hidden
        n_filters = 8
        filter_size = [5, 3, 3]
        stride = args.cnn_stride
        features = 20 
        # set random seed
        self.random = random.Random(args.seed_agent)
        
        # CNN for CRF parameters
        self.conv1 = nn.Conv2d(
                in_channels=3,              
                out_channels=n_filters,   
                kernel_size=filter_size[0],              
                stride=stride,        
            )
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, filter_size[1], stride)
        self.bn2 = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*2, filter_size[2], stride)
        self.bn3 = nn.BatchNorm2d(n_filters*2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = filter_size, stride = stride):
            return (size - (kernel_size - 1) - 1) // stride  + 1

        convw = para_w
        convh = para_h

        for f_size in filter_size:
            convw = conv2d_size_out(convw, kernel_size = f_size)
            convh = conv2d_size_out(convh, kernel_size = f_size)

        linear_input_size = convw * convh * n_filters * 2
        
        self.fc1 = nn.Linear(linear_input_size, rnn_hidden)

        self.conv4 = nn.Conv1d(label_size, features, 1, 1)
        self.bn4 = nn.BatchNorm1d(features)
        self.conv5 = nn.Conv1d(label_size, features, 1, 1)
        self.bn5 = nn.BatchNorm1d(features)
        self.fc2 = nn.Linear(features * features, rnn_hidden)
        
        # LSTM for w sequence
        self.rnn = nn.LSTM(
            input_size=embedding_size ,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1)
    
    def forward(self, status_x, transform_x, seq_x):
        # CNN
        #status_x2 = torch.mat(status_x, status_x)
        x1 = F.relu(self.bn1(self.conv1(status_x)))
        x1 = F.relu(self.bn2(self.conv2(x1)))
        x1 = F.relu(self.bn3(self.conv3(x1)))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        
        x2 = F.relu(self.bn4(self.conv4(transform_x))).permute(0, 2, 1)
        x2 = F.relu(self.bn5(self.conv5(x2)))
        x2 = F.relu(self.fc2(x2.view(x2.size(0), -1)))

        r_out,_ = self.rnn(seq_x, None) 
        x3 = r_out[:, -1, :]
        #print(x1.size(), x2.size(), x3.size())
        #x = torch.cat((x1, x2, x3), 1)
        x = x1+x2+x3
        return self.fc(x) # flatten the output
    
    def get_action(self, state, device, mode):
        self.eval()
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, train, rest_budget]
        seq_embedding, seq_confidence, seq_trellis, tagger_para, queried, scope, budget = state
        # note: index in training set
        #       index in data should be: scope[i] for i in candidates
        candidates = [i for i, idx in enumerate(scope) if idx not in queried]
        
        # use TE to explore
        max_idx = candidates[np.argmin([seq_confidence[i] for i in candidates])]

        status_para, trans_para = tagger_para
        status_para_ts = torch.from_numpy(status_para).type(torch.FloatTensor).unsqueeze(0).to(device)
        trans_para_ts = torch.from_numpy(trans_para).type(torch.FloatTensor).unsqueeze(0).to(device)
        #para_ts = torch.from_numpy(tagger_para).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        embed_ts = torch.from_numpy(seq_embedding[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_qvalue = self.forward(status_para_ts, trans_para_ts, embed_ts).detach().item()
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = EPS
    
        if self.random.random() < eps_threshold and mode != 'online':
            return (0, max_idx, max_qvalue)

        for i in candidates:
            embed_ts = torch.from_numpy(seq_embedding[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            qvalue = self.forward(status_para_ts, trans_para_ts, embed_ts).detach().item()
            if max_qvalue < qvalue:
                max_qvalue = qvalue
                max_idx = i
        return (1, max_idx, max_qvalue)