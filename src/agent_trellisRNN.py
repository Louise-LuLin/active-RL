import math
import random
import numpy as np
from collections import deque
import warnings; warnings.simplefilter('ignore')
import copy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# Hyper Parameters:
GAMMA = 0.999 # decay rate of past observation
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 10

TARGET_UPDATE = 15  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 50  # number of previous transitions to remember
BATCH_SIZE = 5  # size of minibatch
LR = 0.001

class TrellisRNN(nn.Module): # all
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisRNN, self).__init__()
        
        para_h, para_w = trellis_shape
        self.maxlen = para_h
        
        # MLP for confidence
        self.fc1 = nn.Linear(1, rnn_hidden, bias=True)
        # LSTM for trellis
        self.rnn1 = nn.LSTM(
            input_size=para_w,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        # LSTM for w sequence
        self.rnn2 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x, seq_x, conf_x):
        # CNN
        r1_out, _ = self.rnn1(trellis_x, None) 
        r2_out, _ = self.rnn2(seq_x, None) 
        # output of last time step
#         x1 = F.relu(r1_out.view(self.maxlen, -1))
#         x2 = F.relu(r2_out.view(self.maxlen, -1))
        x1 = F.relu(r1_out[:, -1, :]) # only the last hidden layer
        x2 = F.relu(r2_out[:, -1, :])
        x3 = F.relu(self.fc1(conf_x))
    
        x = x1 + x2 + x3
        
        return self.fc(x) # flatten the output

class TrellisRNN1(nn.Module): #  trellis cancatenates seq + conf
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisRNN1, self).__init__()
        
        para_h, para_w = trellis_shape
        
        # MLP for confidence
        self.fc1 = nn.Linear(1, rnn_hidden, bias=True)
        # LSTM for trellis
        self.rnn = nn.LSTM(
            input_size=para_w+embedding_size,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x, seq_x, conf_x):
        x1 = torch.cat((trellis_x, seq_x), 2)
        r1_out, _ = self.rnn(x1, None) 
        x1 = r1_out[:, -1, :]
        x2 = F.relu(self.fc1(conf_x))
        x = x1 + x2
        
        return self.fc(x) # flatten the output
    
class TrellisRNN2(nn.Module): #  trellis + seq
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisRNN2, self).__init__()
        
        para_h, para_w = trellis_shape
        
        # MLP for confidence
        self.fc1 = nn.Linear(1, rnn_hidden, bias=True)
        # LSTM for trellis
        self.rnn1 = nn.LSTM(
            input_size=para_w,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        # LSTM for w sequence
        self.rnn2 = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x, seq_x):
        # CNN
        r1_out, _ = self.rnn1(trellis_x, None) 
        r2_out, _ = self.rnn2(seq_x, None) 
        # output of last time step
        x1 = r1_out[:, -1, :]
        x2 = r2_out[:, -1, :]
    
        x = x1 + x2
        
        return self.fc(x) # flatten the output
    

class TrellisRNN3(nn.Module): # only trellis
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisRNN3, self).__init__()
        
        para_h, para_w = trellis_shape
        
        # MLP for confidence
        self.fc1 = nn.Linear(1, rnn_hidden, bias=True)
        # LSTM for trellis
        self.rnn1 = nn.LSTM(
            input_size=para_w,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        
        # Fully connected layer
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x):
        # CNN
        r1_out, _ = self.rnn1(trellis_x, None) 
        # output of last time step
        x1 = r1_out[:, -1, :]
    
        x = x1
        
        return self.fc(x) # flatten the output
    
    
class AgentTrellisRNN(nn.Module):
    def __init__(self, model = 'trellisCNN', greedy = 'te', max_len=50, embedding_size=200, parameter_shape=(5,5), rnn_hidden = 16, n_filters = 4, filter_size = 3, stride = 2):
        print("=== Agent: created")
        super(AgentTrellisRNN, self).__init__()
        self.random = random.Random(10)
        # replay memory
        self.replay_buffer = deque()
        self.time_step = 0
        self.greedy = greedy
        self.model = model
        
        if self.model == 'trellisRNN':
            self.policy_net = TrellisRNN(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
        elif self.model == 'trellisRNN1':
            self.policy_net = TrellisRNN1(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
        elif self.model == 'trellisRNN2':
            self.policy_net = TrellisRNN2(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
        else:
            self.policy_net = TrellisRNN3(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
            
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        para_size = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        
        if self.model == 'trellisRNN':
            print ('trellis RNN model: trellis + seq + conf')
        elif self.model == 'trellisRNN1':
            print ('trellis1 RNN model: (trellis cat seq) + conf')
        elif self.model == 'trellisRNN2':
            print ('trellis2 RNN model: trellis + seq')
        else:
            print ('trellis3 model: trellis')
        print ('Q-net parameter size: {}'.format(para_size))

    def get_action(self, observation):
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, self.queried]
        seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope = observation
        candidates = list(set(scope)-set(queried))
        
#         max_idx = np.argsort(np.array(seq_confidences), kind='mergesort').tolist()[::-1][0]
        if self.greedy == 'rand':
            max_idx = self.random.choice(candidates)
        else:
            conf_tmp = [seq_confidences[i][0] for i in candidates]
            max_idx = candidates[np.argmax(conf_tmp)]
                    
        seq_trellis_ts = torch.from_numpy(seq_trellis[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        seq_embed_ts = torch.from_numpy(seq_embeddings[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        seq_conf_ts = torch.from_numpy(seq_confidences[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        
        if self.model == 'trellisRNN':
            max_q_value = self.policy_net(seq_trellis_ts, seq_embed_ts, seq_conf_ts)
        elif self.model == 'trellisRNN1':
            max_q_value = self.policy_net(seq_trellis_ts, seq_embed_ts, seq_conf_ts)
        elif self.model == 'trellisRNN2':
            max_q_value = self.policy_net(seq_trellis_ts, seq_embed_ts)
        else:
            max_q_value = self.policy_net(seq_trellis_ts)
            
#         eps_threshold = EPS_END + (EPS_START - EPS_END) * \
#             math.exp(-1. * self.time_step / EPS_DECAY)
        eps_threshold = 0.3
        
        if self.random.random() < eps_threshold:
            return (0, max_idx, max_q_value)

        for i in candidates:
            seq_embed_ts = torch.from_numpy(seq_embeddings[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            seq_trellis_ts = torch.from_numpy(seq_trellis[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            seq_conf_ts = torch.from_numpy(seq_confidences[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            
            if self.model == 'trellisRNN':
                q_value = self.policy_net(seq_trellis_ts, seq_embed_ts, seq_conf_ts)
            elif self.model == 'trellisRNN1':
                q_value = self.policy_net(seq_trellis_ts, seq_embed_ts, seq_conf_ts)
            elif self.model == 'trellisRNN2':
                q_value = self.policy_net(seq_trellis_ts, seq_embed_ts)
            else:
                q_value = self.policy_net(seq_trellis_ts)
            if max_q_value < q_value:
                max_q_value = q_value
                max_idx = i
        return (1, max_idx, max_q_value)

    def update(self, observation, action, reward, observation2, terminal):
        self.current_state = observation
        new_state = observation2
        self.replay_buffer.append((self.current_state, action, reward, new_state, terminal))
        if len(self.replay_buffer) > REPLAY_MEMORY_SIZE:
            self.replay_buffer.popleft()
        
        if len(self.replay_buffer) > 5:
            self.train_qnet()
        if self.time_step % TARGET_UPDATE == 0:
            self.target_net = copy.deepcopy(self.policy_net)

        self.current_state = new_state
        self.time_step += 1
        
    def train_qnet(self):
        # experience = (self.current_state, action, reward, new_state, terminal)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope)
        minibatch = self.random.sample(self.replay_buffer, min(len(self.replay_buffer), BATCH_SIZE))
        
        trellis_batch = torch.from_numpy(np.array([experience[0][2][experience[1]] for experience in minibatch])).type(torch.FloatTensor).to(device) # (batch, channel, height, width)
        action_batch = torch.from_numpy(np.array([experience[0][0][experience[1]] for experience in minibatch])).type(torch.FloatTensor).to(device)
        conf_batch = torch.from_numpy(np.array([experience[0][1][experience[1]] for experience in minibatch])).type(torch.FloatTensor).to(device)
        
        # Compute Q(s_t, a)
        if self.model == 'trellisRNN':
            qvalue_batch = self.policy_net(trellis_batch, action_batch, conf_batch)
        elif self.model == 'trellisRNN1':
            qvalue_batch = self.policy_net(trellis_batch, action_batch, conf_batch)
        elif self.model == 'trellisRNN2':
            qvalue_batch = self.policy_net(trellis_batch, action_batch)
        else:
            qvalue_batch = self.policy_net(trellis_batch)
        
        
        # Compute max Q'(s_t+1, a)
        reward_batch = [experience[2] for experience in minibatch]
        y_qvalue_batch = []
        for i, experience in enumerate(minibatch):
            if experience[4]:
                y_qvalue_batch.append(reward_batch[i])
            else:
                candidates = list(set(experience[3][5])-set(experience[3][4]))
                qvalues = []
                for k in candidates:
                    cur_trellis = torch.from_numpy(experience[3][2][k]).type(torch.FloatTensor).unsqueeze(0).to(device)
                    cur_action = torch.from_numpy(experience[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(device)
                    cur_conf = torch.from_numpy(experience[3][1][k]).type(torch.FloatTensor).unsqueeze(0).to(device)
                    
                    if self.model == 'trellisRNN':
                        qvalue = self.target_net(cur_trellis, cur_action, cur_conf)
                    elif self.model == 'trellisRNN1':
                        qvalue = self.policy_net(cur_trellis, cur_action, cur_conf)
                    elif self.model == 'trellisRNN2':
                        qvalue = self.policy_net(cur_trellis, cur_action)
                    else:
                        qvalue = self.policy_net(cur_trellis)
            
                    qvalues.append(qvalue.cpu().detach().item())
                y_qvalue_batch.append(max(qvalues)* GAMMA + reward_batch[i])
                
        self.policy_net.train()
        y_qvalue_batch = torch.from_numpy(np.array(y_qvalue_batch)).type(torch.FloatTensor).to(device)
        loss = F.mse_loss(qvalue_batch, y_qvalue_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
