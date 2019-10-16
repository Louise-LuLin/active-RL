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

class TrellisCNN(nn.Module): # all
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisCNN, self).__init__()
        
        para_w, para_h = trellis_shape
        
        # CNN for CRF trellis
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=(para_h, filter_size),              
                stride=stride,          # input&output (batch, channel, height, width)
            )
        self.bn1 = nn.BatchNorm2d(n_filters)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_w_out(size, kernel_w_size = 3, stride = stride):
            return (size - (kernel_w_size - 1) - 1) // stride  + 1
        def conv2d_h_out(size, kernel_h_size = para_h, stride = stride):
            return (size - (kernel_h_size - 1) - 1) // stride  + 1
        
        convw = conv2d_w_out(para_w)
        convh = conv2d_h_out(para_h)
        linear_input_size = convw * convh * n_filters
        
        self.fc1 = nn.Linear(linear_input_size, rnn_hidden, bias=True)
        self.fc2 = nn.Linear(1, rnn_hidden, bias=True)
        
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
    def forward(self, trellis_x, seq_x, conf_x):
        # CNN
        x1 = F.relu(self.bn1(self.conv1(trellis_x)))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size) 
        # h_c shape (n_layers, batch, hidden_size)
        r_out,_ = self.rnn(seq_x, None) 
        # output of last time step
#         rnn_out = self.out(r_out[:, -1, :])
        x2 = r_out[:, -1, :]
        
        x3 = F.relu(self.fc2(conf_x))
    
#         x3 = self.fc2(conf_test)
#         x = torch.cat((x1, x2, x3), 1)
#         x = torch.cat((x1, x2), 1)
        x = x1 + x2 + x3
        
        return self.fc(x) # flatten the output
    
class TrellisCNN1(nn.Module): # only seq and conf
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisCNN1, self).__init__()
        self.fc1 = nn.Linear(1, rnn_hidden, bias=True)
        self.rnn = nn.LSTM(
            input_size=embedding_size,
            hidden_size=rnn_hidden, 
            num_layers=1,
            batch_first=True,  # input＆output (batch，time_step，input_size)
        )
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    def forward(self, seq_x, conf_x):
        r_out,_ = self.rnn(seq_x, None) 
        x1 = r_out[:, -1, :]
        x2 = F.relu(self.fc1(conf_x))
        x = x1 + x2
        
        return self.fc(x) # flatten the output

class TrellisCNN2(nn.Module): # only trellis and seq
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisCNN2, self).__init__()
        
        para_w, para_h = trellis_shape
        
        # CNN for CRF trellis
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=(para_h, filter_size),              
                stride=stride,          # input&output (batch, channel, height, width)
            )
        self.bn1 = nn.BatchNorm2d(n_filters)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_w_out(size, kernel_w_size = 3, stride = stride):
            return (size - (kernel_w_size - 1) - 1) // stride  + 1
        def conv2d_h_out(size, kernel_h_size = para_h, stride = stride):
            return (size - (kernel_h_size - 1) - 1) // stride  + 1
        
        convw = conv2d_w_out(para_w)
        convh = conv2d_h_out(para_h)
        linear_input_size = convw * convh * n_filters
        
        self.fc1 = nn.Linear(linear_input_size, rnn_hidden, bias=True)
        
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
    def forward(self, trellis_x, seq_x):
        # CNN
        x1 = F.relu(self.bn1(self.conv1(trellis_x)))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        
        r_out,_ = self.rnn(seq_x, None) 
        x2 = r_out[:, -1, :]
            
        x = x1 + x2
        
        return self.fc(x) # flatten the output

class TrellisCNN3(nn.Module): # only trellis
    def __init__(self, max_len=50, embedding_size=200, trellis_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3, stride=2):
        super(TrellisCNN3, self).__init__()
        
        para_w, para_h = trellis_shape
        
        # CNN for CRF trellis
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=(para_h, filter_size),              
                stride=stride,          # input&output (batch, channel, height, width)
            )
        self.bn1 = nn.BatchNorm2d(n_filters)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_w_out(size, kernel_w_size = 3, stride = stride):
            return (size - (kernel_w_size - 1) - 1) // stride  + 1
        def conv2d_h_out(size, kernel_h_size = para_h, stride = stride):
            return (size - (kernel_h_size - 1) - 1) // stride  + 1
        
        convw = conv2d_w_out(para_w)
        convh = conv2d_h_out(para_h)
        linear_input_size = convw * convh * n_filters
        
        self.fc1 = nn.Linear(linear_input_size, rnn_hidden, bias=True)
        self.fc = nn.Linear(rnn_hidden, 1, bias=True)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, trellis_x):
        # CNN
        x1 = F.relu(self.bn1(self.conv1(trellis_x)))
        x1 = F.relu(self.fc1(x1.view(x1.size(0), -1)))
        
        x = x1
        
        return self.fc(x) # flatten the output
    
class AgentTrellisCNN(nn.Module):
    def __init__(self, model = 'trellisCNN', greedy = 'te', max_len=50, embedding_size=200, parameter_shape=(5,5), rnn_hidden = 16, n_filters = 4, filter_size = 3, stride = 2):
        print("=== Agent: created")
        super(AgentTrellisCNN, self).__init__()
        self.random = random.Random(10)
        # replay memory
        self.replay_buffer = deque()
        self.time_step = 0
        self.greedy = greedy
        self.model = model
        
        if self.model == 'trellisCNN':
            self.policy_net = TrellisCNN(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
        elif self.model == 'trellisCNN1':
            self.policy_net = TrellisCNN1(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
        elif self.model == 'trellisCNN2':
            self.policy_net = TrellisCNN2(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
        else:
            self.policy_net = TrellisCNN3(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size, stride).to(device)
            
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        para_size = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        
        if self.model == 'trellisCNN':
            print ('trellis model: trellis + seq + conf')
        elif self.model == 'trellisCNN1':
            print ('trellis1 model: seq + conf')
        elif self.model == 'trellisCNN2':
            print ('trellis2 model: trellis + seq')
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
                    
        seq_trellis_ts = torch.from_numpy(seq_trellis[max_idx]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        seq_embed_ts = torch.from_numpy(seq_embeddings[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        seq_conf_ts = torch.from_numpy(seq_confidences[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        
        if self.model == 'trellisCNN':
            max_q_value = self.policy_net(seq_trellis_ts, seq_embed_ts, seq_conf_ts)
        elif self.model == 'trellisCNN1':
            max_q_value = self.policy_net(seq_embed_ts, seq_conf_ts)
        elif self.model == 'trellisCNN2':
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
            seq_trellis_ts = torch.from_numpy(seq_trellis[i]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
            seq_conf_ts = torch.from_numpy(seq_confidences[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            
            if self.model == 'trellisCNN':
                q_value = self.policy_net(seq_trellis_ts, seq_embed_ts, seq_conf_ts)
            elif self.model == 'trellisCNN1':
                q_value = self.policy_net(seq_embed_ts, seq_conf_ts)
            elif self.model == 'trellisCNN2':
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
        
        trellis_batch = torch.from_numpy(np.array([experience[0][2][experience[1]] for experience in minibatch])).type(torch.FloatTensor).unsqueeze(1).to(device) # (batch, channel, height, width)
        action_batch = torch.from_numpy(np.array([experience[0][0][experience[1]] for experience in minibatch])).type(torch.FloatTensor).to(device)
        conf_batch = torch.from_numpy(np.array([experience[0][1][experience[1]] for experience in minibatch])).type(torch.FloatTensor).to(device)
        
        # Compute Q(s_t, a)
        if self.model == 'trellisCNN':
            qvalue_batch = self.policy_net(trellis_batch, action_batch, conf_batch)
        elif self.model == 'trellisCNN1':
            qvalue_batch = self.policy_net(action_batch, conf_batch)
        elif self.model == 'trellisCNN2':
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
                    cur_trellis = torch.from_numpy(experience[3][2][k]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
                    cur_action = torch.from_numpy(experience[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(device)
                    cur_conf = torch.from_numpy(experience[3][1][k]).type(torch.FloatTensor).unsqueeze(0).to(device)
                    
                    if self.model == 'trellisCNN':
                        qvalue = self.target_net(cur_trellis, cur_action, cur_conf)
                    elif self.model == 'trellisCNN1':
                        qvalue = self.policy_net(cur_action, cur_conf)
                    elif self.model == 'trellisCNN2':
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
