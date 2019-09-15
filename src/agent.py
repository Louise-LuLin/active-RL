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
EPS_DECAY = 20

TARGET_UPDATE = 15  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 50  # number of previous transitions to remember
BATCH_SIZE = 5  # size of minibatch
LR = 0.001

class ParamRNN(nn.Module):
    def __init__(self, max_len=50, embedding_size=200, parameter_shape=(5,5), rnn_hidden = 64, n_filters = 16, filter_size = 3):
        super(ParamRNN, self).__init__()
        
        para_h, para_w = parameter_shape
        # CNN for CRF parameters
        self.conv1 = nn.Conv2d(
                in_channels=1,              
                out_channels=n_filters,   
                kernel_size=filter_size,              
                stride=1,        
            )
        self.bn1 = nn.BatchNorm2d(n_filters)
        self.conv2 = nn.Conv2d(n_filters, n_filters*2, filter_size, 1)
        self.bn2 = nn.BatchNorm2d(n_filters*2)
        self.conv3 = nn.Conv2d(n_filters*2, n_filters*2, filter_size, 1)
        self.bn3 = nn.BatchNorm2d(n_filters*2)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = filter_size, stride = 1):
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
        self.fc = nn.Linear(2 * rnn_hidden, 1)
    
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
        x = torch.cat((x1, x2), 1)
        
        return self.fc(x) # flatten the output


class AgentParamRNN(nn.Module):
    def __init__(self, max_len=50, embedding_size=200, parameter_shape=(5,5), rnn_hidden = 16, n_filters = 4, filter_size = 3):
        print("=== Agent: created")
        super(AgentParamRNN, self).__init__()
        # replay memory
        self.replay_buffer = deque()
        self.time_step = 0
        
        self.policy_net = ParamRNN(max_len, embedding_size, parameter_shape, rnn_hidden, n_filters, filter_size).to(device)
        self.target_net = copy.deepcopy(self.policy_net)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        para_size = sum(p.numel() for p in self.policy_net.parameters() if p.requires_grad)
        print ('Q-net parameter size: {}'.format(para_size))


    
    def get_action(self, observation):
        # observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, self.queried]
        seq_embeddings, seq_confidences, tagger_para, self.queried = observation
        
        max_idx = np.argsort(np.array(seq_confidences), kind='mergesort').tolist()[::-1][0]
        tagger_para_ts = torch.from_numpy(tagger_para).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
        seq_embed_ts = torch.from_numpy(seq_embeddings[max_idx]).type(torch.FloatTensor).unsqueeze(0).to(device)
        max_q_value = self.policy_net(tagger_para_ts, seq_embed_ts)

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.time_step / EPS_DECAY)

        if random.random() < eps_threshold:
            return (0, max_idx, max_q_value)

        for i in range(len(seq_embeddings)):
            seq_embed_ts = torch.from_numpy(seq_embeddings[i]).type(torch.FloatTensor).unsqueeze(0).to(device)
            q_value = self.policy_net(tagger_para_ts, seq_embed_ts)
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
        minibatch = random.sample(self.replay_buffer, min(len(self.replay_buffer), BATCH_SIZE))
        
        state_batch = torch.from_numpy(np.array([experience[0][2] for experience in minibatch])).type(torch.FloatTensor).unsqueeze(1).to(device)
        action_batch = torch.from_numpy(np.array([experience[0][0][experience[1]] for experience in minibatch])).type(torch.FloatTensor).to(device)
        # Compute Q(s_t, a)
        qvalue_batch = self.policy_net(state_batch, action_batch)
        
        # Compute max Q'(s_t+1, a)
        reward_batch = [experience[2] for experience in minibatch]
        y_qvalue_batch = []
        for i, experience in enumerate(minibatch):
            if experience[4]:
                y_qvalue_batch.append(reward_batch[i])
            else:
                cur_state = torch.from_numpy(experience[3][2]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(device)
                qvalues = []
                for seq_embedding in experience[3][0]:
                    cur_action = torch.from_numpy(seq_embedding).type(torch.FloatTensor).unsqueeze(0).to(device)
                    qvalue = self.target_net(cur_state, cur_action)
                    qvalues.append(qvalue.cpu().detach().numpy()[0])
                y_qvalue_batch.append(max(qvalues)* GAMMA + reward_batch[i])
                
        self.policy_net.train()
        y_qvalue_batch = torch.from_numpy(np.array(y_qvalue_batch)).type(torch.FloatTensor).to(device)
        print (qvalue_batch)
        print (y_qvalue_batch)
        loss = F.mse_loss(qvalue_batch, y_qvalue_batch)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
