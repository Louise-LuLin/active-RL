"""
Worker for multiprocessing: each worker keep a local copy of the global agent.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp

import random
import numpy as np
import copy
from collections import deque

from agent import ParamRNN
from environment import LabelEnv

MAX_EP = 50
UPDATE_GLOBAL_ITER = 10
GAMMA = 0.999 # decay rate of past observation
TARGET_UPDATE_ITER = 10  # timesteps to observe before training
REPLAY_MEMORY_SIZE = 50  # number of previous transitions to remember
BATCH_SIZE = 5  # size of minibatch
MAX_GD_NORM = 10    


def record(global_ep, global_ep_r, ep_r, res_queue, pid):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print("*** cpu {} complete ep {} | ep_r={}".format(pid, global_ep.value, global_ep_r.value))
    
class Trainer(mp.Process):
    def __init__(self, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid, seed):
        super(Worker, self).__init__()
        self.device = device
        self.id = pid
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.env = LabelEnv(args, 'train')
        self.lnet = ParamRNN(self.env, args).to(self.device)     # local network
        self.target_net = copy.deepcopy(self.lnet)
        # replay memory
        self.random = random.Random(seed)
        self.buffer = deque()
        self.time_step = 0
        
    def run(self):        
        total_step = 1
        while self.g_ep.value < MAX_EP:
            state = self.env.start(episode + self.id)
            ep_r = 0
            while true:
                greedy_flag, action, qvalue = self.lnet.get_action(state)
                reward, state2, done = self.env.feedback(action)
                ep_r += reward
                self.push_to_buffer(state, action, reward, state2, done)
                state = state2
                # sync
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.update()
                    print ("-- cpu {}: ep={}, left={}".format(self.id, self.g_ep.value, state[-1]))
                if done:
                    record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.pid)
                total_step += 1
        self.res_queue.put(None)
    
    # push new experience to the buffer
    def push_to_buffer(self, state, action, reward, state2, done):
        self.buffer.append((state, action, reward, state2, done))
        if len(self.buffer) > REPLAY_MEMORY_SIZE:
            self.buffer.popleft()
    
    # construct training batch (of y and qvalue)
    def sample_from_buffer(self, batch_size):
        # experience = (state, action, reward, state2, done)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope, rest_budget)
        minibatch = self.random.sample(self.buffer, min(len(self.buffer), batch_size))
        s_batch = torch.from_numpy(np.array([e[0][3] for e in minibatch])).type(torch.FloatTensor).unsqueeze(1).to(self.device)
        a_batch = torch.from_numpy(np.array([e[0][0][e[1]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        # compute Q(s_t, a)
        q_batch = self.lnet(s_batch, a_batch)
        # compute max Q'(s_t+1, a)
        r_batch = [e[2] for e in minibatch]
        y_batch = []
        for i, e in enumerate(minibatch):
            if e[4]:
                y_batch.append(r_batch[i])
            else:
                s_t = torch.from_numpy(e[3][3]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(self.device)
                candidates = list(set(e[3][5])-set(e[3][4]))
                q_values = []
                for k in candidates:
                    a = torch.from_numpy(e[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    q = self.target_net(s_t, a)
                    q_values.append(q.cpu().detach().item())
                y_batch.append(max(q_values) * GAMMA + r_batch[i])
        y_batch = torch.from_numpy(np.array(y_batch)).type(torch.FloatTensor).to(self.device)
        return q_batch, y_batch
                
    def update(self):
        self.lnet.train()
        q_batch, y_batch = self.sample_from_buffer(BATCH_SIZE)
        loss = F.mse_loss(q_batch, y_batch)
        # set gnet grad = 0
        self.opt.zero_grad() 
        # compute lnet gradients
        loss.backward() 
        for param in self.lnet.parameters():
            param.grad.data.clamp_(-1, 1)
#         torch.nn.utils.clip_grad_norm_(self.l_agent.parameters(), MAX_GD_NORM)
        # update gnet's grad with lnet's grad
        for lp, gp in zip(self.lnet.parameters(), self.gnet.parameters()):
            if gp.grad is not None: # if is not cleared (not zero_grad())
                return
            gp._grad = lp.grad
        # update gnet one step forward
        self.opt.step()
        # pull gnet's parameters to local
        self.lnet.load_state_dict(self.gnet.state_dict())
        # update target_net
        if self.time_step % TARGET_UPDATE == 0:
            self.target_net = copy.deepcopy(self.lnet)
        self.time_step += 1
        
    