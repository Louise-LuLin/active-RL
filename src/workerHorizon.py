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

from agent import ParamRNN, ParamRNNBudget, TrellisCNN, PAL
from environment import LabelEnv

# step of update lnet (and push to gnet)
UPDATE_GLOBAL_ITER = 4
# decay rate of past observation
GAMMA = 0.999
# step of update target net
UPDATE_TARGET_ITER = 10
# number of previous transitions to remember
REPLAY_BUFFER_SIZE = 50
# size of minibatch
BATCH_SIZE = 5 

class WorkerHorizon(mp.Process):
    def __init__(self, mode, device, gnets, opts, args, global_ep, global_ep_r, res_queue, pid):
        super(WorkerHorizon, self).__init__()
        self.mode = mode
        self.device = device
        self.id = pid
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnets, self.opts = gnets, opts
        self.env = LabelEnv(args, self.mode)
        self.lnets = []
        self.buffers = []
        for i in range(args.budget):
            agent = TrellisCNN(self.env, args).to(self.device)
            # store all agents
            agent.load_state_dict(self.gnets[i].state_dict())
            self.lnets.append(agent)
            self.buffers.append(deque())
        self.random = random.Random(self.id + args.seed_batch)
        # episode
        self.max_ep = args.episode_train if self.mode == 'offline' else args.episode_test

    def run(self):      
        total_step = 1
        ep = 1
        while self.g_ep.value < self.max_ep:
            state = self.env.start(self.id + ep)
            ep_r = 0
            res_cost = []
            res_explore = []
            res_qvalue = []
            res_reward = []
            res_acc_test = []
            res_acc_valid = []
            while True:
                # play one step
                horizon = self.env.get_horizon()
                explore_flag, action, qvalue = self.lnets[horizon-1].get_action(state, self.device, self.mode)
                reward, state2, done = self.env.feedback(action)
                self.push_to_buffer(state, action, reward, state2, done, horizon)
                state = state2
                # record results
                ep_r += reward
                (acc_test, acc_valid) = self.env.eval_tagger()
                res_cost.append(len(self.env.queried))
                res_explore.append(explore_flag)
                res_qvalue.append(qvalue)
                res_reward.append(ep_r)
                res_acc_test.append(acc_test)
                res_acc_valid.append(acc_valid)
                # sync
                self.update(horizon)
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    print ("--{} {}: ep={}, left={}".format(self.device, self.id, self.g_ep.value, state[-1]))
                if done:
                    self.record(res_cost, res_explore, res_qvalue, res_reward, res_acc_test, res_acc_valid)
                    print ('cost: {}'.format(res_cost))
                    print ('explore: {}'.format(res_explore))
                    print ('qvalue: {}'.format(res_qvalue))
                    print ('reward: {}'.format(res_reward))
                    print ('acc_test: {}'.format(res_acc_test))
                    print ('acc_valid: {}'.format(res_acc_valid))
                    ep += 1
                    break
                total_step += 1
        self.res_queue.put(None)
    
    # push new experience to the buffer
    def push_to_buffer(self, state, action, reward, state2, done, horizon):
        self.buffers[horizon-1].append((state, action, reward, state2, done))
        if len(self.buffers[horizon-1]) > REPLAY_BUFFER_SIZE:
            self.buffers[horizon-1].popleft()
    
    # construct training batch (of y and qvalue)
    def sample_from_buffer(self, batch_size, horizon):
        # experience = (state, action, reward, state2, done)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope, rest_budget)
        minibatch = self.random.sample(self.buffers[horizon-1], min(len(self.buffers[horizon-1]), batch_size))
        t_batch = torch.from_numpy(np.array([e[0][2][e[1]] for e in minibatch])).type(torch.FloatTensor).unsqueeze(1).to(self.device)
        a_batch = torch.from_numpy(np.array([e[0][0][e[1]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        c_batch = torch.from_numpy(np.array([[e[0][1][e[1]]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        # compute Q(s_t, a)
        q_batch = self.lnets[horizon-1](t_batch, a_batch, c_batch)
        # compute max Q'(s_t+1, a)
        r_batch = [e[2] for e in minibatch]
        y_batch = []
        for i, e in enumerate(minibatch):
            if e[4]:
                y_batch.append(r_batch[i])
            else:
                candidates = [k for k, idx in enumerate(e[3][5]) if idx not in e[3][4]]
                q_values = []
                for k in candidates:
                    t = torch.from_numpy(e[3][2][k]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(self.device)
                    a = torch.from_numpy(e[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    c = torch.from_numpy(np.array(e[3][1][k])).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    q = self.lnets[horizon-2](t, a, c).detach().item()
                    q_values.append(q)
                y_batch.append(max(q_values) * GAMMA + r_batch[i])
        y_batch = torch.from_numpy(np.array(y_batch)).type(torch.FloatTensor).to(self.device)
        return q_batch, y_batch
                
    def update(self, horizon):
        self.lnets[horizon-1].train()
        q_batch, y_batch = self.sample_from_buffer(BATCH_SIZE, horizon)
        loss = F.mse_loss(q_batch, y_batch)
        # set gnet grad = 0
        self.opts[horizon-1].zero_grad() 
        # compute lnet gradients
        loss.backward() 
        for param in self.lnets[horizon-1].parameters():
            param.grad.data.clamp_(-1, 1)
#         torch.nn.utils.clip_grad_norm_(self.l_agent.parameters(), MAX_GD_NORM)
        # update gnet's grad with lnet's grad
        for lp, gp in zip(self.lnets[horizon-1].parameters(), self.gnets[horizon-1].parameters()):
            if gp.grad is not None: # if is not cleared (not zero_grad())
                return
            gp._grad = lp.grad
        # update gnet one step forward
        self.opts[horizon-1].step()
        # pull gnet's parameters to local
        if self.mode == 'offline':
            self.lnets[horizon-1].load_state_dict(self.gnets[horizon-1].state_dict())
        
        
    def record(self, res_cost, res_explore, res_qvalue, res_reward, res_acc_test, res_acc_valid):
        with self.g_ep.get_lock():
            self.g_ep.value += 1
        res = (self.g_ep.value, res_cost, res_explore, res_qvalue, res_reward, res_acc_test, res_acc_valid)
        self.res_queue.put(res)
        # monitor
        with self.g_ep_r.get_lock():
            if self.g_ep_r.value == 0.:
                self.g_ep_r.value = res_reward[-1]
            else:
                self.g_ep_r.value = self.g_ep_r.value * 0.9 + res_reward[-1] * 0.1
        print("*** {} {} complete ep {} | ep_r={}".format(self.device, self.pid, self.g_ep.value, self.g_ep_r.value))
                    
