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

from agent import ParamRNN, ParamRNNBudget, TrellisCNN, PAL, SepRNN
from environment import LabelEnv

# step of update lnet (and push to gnet)
UPDATE_GLOBAL_ITER = 5
# decay rate of past observation
GAMMA = 0.999
# step of update target net
UPDATE_TARGET_ITER = 10
# number of previous transitions to remember
REPLAY_BUFFER_SIZE = 50
# size of minibatch
BATCH_SIZE = 5 

class Worker(mp.Process):
    def __init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid):
        super(Worker, self).__init__()
        self.mode = mode
        self.device = device
        self.id = pid
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.env = LabelEnv(args, self.mode)
        # will be changed to specific model
        self.lnet = self.gnet
        self.target_net = self.lnet
        # replay memory
        self.random = random.Random(self.id + args.seed_batch)
        self.buffer = deque()
        self.time_step = 0
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
                explore_flag, action, qvalue = self.lnet.get_action(state, self.device)
                reward, state2, done = self.env.feedback(action)
                self.push_to_buffer(state, action, reward, state2, done)
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
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:
                    self.update()
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
    def push_to_buffer(self, state, action, reward, state2, done):
        self.buffer.append((state, action, reward, state2, done))
        if len(self.buffer) > REPLAY_BUFFER_SIZE:
            self.buffer.popleft()
    
    # construct training batch (of y and qvalue)
    def sample_from_buffer(self, batch_size):
        # experience = (state, action, reward, state2, done)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope, rest_budget)
        q_batch = torch.ones([1, batch_size], dtype=torch.float64).to(self.device)
        y_batch = torch.ones([1, batch_size], dtype=torch.float64).to(self.device)
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
        if self.mode == 'offline':
            self.lnet.load_state_dict(self.gnet.state_dict())
        # update target_net
        if self.time_step % UPDATE_TARGET_ITER == 0:
            self.target_net = copy.deepcopy(self.lnet)
        self.time_step += 1
        
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

class WorkerParam(Worker):
    def __init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid):
        Worker.__init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid)
        self.lnet = ParamRNN(self.env, args).to(self.device)
        self.lnet.load_state_dict(self.gnet.state_dict())
        self.target_net = copy.deepcopy(self.lnet)
    
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
                candidates = [k for k, idx in enumerate(e[3][5]) if idx not in e[3][4]]
                q_values = []
                for k in candidates:
                    a = torch.from_numpy(e[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    q = self.target_net(s_t, a).detach().item()
                    q_values.append(q)
                y_batch.append(max(q_values) * GAMMA + r_batch[i])
        y_batch = torch.from_numpy(np.array(y_batch)).type(torch.FloatTensor).to(self.device)
        return q_batch, y_batch
    
class WorkerBudget(Worker):
    def __init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid):
        Worker.__init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid)
        self.lnet = ParamRNNBudget(self.env, args).to(self.device)
        self.lnet.load_state_dict(self.gnet.state_dict())
        self.target_net = copy.deepcopy(self.lnet)
    
    # construct training batch (of y and qvalue)
    def sample_from_buffer(self, batch_size):
        # experience = (state, action, reward, state2, done)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope, rest_budget)
        minibatch = self.random.sample(self.buffer, min(len(self.buffer), batch_size))
        s_batch = torch.from_numpy(np.array([e[0][3] for e in minibatch])).type(torch.FloatTensor).unsqueeze(1).to(self.device)
        a_batch = torch.from_numpy(np.array([e[0][0][e[1]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        b_batch = torch.from_numpy(np.array([[e[0][6]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        # compute Q(s_t, a)
        q_batch = self.lnet(s_batch, a_batch, b_batch)
        # compute max Q'(s_t+1, a)
        r_batch = [e[2] for e in minibatch]
        y_batch = []
        for i, e in enumerate(minibatch):
            if e[4]:
                y_batch.append(r_batch[i])
            else:
                s_t = torch.from_numpy(e[3][3]).type(torch.FloatTensor).unsqueeze(0).unsqueeze(0).to(self.device)
                b_t = torch.from_numpy(np.array([e[3][6]])).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                candidates = [k for k, idx in enumerate(e[3][5]) if idx not in e[3][4]]
                q_values = []
                for k in candidates:
                    a = torch.from_numpy(e[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    q = self.target_net(s_t, a, b_t).detach().item()
                    q_values.append(q)
                y_batch.append(max(q_values) * GAMMA + r_batch[i])
        y_batch = torch.from_numpy(np.array(y_batch)).type(torch.FloatTensor).to(self.device)
        return q_batch, y_batch    
       
class WorkerTrellis(Worker):
    def __init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid):
        Worker.__init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid)
        if args.model == 'TrellisCNN':
            self.lnet = TrellisCNN(self.env, args).to(self.device)
        elif args.model == 'PAL':
            self.lnet = PAL(self.env, args).to(self.device)
        self.lnet.load_state_dict(self.gnet.state_dict())
        self.target_net = copy.deepcopy(self.lnet)
    
    # construct training batch (of y and qvalue)
    def sample_from_buffer(self, batch_size):
        # experience = (state, action, reward, state2, done)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope, rest_budget)
        minibatch = self.random.sample(self.buffer, min(len(self.buffer), batch_size))
        t_batch = torch.from_numpy(np.array([e[0][2][e[1]] for e in minibatch])).type(torch.FloatTensor).unsqueeze(1).to(self.device)
        a_batch = torch.from_numpy(np.array([e[0][0][e[1]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        c_batch = torch.from_numpy(np.array([[e[0][1][e[1]]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        # compute Q(s_t, a)
        q_batch = self.lnet(t_batch, a_batch, c_batch)
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
                    q = self.target_net(t, a, c).detach().item()
                    q_values.append(q)
                y_batch.append(max(q_values) * GAMMA + r_batch[i])
        y_batch = torch.from_numpy(np.array(y_batch)).type(torch.FloatTensor).to(self.device)
        return q_batch, y_batch
    
class WorkerSep(Worker):
    def __init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid):
        Worker.__init__(self, mode, device, gnet, opt, args, global_ep, global_ep_r, res_queue, pid)
        self.lnet = SepRNN(self.env, args).to(self.device)
        self.lnet.load_state_dict(self.gnet.state_dict())
        self.target_net = copy.deepcopy(self.lnet)
    
    # construct training batch (of y and qvalue)
    def sample_from_buffer(self, batch_size):
        # experience = (state, action, reward, state2, done)
        # state = (seq_embeddings, seq_confidences, seq_trellis, tagger_para, queried, scope, rest_budget)
        minibatch = self.random.sample(self.buffer, min(len(self.buffer), batch_size))
        s0_batch = torch.from_numpy(np.array([e[0][3][0] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        s1_batch = torch.from_numpy(np.array([e[0][3][1] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        a_batch = torch.from_numpy(np.array([e[0][0][e[1]] for e in minibatch])).type(torch.FloatTensor).to(self.device)
        # compute Q(s_t, a)
        q_batch = self.lnet(s0_batch, s1_batch, a_batch)
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
                    st0 = torch.from_numpy(np.array(e[3][3][0])).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    st1 = torch.from_numpy(np.array(e[3][3][1])).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    a = torch.from_numpy(e[3][0][k]).type(torch.FloatTensor).unsqueeze(0).to(self.device)
                    q = self.target_net(st0, st1, a).detach().item()
                    q_values.append(q)
                y_batch.append(max(q_values) * GAMMA + r_batch[i])
        y_batch = torch.from_numpy(np.array(y_batch)).type(torch.FloatTensor).to(self.device)
        return q_batch, y_batch
    