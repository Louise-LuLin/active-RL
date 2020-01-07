import sys
sys.path.insert(1, './src')
import os

from environment import LabelEnv
from agent import ParamRNN, ParamRNNBudget, TrellisCNN, PAL, SepRNN
from sharedAdam import SharedAdam
from workerHorizon import WorkerHorizon

from gensim.models import KeyedVectors
import numpy as np
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle
import time

import warnings; warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.multiprocessing as mp
from torch.autograd import Variable  

parser = argparse.ArgumentParser(description='Asyncronous DQN')
# environment set: dataset, data split, budget
parser.add_argument('--folder', default="./datasets/", 
                   help='dataset folder')
parser.add_argument('--data', default='conll', 
                   help='choose a dataset')
parser.add_argument('--num-flag', default=True, 
                   help='replace number with NUM')
parser.add_argument('--embed-flag', default=True, 
                   help='use embedding or one-hot')
parser.add_argument('--budget', type=int, default=75,
                   help='budget size')
parser.add_argument('--seed-data', type=int, default=8,
                   help='random seed for data separation')
parser.add_argument('--seed-batch', type=int, default=8,
                   help='random seed for batch sampling')
parser.add_argument('--seed-agent', type=int, default=8,
                   help='random seed for agent epsilon greedy')
parser.add_argument('--init', type=int, default=5,
                   help='pretrain size')
# agent set
parser.add_argument('--model', default='TrellisCNN',
                   help='dqn net: ParamRNN, ParamRNNBudget, TrellisCNN, PAL')
parser.add_argument('--feature', default='all', 
                   help='use feature parameter: all, node, edge')
parser.add_argument('--reweight', default='valid2Vx',
                   help='reweight reward: [valid, test]2[V, T][x,y]')
# NN parameters
parser.add_argument('--rnn-hidden', type=int, default=128,
                   help='hidden size in RNN')
parser.add_argument('--cnn-flt-n', type=int, default=16,
                   help='number of filters in CNN')
parser.add_argument('--cnn-flt-size', type=int, default=2,
                   help='size of filters in CNN')
parser.add_argument('--cnn-stride', type=int, default=2,
                   help='stride in CNN')
# server
parser.add_argument('--cuda', type=int, default=0, 
                   help='which cuda to use')
parser.add_argument('--episode-train', type=int, default=50,
                   help='training episode number')
parser.add_argument('--episode-test', type=int, default=10,
                   help='test episode number')

def main():
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:{}".format(args.cuda))
    else:
        device = torch.device("cpu")
    
    
    # important! Without this, lnet in worker cannot forward
    # spawn: for unix and linux; fork: for unix only
    mp.set_start_method('spawn')
    
    # === multiprocessing ====
    # global agents with different budgets
    agents = []
    opts = []
    for i in range(args.budget):
        agent = TrellisCNN(LabelEnv(args, None), args).to(device)
        # share the global parameters
        agent.share_memory() 
        # optimizer for global model
        opt = SharedAdam(agent.parameters(), lr=0.001)
        # store all agents
        agents.append(agent)
        opts.append(opt)
        if i == 0:
            para_size = sum(p.numel() for p in agent.parameters() if p.requires_grad)
            print ('global parameter size={}*{},'.format(para_size, args.budget))
    
    # offline train agent for args.episode_train rounds
    start_time = time.time()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    tr_workers = [WorkerHorizon('offline', device, agents, opts, args, global_ep, global_ep_r, res_queue, pid) for pid in range(5)]
    [w.start() for w in tr_workers]
    tr_result = []
    while True:
        res = res_queue.get()
        if res is not None:
            tr_result.append(res)
        else:
            break
    [w.join() for w in tr_workers]
    print ("Training Done! Cost {} for {} episodes.".format(time.time()-start_time, args.episode_train))
    
    # online test agent for args.episode_test rounds
    start_time = time.time()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    ts_workers = [WorkerHorizon('online', device, agents, opts, args, global_ep, global_ep_r, res_queue, pid) for pid in range(5)]
    [w.start() for w in ts_workers]
    ts_result = []
    while True:
        res = res_queue.get()
        if res is not None:
            ts_result.append(res)
        else:
            break
    [w.join() for w in ts_workers]
    print ("Testing Done! Cost {} for {} episodes.".format(time.time()-start_time, args.episode_test))
    
    num = "num" if args.num_flag else ""
    emb = "embed" if args.embed_flag else ""
    filename = "./results_mp/" + args.data + num + emb + "_" + args.model + "_horizon_" \
                + str(args.budget) + "bgt_" + str(args.init) + "init_" \
                + str(args.episode_train) + "trainEp_" + str(args.episode_test) + "testEp"

    with open(filename + ".mp", "wb") as result:
        # format: 
        # tr_result = [res]
        # res = (g_ep, cost, explore, qvalue, r, acc_test, acc_valid)
        pickle.dump((tr_result, ts_result), result)

if __name__ == '__main__':
    main()
