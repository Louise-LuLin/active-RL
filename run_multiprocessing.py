import sys
sys.path.insert(1, './src')

from environment import LabelEnv
from agent import ParamRNN
from sharedAdam import SharedAdam
from worker import Worker

from gensim.models import KeyedVectors
import numpy as np
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os

import warnings; warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.multiprocessing as mp
from torch.autograd import Variable  

# if gpu is to be used
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
# print ("== Cuda: {} ===".format(torch.cuda.is_available()))
# if torch.cuda.is_available():
#     mp.set_start_method('spawn')

parser = argparse.ArgumentParser(description='Asyncronous DQN')
# environment set: dataset, data split, budget
parser.add_argument('--folder', default="./datasets/", 
                    help='dataset folder')
parser.add_argument('--data', default='sod', 
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
parser.add_argument('--feature', default='all', 
                    help='use feature parameter: all, node, edge')
parser.add_argument('--model', default='ParamRNN', 
                    help='agent model: ParamRNN, trellisRNN, trellisCNN')
parser.add_argument('--reweight', default='valid2Vx',
                   help='reweight reward: [valid, test]2[V, T][x,y]')
# NN parameters
parser.add_argument('--rnn-hidden', type=int, default=64,
                   help='hidden size in RNN')
parser.add_argument('--cnn-flt-n', type=int, default=16,
                   help='number of filters in CNN')
parser.add_argument('--cnn-flt-size', type=int, default=4,
                   help='size of filters in CNN')
parser.add_argument('--cnn-stride', type=int, default=2,
                   help='stride in CNN')
# server
parser.add_argument('--cuda', default=False, 
                    help='using cuda or cpu')            

if __name__ == '__main__':
    args = parser.parse_args()
    
    use_cuda = args.cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    mp.set_start_method('spawn')
    
    # === multiprocessing ====
    # global agent
    agent = ParamRNN(LabelEnv(args, None), args).to(device)
    # share the global parameters
    agent.share_memory() 
    para_size = sum(p.numel() for p in agent.parameters() if p.requires_grad)
    print ('global parameter size={}'.format(para_size))
    # optimizer for global model
    opt = SharedAdam(agent.parameters(), lr=0.001)
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    
    # offline train agent
    tr_workers = [Worker('train', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(5)]
    [w.start() for w in workers]
    tr_result = []
    while True:
        res = res_queue.get()
        if res is not None:
            tr_result.append(res)
        else:
            break
    [w.join() for w in workers]
    print "Training Done!"
    
    # online test agent
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    ts_workers = [Worker('test', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(5)]
    [w.start() for w in workers]
    ts_result = []
    while True:
        res = res_queue.get()
        if res is not None:
            ts_result.append(res)
        else:
            break
    [w.join() for w in workers]
    print "testing Done!"
    
    num = "num" if args.num_flag else ""
    emb = "embed" if args.embed_flag else ""
    filename = "./results_mp/" + args.data + num + emb + "_single_" + str(args.budget) + "bgt_" + str(args.init) + "init"

    with open(filename + ".mp", "wb") as result:
        # format: 
        # tr_result = [res]
        # res = (g_ep, cost, qvalue, r, acc_test, acc_valid)
        pickle.dump((tr_result, ts_result), result)
    