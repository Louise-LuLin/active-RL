import sys
sys.path.insert(1, './src')
import os

from environment import LabelEnv
from agent import ParamRNN, ParamRNNBudget, TrellisCNN, PAL, SepRNN, TrellisBudget
import agent
from sharedAdam import SharedAdam
from worker import WorkerParam, WorkerBudget, WorkerTrellis, WorkerSep, WorkerHeur, WorkerTrellisBudget, WorkerSupervised

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
parser.add_argument('--budget', type=int, default=20,
                   help='budget size')
parser.add_argument('--seed-data', type=int, default=8,
                   help='random seed for data separation')
parser.add_argument('--seed-batch', type=int, default=8,
                   help='random seed for batch sampling')
parser.add_argument('--seed-agent', type=int, default=8,
                   help='random seed for agent epsilon greedy')
parser.add_argument('--init', type=int, default=10,
                   help='pretrain size')
# agent set
parser.add_argument('--model', default='PAL',
                   help='dqn net: ParamRNN, ParamRNNBudget, SepRNN, TrellisCNN, TrellisBudget, TrellisSupervised')
parser.add_argument('--feature', default='all', 
                   help='use feature parameter: all, node, edge')
parser.add_argument('--reweight', default='valid2Vx',
                   help='reweight reward: [valid, test]2[V, T][x,y]')
# NN parameters
parser.add_argument('--rnn-hidden', type=int, default=128,
                   help='hidden size in RNN')
parser.add_argument('--cnn-flt-n', type=int, default=16,
                   help='number of filters in CNN')
parser.add_argument('--cnn-flt-size', type=int, default=3,
                   help='size of filters in CNN')
parser.add_argument('--cnn-stride', type=int, default=1,
                   help='stride in CNN')
# server
parser.add_argument('--cuda', type=int, default=0, 
                   help='which cuda to use')
parser.add_argument('--episode-train', type=int, default=10,
                   help='training episode number')
parser.add_argument('--episode-test', type=int, default=10,
                   help='test episode number')
parser.add_argument('--worker-n', type=int, default=5,
                   help='worker number')

def main():
    args = parser.parse_args()
    '''
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device = torch.device("cuda:{}".format(args.cuda))
    else:
        device = torch.device("cpu")
    
    
    # important! Without this, lnet in worker cannot forward
    # spawn: for unix and linux; fork: for unix only
    mp.set_start_method('spawn')
    
    # === multiprocessing ====
    # global agent
    
    if args.model == 'ParamRNN':
        agent = ParamRNN(LabelEnv(args, None), args).to(device)
    elif args.model == 'ParamRNNBudget':
        agent = ParamRNNBudget(LabelEnv(args, None), args).to(device)
    elif args.model == 'TrellisCNN' or args.model == 'TrellisSupervised':
        agent = TrellisCNN(LabelEnv(args, None), args).to(device)
    elif args.model == 'TrellisBudget':
        agent = TrellisBudget(LabelEnv(args, None), args).to(device)
    elif args.model == 'PAL':
        agent = PAL(LabelEnv(args, None), args).to(device)
    elif args.model == 'SepRNN':
        agent = SepRNN(LabelEnv(args, None), args).to(device)
    elif args.model == 'Rand' or args.model == 'TE':
        agent = None
    else:
        print ("agent model {} not implemented!!".format(args.model))
        return
    # optimizer for global model
    opt = SharedAdam(agent.parameters(), lr=0.001) if agent else None
    # share the global parameters
    if agent:
        agent.share_memory() 
        para_size = sum(p.numel() for p in agent.parameters() if p.requires_grad)
        print ('global parameter size={}'.format(para_size))
    
    # offline train agent for args.episode_train rounds
    start_time = time.time()
    global_ep, global_ep_r, res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
    if args.model == 'ParamRNN':
        tr_workers = [WorkerParam('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'ParamRNNBudget':
        tr_workers = [WorkerBudget('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'TrellisCNN' or args.model == 'PAL':
        tr_workers = [WorkerTrellis('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'TrellisSupervised':
        tr_workers = [WorkerSupervised('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'TrellisBudget':
        tr_workers = [WorkerTrellisBudget('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'SepRNN':
        tr_workers = [WorkerSep('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    tr_result = []
    if agent:
        [w.start() for w in tr_workers]
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
    if args.model == 'ParamRNN':
        ts_workers = [WorkerParam('online', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'ParamRNNBudget':
        ts_workers = [WorkerBudget('online', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'TrellisCNN' or args.model == 'PAL':
        ts_workers = [WorkerTrellis('online', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'TrellisSupervised':
        ts_workers = [WorkerSupervised('online', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'TrellisBudget':
        ts_workers = [WorkerTrellisBudget('offline', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'SepRNN':
        ts_workers = [WorkerSep('online', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]
    elif args.model == 'Rand' or args.model == 'TE':
        ts_workers = [WorkerHeur('online', device, agent, opt, args, global_ep, global_ep_r, res_queue, pid) for pid in range(args.worker_n)]

    ts_result = []
    [w.start() for w in ts_workers]
    while True:
        res = res_queue.get()
        if res is not None:
            ts_result.append(res)
        else:
            break
    [w.join() for w in ts_workers]
    print ("Testing Done! Cost {} for {} episodes.".format(time.time()-start_time, args.episode_test))
    ''' 
    lbenv = LabelEnv(args, 'online')
    greedy_gt, greedy_acc = lbenv.get_greedy_ground_truth()
    print("greedy_acc = {}, the episode is {}".format(greedy_acc, greedy_gt))
    
    lbenv = LabelEnv(args, 'online')
    comb_gt, comb_acc = lbenv.get_combination_ground_truth(args.budget)
    print("comb_acc = {}, the episode is {}".format(comb_acc, comb_gt))
    '''
    num = "num" if args.num_flag else ""
    emb = "embed" if args.embed_flag else ""
    filename = "./results_mp/" + args.data + num + emb + "_" + args.model + "_" \
                + str(args.budget) + "bgt_" + str(args.init) + "init_" \
                + str(args.episode_train) + "trainEp_" + str(args.episode_test) + "testEp"
    
    with open(filename + ".mp", "wb") as result:
        # format: 
        # tr_result = [res]
        # res = (g_ep, cost, qvalue, r, acc_test, acc_valid)
        pickle.dump((tr_result, ts_result), result)
    '''
if __name__ == '__main__':
    main()
