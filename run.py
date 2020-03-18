import os
import sys
import time
sys.path.insert(1, './src')

from tagger import CrfModel
from data_loader import BuildDataLoader
from environment import LabelEnv
from agent import ParamRNN, AgentParamRNN
from agent_trellisCNN import TrellisCNN, AgentTrellisCNN
from agent_trellisRNN import TrellisRNN, AgentTrellisRNN

from gensim.models import KeyedVectors
import numpy as np
import random
import argparse
import matplotlib
import matplotlib.pyplot as plt
import pickle

import warnings; warnings.simplefilter('ignore')
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable  

# if gpu is to be used
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
print ("== Cuda: {} ===".format(torch.cuda.is_available()))

# ======================================== setting =====================================================
# parser definition
parser = argparse.ArgumentParser(description='Tune sequence active learning')
parser.add_argument('-b', '--budget', type=int, help='budget size', default=75)                
parser.add_argument('-f', '--feature', choices=['all', 'node', 'edge'], help='choose features', default='all') # test/decode TBD
parser.add_argument('-g', '--greedy', help='top m test instances', default='rand')
parser.add_argument('-m', '--model', help='model of encoding state', default='ParamRNN')
parser.add_argument('-r', '--reward', help='shifted reward strategy', default='valid2V')
parser.add_argument('-s', '--source', help='dataset', default='sod')                
parser.add_argument('-u', '--upLoop', type=int, help='increase loop by N', default=10)
parser.add_argument('-d', '--decay', type=int, help='loop decay', default=3)                
parser.add_argument('-n', '--hidden', type=int, help='rnn hidden size', default=64)
parser.add_argument('-w', '--word2vec', type=int, help='use word2vec', default=0)
parser.add_argument('-x', '--replaceX', type=int, help='use x to replace digit', default=1)
parser.add_argument('-e', '--cuda', default='2')

args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda

SOURCE = args.source
DATA_PATH = "./datasets/" + SOURCE
NUM_FLAG = True if args.replaceX == 1 else False
EMBED_FLAG = True if args.word2vec == 1 else False
FEAT = args.feature
MODEL = args.model

dataloader = BuildDataLoader(SOURCE, DATA_PATH, NUM_FLAG, EMBED_FLAG)
crf = CrfModel(dataloader, FEAT)

samples = [4]
VALID_N = 200
TEST_N = 200
TRAIN_N = 600
REWEIGHT = args.reward
GEEDY = args.greedy
BUDGET = args.budget
RNN_HIDDEN = 128
N_FILTER = 16
FILTER_SIZE = 4
N_STRIDE = 2
max_len = dataloader.get_max_len()
embedding_size = dataloader.get_embed_size()
parameter_shape = crf.get_para_shape()
trellis_shape = crf.get_trellis_shape()
print ("max_len is: {}".format(max_len))
print ("crf para shape: {}".format(parameter_shape))
print ("crf trellis shape: {}".format(trellis_shape))

# ======================================== active learning =====================================================
# use trellis CNN to encode state
qvalue_list = []
action_mark_list = []
prob_list = []
for seed in samples:
    env = LabelEnv(dataloader, crf, seed, VALID_N, TEST_N, TRAIN_N, REWEIGHT, BUDGET)
    if MODEL.startswith('trellisCNN'):
        agent = AgentTrellisCNN(MODEL, GEEDY, max_len, embedding_size, trellis_shape, RNN_HIDDEN, N_FILTER, FILTER_SIZE, N_STRIDE)
    elif MODEL.startswith('ParamRNN'):
        agent = AgentParamRNN(GEEDY, max_len, embedding_size, parameter_shape, RNN_HIDDEN, N_FILTER, FILTER_SIZE, N_STRIDE)
    else:
        agent = AgentTrellisRNN(MODEL, GEEDY, max_len, embedding_size, trellis_shape, RNN_HIDDEN, N_FILTER, FILTER_SIZE, N_STRIDE)

    
    print ("=== Start play ===")
    step = 20
    while env.cost < BUDGET:
        env.resume()
        # observ = [self.seq_embeddings, self.seq_confidences, self.seq_trellis, self.tagger_para, self.queried, self.scope]
        
        time1 = time.time()
        observation = env.get_state()
        time2 = time.time()
        greedy_flg, action, q_value = agent.get_action(observation)
        time3 = time.time()
        reward, observation2, terminal = env.feedback(action)
        time4 = time.time()
        (acc_test, acc_valid) = env.eval_tagger()
        time5 = time.time()
        print ("cost {}: queried {} with greedy {}:{}, acc=({}, {})".format(env.cost, action, greedy_flg, 
                                                                            q_value.item(), acc_test, acc_valid))
        print ("time: get_sate={}, get_action={}, feedback={}, eval_tagger={}".format(time2-time1, time3-time2, time4-time3, time5-time4))
        
        qvalue_list.append(q_value.item())
        action_mark_list.append(greedy_flg)
        prob_list.append(observation[1][action])
        
        if env.cost % 10 == 0:
            step += 10
        
        if env.cost < 6:
            continue
         
        for n in range(step):
            env.reboot()
            while env.terminal == False:
                observation = env.get_state()
                greedy_flg, action, q_value = agent.get_action(observation)
                reward, observation2, terminal = env.feedback(action)
                agent.update(observation, action, reward, observation2, terminal)
        time6 = time.time()
        print ("loop time: loop={}, time={}".format(step, time6-time5))
# ======================================== store result =====================================================
cost_list = sorted(env.acc_trace.keys())
acc_list = [env.acc_trace[i][0] for i in cost_list]
acc_valid_list = [env.acc_trace[i][1] for i in cost_list]

num = "num" if NUM_FLAG else ""
emb = "embed" if EMBED_FLAG else ""
filename = "./results_run/" + SOURCE + num + emb + "_" + MODEL + "_" + str(BUDGET) + "bgt_" + GEEDY + "_" + REWEIGHT + "_" + FEAT

with open(filename + ".bin", "wb") as result:
    pickle.dump((cost_list, acc_list, acc_valid_list), result)
    

# ======================================== sanity check =====================================================
x = range(len(qvalue_list))

fig, axes = plt.subplots(ncols=2, nrows=1)
ax = axes.flatten()

ax[0].plot(x, qvalue_list, color='0.5')
x2 = []
y2 = []
for i in x:
    if action_mark_list[i] == 1:
        x2.append(i)
        y2.append(qvalue_list[i])
l1 = ax[0].scatter(x2, y2, color='r', marker='o')
x2 = []
y2 = []
for i in x:
    if action_mark_list[i] == 0:
        x2.append(i)
        y2.append(qvalue_list[i])
l2 = ax[0].scatter(x2, y2, color='g', marker='x')
ax[0].legend((l1,l2),
           ('$action > \\epsilon$', '$action < \\epsilon$'))

ax[0].set_title('{} dataset'.format(SOURCE))
# plt.xlim(0, 20)
ax[0].set_ylabel('Q value')
ax[0].set_xlabel('Sequence number')

# print (qvalue_list)
ax[1].scatter(qvalue_list, prob_list)
for i in range(len(qvalue_list)):
    ax[1].annotate(i, (qvalue_list[i], prob_list[i]))
ax[1].set_xlabel('Q value')
ax[1].set_ylabel('Likelihood')
plt.subplots_adjust(wspace=0.1)
fig.set_size_inches(15,5)
# plt.show()
plt.savefig(filename + '_check.png')
    
    