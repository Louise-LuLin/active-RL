import math
import random
import copy
import numpy as np
from tqdm import tqdm
import scipy
from sklearn.feature_extraction.text import CountVectorizer as CV
import re

from tagger import CrfModel
from data_loader import DataLoader

class LabelEnv:
    def __init__(self, args, mode):
        self.dataloader = DataLoader(args.data, args.folder, args.num_flag, args.embed_flag, args.seed_data)
        self.data = self.dataloader.sequence
        self.model = args.model
        # initialize unlabeled/validation/test set
        if mode == 'offline':
            self.data_idx = self.dataloader.offline_idx[:]
        else:
            self.data_idx = self.dataloader.online_idx[:]
        self.valid = self.data_idx[:100]
        self.train  = self.data_idx[100:]
        self.test = self.dataloader.test_idx
        # initialize tagger
        self.tagger = CrfModel(self.dataloader, args.feature)
        self.init_n = args.init
        self.queried = [i for i in self.train[:self.init_n]]
        self.set_tagger([self.data[i] for i in self.queried])
        self.budget = args.budget
        # initialize reweight strategy
        self.reweight = args.reweight
        self.sim_valid2test = np.zeros((len(self.valid)))
        if self.reweight == 'kmers':
            self.sim_valid2test = self.get_similarity2test()
        # store instance embeddings
        self.seq_embedding = []
        for i in self.train:
            self.seq_embedding.append(self.dataloader.get_embedding(self.data[i]))
    
    # retrain tagger
    def set_tagger(self, seqs):
        self.tagger.reset()
        self.tagger.add_instances(seqs)
        self.tagger.train()
    
    # evaluate tagger on test/valid set
    def eval_tagger(self):
        self.set_tagger([self.data[i] for i in self.queried])
        _, _, _, acc_test = self.tagger.evaluate_acc([self.data[i] for i in self.test])
        _, _, _, acc_valid = self.tagger.evaluate_acc([self.data[i] for i in self.valid])
        return (acc_test, acc_valid)
    
    # start the game
    def start(self, seed):
        # shuffle data
        random.Random(seed).shuffle(self.data_idx)
        self.valid = self.data_idx[:100]
        self.train  = self.data_idx[100:]
        # initialize tagger
        self.queried = [i for i in self.train[:self.init_n]]
        self.set_tagger([self.data[i] for i in self.queried])
        self.acc = self.reweight_acc()
        return self.get_state()
    
    # get current state
    def get_state(self):
        # compute confidence, trellis structure
        seq_confidence = []
        seq_trellis = []
        for seq in [self.data[i] for i in self.train]:
            seq_confidence.append(self.tagger.compute_confidence(seq))
            seq_trellis.append(self.tagger.get_marginal_matrix(seq))
        # construct parameter matrix for current tagger
        if self.model == 'SepRNN':
            tagger_para = self.tagger.get_sep_matrix()
        else:
            tagger_para = self.tagger.get_parameter_matrix()
        observation = [self.seq_embedding, seq_confidence, seq_trellis, tagger_para, 
                       self.queried, self.train, float(self.budget-len(self.queried))/float(self.budget)]
        return observation
    
    def get_horizon(self):
        return self.budget-len(self.queried)
 
    def feedback(self, new_idx):
        # reward
        new_seq_idx = self.train[new_idx]
        self.tagger.add_instances([self.data[new_seq_idx]])
        self.tagger.train()
        new_acc = self.reweight_acc()
        if self.model == 'TrellisSupervised':
            reward = new_acc
        else:
            reward = new_acc - self.acc
        self.acc = new_acc
        # mark queried
        self.queried.append(new_seq_idx)
        # next state
        next_observation = self.get_state()
        # mark termination
        terminal = True if next_observation[-1] == 0 else False
        return (reward, next_observation, terminal)

    def reweight_acc(self):
        if self.reweight.startswith('test2T'):
            source_idx = self.test
            target_idx = self.test
        elif self.reweight.startswith('valid2V'):
            source_idx = self.valid
            target_idx = self.valid
        else:
            source_idx = self.valid
            target_idx = self.test

        source_ratio = self.gen_dataDistr([self.data[i] for i in source_idx])
        target_ratio = self.gen_dataDistr([self.data[i] for i in target_idx])
        accs_reweighted = []
        norms = []
        for i in source_idx:
            seq = self.data[i]
            _, _, _, acc = self.tagger.evaluate_acc([seq])
            if self.reweight.endswith('x'):
                x = ",".join(str(char) for char in seq[0])
            else:
                x = ",".join(str(char) for char in seq[1])

            if self.reweight == 'kmers':
                accs_reweighted.append(self.sim_valid2test[i] * acc)
                norms.append(self.sim_valid2test[i])
            elif x in target_ratio:
                accs_reweighted.append(target_ratio[x] / source_ratio[x] * acc)
                norms.append(target_ratio[x] / source_ratio[x])
        if len(norms) == 0:
            print ("Error: the reweight strategy is not suitable, since no overlapping found between source and target set.")
        acc = sum(accs_reweighted) / sum(norms)
        return acc
        

    def get_similarity2test(self):
        # Vectorized and clustered test set.
        Xs = [self.data[i][0] for i in self.test]
        Xs.extend([self.data[i][0] for i in self.valid])
        vec, _ = self.vectorize_string(Xs)
        test_vec = vec[:len(self.test)].tolist()
        valid_vec = vec[len(self.test):].tolist()
        # Pre-calculate similarity: between validation and test
        sim2test_matrix = np.zeros((len(valid_vec), len(test_vec)))
        try:
            with tqdm(range(len(valid_vec))) as iterator:
                for i in iterator:
                    for j in range(len(test_vec)):
                        # cosine distance is 1-cosine(a,b)
                        sim2test_matrix[i, j] = 1 - scipy.spatial.distance.cosine(valid_vec[i], test_vec[j])
        except KeyboardInterrupt:
            iterator.close()
            raise
        iterator.close()
        sim_weight = scipy.special.softmax(np.sum(sim2test_matrix, axis=1) / sim2test_matrix.shape[1])
        return sim_weight

    # Vectorize a set of string by n-grams.
    def vectorize_string(self, Xs_list):
        vc = CV(analyzer='char_wb', ngram_range=(3, 4), min_df=1, token_pattern='[a-z]{2,}')
        name = []
        for i in Xs_list:
            s = re.findall('(?i)[a-z]{2,}', "".join(str(x) for x in i))
            name.append(' '.join(s))
        vc.fit(name)
        vec = vc.transform(name).toarray()
        dictionary = vc.get_feature_names()
        return vec, dictionary

    # generate data distribution
    def gen_dataDistr(self, sequences):
        data_q = {}
        for seq in sequences:
            if self.reweight.endswith('x'):
                x = ",".join(str(char) for char in seq[0])
            else:
                x = ",".join(str(char) for char in seq[1])
            if x not in data_q:
                data_q[x] = 1
            else:
                data_q[x] += 1
        for k, v in data_q.items():
            data_q[k] = v / len(sequences)
        return data_q

