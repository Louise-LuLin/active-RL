import math
import random
import numpy as np
from tqdm import tqdm
import scipy
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer as CV
import re

class LabelEnv:
    def __init__(self, dataloader, tagger, seed=4, pretrain_size=15, valid_size=200, test_size=200, reweight='valid2T', budget=75):
        self.dataloader = dataloader
        self.dataloader.shuffle(seed)
        
        print ("=== Env: data ===")
        self.pretrain_list = self.dataloader.sequence[:pretrain_size]
        self.test_list = self.dataloader.sequence[-test_size:]
        self.valid_list = self.dataloader.sequence[-test_size - valid_size : -test_size]
        self.train_list  = self.dataloader.sequence[pretrain_size : -test_size - valid_size]
        print ("pretrain  : {}".format(len(self.pretrain_list)))
        print ("train     : {}".format(len(self.train_list)))
        print ("validation: {}".format(len(self.valid_list)))
        print ("test      : {}".format(len(self.test_list)))
                        
        print ("=== Env: setup ===")
        self.reweight = reweight
        print ("reward reweight: {}".format(self.reweight))
        self.sim_valid2test = np.zeros((len(self.valid_list)))
        if self.reweight == 'kmers':
            self.sim_valid2test = self.get_similarity2test()
        print ("similarity for valid to test done -- size={}".format(self.sim_valid2test.shape))
        self.budget = budget
        print ("budget: {}".format(self.budget))
        
        self.tagger = tagger
        self.tagger.add_instances([self.train_list[0]])
        self.tagger.train()
        _, _, _, acc_test = self.tagger.evaluate_acc(self.test_list)
        _, _, _, acc_valid = self.tagger.evaluate_acc(self.valid_list)
        self.acc = self.reweight_acc()
        print ("start acc: {} on test, {} on valid, {} after reweight".format(acc_test, acc_valid, self.acc))
        
        self.terminal = False
        self.queried = {0}
        self.cost = 1
        self.acc_trace = {self.cost:(acc_test, acc_valid)}
    
    def reboot(self):
        self.tagger.reset()
    
    def get_state(self):
        # compute embedding, confidence, trellis structure for each candidate
        seq_embeddings = []
        seq_confidences = []
        seq_trellis = []
        for seq in self.train_list:
            seq_embeddings.append(self.dataloader.get_embedding(seq))
            seq_confidences.append(self.tagger.compute_confidence(seq))
            seq_trellis.append(self.tagger.get_marginal_matrix(seq))
        # construct parameter matrix for current tagger
        tagger_para = self.tagger.get_parameter_matrix()
        observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, self.queried]
        return observation
        
    def reweight_acc(self):
        if self.reweight.startswith('test2T'):
            source_seqs = self.test_list
            target_seqs = self.test_list
        elif self.reweight.startswith('valid2V'):
            source_seqs = self.valid_list
            target_seqs = self.valid_list
        else:
            source_seqs = self.valid_list
            target_seqs = self.test_list

        source_ratio = self.gen_dataDistr(source_seqs)
        target_ratio = self.gen_dataDistr(target_seqs)
        accs_reweighted = []
        norms = []
        for i, seq in enumerate(source_seqs):
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

        acc = sum(accs_reweighted) / sum(norms)
        return acc

    def feedback(self, new_seq_idx):
        # reward
        self.tagger.add_instances([self.train_list[new_seq_idx]])
        self.tagger.train()
        new_acc = self.reweight_acc()
        reward = new_acc - self.acc
        self.acc = new_acc
        
        # next state
        # mark queried
        if new_seq_idx not in self.queried:
            self.queried.add(new_seq_idx)
            self.cost += 1
        _,_,_,acc_test = self.tagger.evaluate_acc(self.test_list)
        _,_,_,acc_valid = self.tagger.evaluate_acc(self.valid_list)
        self.acc_trace[self.cost] = (acc_test, acc_valid)
        # compute embedding, confidence, trellis structure for each candidate
        seq_embeddings = []
        seq_confidences = []
        seq_trellis = []
        for seq in self.train_list:
            seq_embeddings.append(self.dataloader.get_embedding(seq))
            seq_confidences.append(self.tagger.compute_confidence(seq))
            seq_trellis.append(self.tagger.get_marginal_matrix(seq))
        # construct parameter matrix for current tagger
        tagger_para = self.tagger.get_parameter_matrix()
        
        if self.cost == self.budget:
            self.terminal = True
        
        next_observation = [seq_embeddings, seq_confidences, seq_trellis, tagger_para, self.queried]
        return (reward, next_observation, self.terminal)
        
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

    def get_similarity2test(self):
        # Vectorized and clustered test set.
        Xs = [seq[0] for seq in self.test_list]
        Xs.extend([seq[0] for seq in self.valid_list])
        vec, _ = self.vectorize_string(Xs)
        test_vec = vec[:len(self.test_list)].tolist()
        valid_vec = vec[len(self.test_list):].tolist()
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
        sim_weight = softmax(np.sum(sim2test_matrix, axis=1) / sim2test_matrix.shape[1])
        return sim_weight

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

