import math
import random
import numpy as np
from tqdm import tqdm
import scipy
from scipy.special import softmax
from sklearn.feature_extraction.text import CountVectorizer as CV
import re

class LabelEnv:
    def __init__(self, dataloader, tagger, seed=4, valid_size=200, test_size=200, train_size=600, reweight='valid2T', budget=75):
        self.dataloader = dataloader
        self.dataloader.shuffle(seed)
        
        print ("=== Env: data ===")
        self.test_list = self.dataloader.sequence[-test_size:]
        self.valid_list = self.dataloader.sequence[-test_size - valid_size : -test_size]
        self.train_list  = self.dataloader.sequence[0 : train_size]
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
        self.set_tagger([self.train_list[0]])
        
        # store instance embeddings, initialize trellis and confidence
        self.seq_embeddings = []
        for seq in self.train_list:
            self.seq_embeddings.append(self.dataloader.get_embedding(seq))
        self.seq_confidences = []
        self.seq_trellis = []
        self.tagger_para = self.tagger.get_parameter_matrix()
        
        self.terminal = False
        self.scope = range(len(self.train_list)) # or cache for reboot inner loop
        self.cache = [0]
        self.queried = []
        self.cost = 0
        self.acc_trace = {}
    
    def set_tagger(self, seqs):
        self.tagger.reset()
        self.tagger.add_instances(seqs)
        self.tagger.train()
    
    def eval_tagger(self):
        # evaluate tagger
        self.cache = self.queried[:]
        self.set_tagger([self.train_list[i] for i in self.cache])
        _, _, _, acc_test = self.tagger.evaluate_acc(self.test_list)
        _, _, _, acc_valid = self.tagger.evaluate_acc(self.valid_list)
        self.cost = len(self.cache)
        self.acc_trace[self.cost] = (acc_test, acc_valid)
        return (acc_test, acc_valid)

    # reboot to generate more experience
    def reboot(self):
        self.scope = self.cache[:]
        random.shuffle(self.scope)
        init = int(0.5 * len(self.scope))
        self.queried = [self.scope[i] for i in range(init)]
        self.set_tagger([self.train_list[i] for i in self.queried])
        self.acc = self.reweight_acc()
        self.terminal = False
        
    def resume(self):
        self.scope = range(len(self.train_list))
        self.queried = self.cache[:]
        self.set_tagger([self.train_list[self.scope[i]] for i in self.queried])
        self.acc = self.reweight_acc()
        self.terminal = False
        
    def get_state(self):
        # compute confidence, trellis structure for each candidate
        self.seq_confidences = []
        self.seq_trellis = []
        for seq in self.train_list:
            self.seq_confidences.append(np.array([self.tagger.compute_confidence(seq)]))
            self.seq_trellis.append(self.tagger.get_marginal_matrix(seq))
        # construct parameter matrix for current tagger
        self.tagger_para = self.tagger.get_parameter_matrix()
        observation = [self.seq_embeddings, self.seq_confidences, self.seq_trellis, self.tagger_para, self.queried, self.scope]
        return observation
 
    def feedback(self, new_seq_idx):
        # reward
        self.tagger.add_instances([self.train_list[new_seq_idx]])
        self.tagger.train()
        new_acc = self.reweight_acc()
        reward = new_acc - self.acc
        self.acc = new_acc
        
        # next state
        # compute embedding, confidence, trellis structure for each candidate
        self.seq_confidences = []
        self.seq_trellis = []
        for seq in self.train_list:
            self.seq_confidences.append(np.array([self.tagger.compute_confidence(seq)]))
            self.seq_trellis.append(self.tagger.get_marginal_matrix(seq))
        # construct parameter matrix for current tagger
        self.tagger_para = self.tagger.get_parameter_matrix()
        
        # mark queried
        self.queried.append(new_seq_idx)        
        
        if len(self.queried) >= len(self.scope)-1:
            self.terminal = True
        
        next_observation = [self.seq_embeddings, self.seq_confidences, self.seq_trellis, self.tagger_para, self.queried, self.scope]
        return (reward, next_observation, self.terminal)

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

