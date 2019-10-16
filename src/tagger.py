import math
import random
import numpy as np
import scipy
import pickle
import sys
from gensim.models import KeyedVectors
from sklearn_crfsuite import CRF

# CRF
class CrfModel(object):
    
    def __init__(self, dataloader, feature):
        self.label_dict = dataloader.label_dict
        self.word_dict = dataloader.word_dict
        self.max_len = dataloader.max_len
        self.feature = feature
        
        self.crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
        
        self.X_train=[]
        self.Y_train=[]
    
        print ('=== Tagger initialized ===')
        print ('-- label dict size: {}'.format(len(self.label_dict)))
        print ('-- word dict size: {}'.format(len(self.word_dict)))
        
    def reset(self):
        self.X_train=[]
        self.Y_train=[]

    def char2feature(self, sent, i):
        # for current character
        features = {'0:word': sent[i]}
        # for previous character
        if i > 0:
            features.update({'-1:word': sent[i-1]})
        # for next character
        if i < len(sent)-1:
            features.update({'+1:word': sent[i+1]})
        return features
    
    def char2feature2(self, sent, i):
        word = sent[i]
        features = [
            'bias',
            'word.lower=' + word.lower(),
            'word[-3:]=' + word[-3:],
            'word[-2:]=' + word[-2:],
            'word.isupper=%s' % word.isupper(),
            'word.istitle=%s' % word.istitle(),
            'word.isdigit=%s' % word.isdigit(),
            #'postag=' + postag,
            #'postag[:2]=' + postag[:2],
        ]
        if i > 0:
            word1 = sent[i - 1]
            #postag1 = sent[i - 1][1]
            features.extend([
                '-1:word.lower=' + word1.lower(),
                '-1:word.istitle=%s' % word1.istitle(),
                '-1:word.isupper=%s' % word1.isupper(),
                #'-1:postag=' + postag1,
                #'-1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('BOS')
        if i < len(sent) - 1:
            word1 = sent[i + 1]
            #postag1 = sent[i + 1][1]
            features.extend([
                '+1:word.lower=' + word1.lower(),
                '+1:word.istitle=%s' % word1.istitle(),
                '+1:word.isupper=%s' % word1.isupper(),
                #'+1:postag=' + postag1,
                #'+1:postag[:2]=' + postag1[:2],
            ])
        else:
            features.append('EOS')
        return features
    
    def add_instances(self, sequences):
        for seq in sequences:
            x = seq[0]
            y = seq[1]
            self.X_train.append([self.char2feature(x, i) for i in range(len(x))])
            self.Y_train.append(y)
    
    def compute_confidence(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        y_pred = self.crf.tagger_.tag(x)
        conf = self.crf.tagger_.probability(y_pred)
        conf_norm = pow(conf, 1. / len(y_pred))
        return conf_norm
    
    def compute_point_confidence(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        label_list = self.crf.tagger_.labels()
        prob_list = []
        for i in range(len(x)):
            marginal_prob = [self.crf.tagger_.marginal(k, i) for k in label_list]
            prob_list.append(max(marginal_prob))
        return (prob_list, sum(prob_list))
    
    def compute_point_entropy(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        label_list = self.crf.tagger_.labels()
        self.crf.tagger_.set(x)
        entropy_seq = []
        for i in range(len(x)):
            marginal_prob = [self.crf.tagger_.marginal(k, i) for k in label_list]
            entropy_seq.append(scipy.stats.entropy(marginal_prob))
        return (entropy_seq, sum(entropy_seq))
    
    def train(self):
        self.crf.fit(self.X_train, self.Y_train) 
        return len(self.X_train)
    
    def predict(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        return self.crf.tagger_.tag(x)    
    
    def evaluate_acc(self, sequences):
        # Calculate phrase-level accuracy and out-of-phrase accuracy
        X_test = [[self.char2feature(seq[0], i) for i in range(len(seq[0]))] for seq in sequences]
        Y_test = [seq[1] for seq in sequences]
        Y_pred = self.crf.predict(X_test)
        
        # Consider the accuracy in phrase level.
        in_cnt,  in_crt = 0, 0    # Total/correct number of phrases
        out_cnt, out_crt = 0, 0   # Total/correct number of "o"
        all_cnt, all_crt = 0, 0   # Total/correct number of all words

        acc = []
        for y_test, y_pred in zip(Y_test, Y_pred):
            cnt, crt = 0, 0
            correct_flag = False
            for j in range(len(y_test)):
                all_cnt += 1
                cnt += 1
                if y_test[j] == y_pred[j]:
                    all_crt += 1
                    crt += 1

                # If the character is a beginning-of-phrase.
                if y_test[j][0] == 'b':
                    in_cnt += 1
                    if y_test[j] == y_pred[j]:
                        if correct_flag:
                            in_crt += 1
                        correct_flag = True
                    else:
                        if correct_flag:
                            if y_pred[j][2:] != y_pred[j-1][2:]:  # special case
                                in_crt += 1
                        correct_flag = False

                # If the character is an inside-of-phrase.
                elif y_test[j][0] == 'i':
                    if y_test[j] != y_pred[j]:
                        correct_flag = False

                # If the character is an out-of-phrase.
                elif y_test[j][0] == 'o':
                    out_cnt += 1
                    if y_test[j] == y_pred[j]:
                        out_crt += 1
                        if correct_flag:
                            in_crt += 1
                            correct_flag = False
                    else:
                        if correct_flag:
                            if y_pred[j][2:] != y_pred[j-1][2:]:  # special case
                                in_crt += 1
                            correct_flag = False

            acc.append(crt/cnt)
            # For the case where the phrase is at the end of a string.
            if correct_flag:
                in_crt += 1
        in_acc = 0 if in_cnt == 0 else in_crt/in_cnt
        out_acc = 0 if out_cnt == 0 else out_crt/out_cnt
        all_acc = 0 if all_cnt == 0 else all_crt/all_cnt 
            
        return in_acc, out_acc, all_acc, sum(acc)/len(acc)
    
    def get_parameter_matrix(self):
        loc = {'0':0, '-1':1, '+1':2}
        if self.feature == 'all':
            paras = np.zeros(shape=(len(loc) * len(self.word_dict) + len(self.label_dict), len(self.label_dict)))
            for (attr, label), weight in self.crf.state_features_.items():
                attr = attr.split(":")
                dim1 = loc[attr[0]] * self.word_dict[':'.join(attr[2:])]
                dim2 = self.label_dict[label]
                paras[dim1][dim2] = weight
            for (label_from, label_to), weight in self.crf.transition_features_.items():
                dim1 = len(loc) * len(self.word_dict) + self.label_dict[label_from]
                dim2 = self.label_dict[label_to]
                paras[dim1][dim2] = weight
        elif self.feature == 'node':
            paras = np.zeros(shape=(len(loc) * len(self.word_dict), len(self.label_dict)))
            for (attr, label), weight in self.crf.state_features_.items():
                attr = attr.split(":")
                dim1 = loc[attr[0]] * self.word_dict[':'.join(attr[2:])]
                dim2 = self.label_dict[label]
                paras[dim1][dim2] = weight
        else:
            paras = np.zeros(shape=(len(self.label_dict), len(self.label_dict)))
            for (label_from, label_to), weight in self.crf.transition_features_.items():
                dim1 = self.label_dict[label_from]
                dim2 = self.label_dict[label_to]
                paras[dim1][dim2] = weight
        return paras

    def get_marginal_matrix(self, sequence):
        x = [self.char2feature(sequence[0], i) for i in range(len(sequence[0]))]
        label_list = self.crf.tagger_.labels()
        self.crf.tagger_.set(x)
        marginals = np.zeros(shape=(self.max_len, len(self.label_dict)))
        for i in range(len(x)):
            for lbl in label_list:
                marginals[i][self.label_dict[lbl]] = self.crf.tagger_.marginal(lbl, i)
        return marginals
    
    def get_label_size(self):
        return len(self.label_dict)

    def get_para_shape(self):
        self.train()
        return self.get_parameter_matrix().shape
    
    def get_trellis_shape(self):
        marginals = np.zeros(shape=(self.max_len, len(self.label_dict)))
        return marginals.shape