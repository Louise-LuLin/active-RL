import random
import numpy as np
import scipy
from gensim.models import KeyedVectors

# Sequence Loader
class BuildDataLoader:
    
    def __init__(self, source, folder, num_flag, embed_flag):
        self.folder = folder
        self.num_flag = num_flag
        self.embed_flag = embed_flag
        
        self.sequence = []
        self.w_embeddings = []
        self.word_dict = {}
        self.label_dict = {}
        
        # load seq data
        with open(folder + ".all", 'r') as file:
            x=[]
            y=[]
            for line in file:
                line = line.replace("\n",'')
                if line == "":
                    self.sequence.append((x,y))
                    x = []
                    y = []
                else:
                    char, label = line.split('\t')
                    if self.num_flag and char.replace('.','').isdigit():
                        char = 'NUM'
                    x.append(char)
                    y.append(label)
                    if char not in self.word_dict:
                        self.word_dict[char] = len(self.word_dict)
                    if label not in self.label_dict:
                        self.label_dict[label] = len(self.label_dict)
                        
        # calculate the maximum length of all sequences            
        lens = [len(seq[0]) for seq in self.sequence]
        self.max_len = max(lens)
         
        # load embeddings or create bag-of-word embeddings
        self.w_embeddings = {}
        if self.embed_flag:
            num_str = "NUM" if self.num_flag else ""
            wv = KeyedVectors.load(self.folder + "word2vec" + num_str + ".kv", mmap='r')
            for k in wv.vocab:
                    self.w_embeddings[k] = wv[k]
        else:
            for k, v in self.word_dict.items():
                embed = np.zeros(shape=(len(self.word_dict)))
                embed[v] = 1
                self.w_embeddings[k] = embed
                
        print ("=== data loading ===")
        print ("word dict size: {}".format(len(self.word_dict)))
        print ("label dict size: {}".format(len(self.label_dict)))
        print ("Replace digit with NUM: {}".format(str(self.num_flag)))
        print ("use embedding: {}".format(str(self.embed_flag)))
        print ("embedding size: {}".format(self.get_embed_size()))
    
    def shuffle(self, seed = 4):
        random.Random(4).shuffle(self.sequence)
    
    def get_embedding(self, sequence):
        embeddings = np.zeros(shape=(self.max_len, self.get_embed_size()))
        for i, char in enumerate(sequence[0]):
            embeddings[i] = self.w_embeddings[char]
        return embeddings
    
    def get_embed_size(self):
        for k, v in self.w_embeddings.items():
            return len(v)
        
    def get_max_len(self):
        return self.max_len
    
    def get_word_dict(self):
        return self.word_dict
    
    def get_label_dict(self):
        return self.label_dict


