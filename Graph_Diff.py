#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import subprocess
import numpy as np
from Bio import SeqIO

import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD, Adam, Adadelta, RMSprop
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, LSTM, SimpleRNN, GRU

from sklearn.metrics import confusion_matrix
import itertools


# In[ ]:


intr_file = '../ubuntu/data/hg19_intr_clean.fa'
depl_file = '../ubuntu/data/hg19_depl_clean.fa'

e = 0
intr_seqs = []
depl_seqs = []
for intr, depl in zip(SeqIO.parse(intr_file, 'fasta'), SeqIO.parse(depl_file, 'fasta')):
    
    #cutoff = 200
    #my_intr_seq = str(intr.seq)[0:cutoff]
    #my_depl_seq = str(depl.seq)[0:cutoff]
    #intr_seqs.append(my_intr_seq)
    #depl_seqs.append(my_depl_seq)
    
    step = 200; jump = 1; a = 0; b = step; n_jumps = 5
    for j in range(n_jumps):
        s_intr = str(intr.seq)[a:b]
        s_depl = str(depl.seq)[a:b]
        intr_seqs.append(s_intr)
        depl_seqs.append(s_depl)
        a = a + jump
        b = a + step
    
    e = e + 1
    if e%20000 == 0:
        print('Finished ' + str(e) + ' entries')
        
def getKmers(sequence, size):
    return [sequence[x:x+size].upper() for x in range(len(sequence) - size + 1)]


# In[ ]:


kmer = [9,10]
for k in kmer:
    print('KMER: ', k)
    intr_texts = [' '.join(getKmers(i, k)) for i in intr_seqs]
    depl_texts = [' '.join(getKmers(i, k)) for i in depl_seqs]

    merge_texts = intr_texts + depl_texts

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(merge_texts)
    #X = tokenizer.texts_to_matrix(merge_texts, mode = 'freq')

    encoded_docs = tokenizer.texts_to_sequences(merge_texts)
    max_length = max([len(s.split()) for s in merge_texts])
    X = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')

    l_intr = len(intr_texts)
    l_depl = len(depl_texts)
    vocab_size = len(tokenizer.word_index) + 1

    X_intr = X[0:l_intr]
    X_depl = X[l_intr:]

    adj_intr = np.zeros(shape = (vocab_size,vocab_size))
    for s in X_intr:
        for i in range(len(s)-1):
            j = int(s[i])
            k = int(s[i+1])
            adj_intr[j,k] = 1
            adj_intr[k,j] = 1
    adj_depl = np.zeros(shape = (vocab_size,vocab_size))
    for s in X_depl:
        for i in range(len(s)-1):
            j = int(s[i])
            k = int(s[i+1])
            adj_depl[j,k] = 1
            adj_depl[k,j] = 1

    diff_matrix = np.abs(np.subtract(adj_intr, adj_depl))
    diff_node_degree = np.sum(diff_matrix)
    print('DIFFERENCE GRADE: ', diff_node_degree)

