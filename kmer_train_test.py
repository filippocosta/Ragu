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
import matplotlib.pyplot as plt
import itertools
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
intr_file = '../ubuntu/data/hg19_intr_clean.fa'
depl_file = '../ubuntu/data/hg19_depl_clean.fa'
e = 0
intr_seqs = []
depl_seqs = []
for intr, depl in zip(SeqIO.parse(intr_file, 'fasta'), SeqIO.parse(depl_file, 'fasta')):
    
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

kmer = [10]
d = 2
results = []
emb = []
for k in kmer:
    print('Kmer: ',k)
    intr_texts = [' '.join(getKmers(i, k)) for i in intr_seqs]
    depl_texts = [' '.join(getKmers(i, k)) for i in depl_seqs]
    merge_texts = intr_texts + depl_texts
    labels = list(np.ones(len(intr_texts))) + list(np.zeros(len(depl_texts)))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(merge_texts)

    encoded_docs = tokenizer.texts_to_sequences(merge_texts)
    max_length = max([len(s.split()) for s in merge_texts])
    X = pad_sequences(encoded_docs, maxlen = max_length, padding = 'post')

    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size = 0.20, random_state = 42)

    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    max_length = max([len(s.split()) for s in merge_texts])

    vocab_size = len(tokenizer.word_index) + 1

    from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, LSTM, SimpleRNN, GRU, Bidirectional

    model = Sequential()
    model.add(Embedding(vocab_size, d))

    model.add(Bidirectional(LSTM(int(d/2))))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))

    epochs = 5
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])
    # checkpoint = ModelCheckpoint("weights.best.hdf5", monitor = 'val_acc', verbose = 1, 
    #                             save_best_only = True, mode = 'max')
            
    print(model.summary())

    history = model.fit(X_train, y_train, 
                        epochs = epochs, verbose = 2, validation_split = 0.2, batch_size = 32, shuffle = True, 
    #                    callbacks = [checkpoint]
                       )

    predicted_labels = model.predict(X_test)
    cm = confusion_matrix(y_test, [np.round(i[0]) for i in predicted_labels])
    cm = cm.astype('float') / cm.sum(axis = 1)[:, np.newaxis]

    scores = model.evaluate(X_test, y_test, verbose = 0)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    results.append(scores[1]*100)
