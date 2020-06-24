#!/usr/bin/env python

from __future__ import division

import sys
import time
from svector import svector
import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction import stop_words
from string import punctuation

def read_from(textfile):
    for line in open(textfile):
        label, words = line.strip().split("\t")
        yield (1 if label=="+" else -1, words.split())

def make_vector(words):
    v = svector()
    for word in words:
        v[word] += 1
    return v
    
def test(devfile, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(read_from(devfile), 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now


def test_cache(dev_data, model):
    tot, err = 0, 0
    for i, (label, words) in enumerate(dev_data, 1): # note 1...|D|
        err += label * (model.dot(make_vector(words))) <= 0
    return err/i  # i is |D| now


def find_quant(data):
    all_words = {}
    for x, y in data:
        for word in y:
            if word in all_words:
                all_words[word] += 1
            else:
                all_words[word] = 1
    return all_words
            
def train(trainfile, devfile, epochs=5, counts=0):
    
   
    
    train_data = list(read_from(trainfile))
    
    all_words = find_quant(train_data)
 
#        list(punctuation) +  ["'s", "n't"] + list(stop_words.ENGLISH_STOP_WORDS) +
    removed_words = set([word for word, quan in all_words.items() if quan <= counts])
    
    
    pruned_train = []
    for label, words in train_data:
        pruned_train.append((label, [x for x in words if x not in removed_words]))
        
    t = time.time()
    
    best_err = 1.
    model = svector()
    modela = svector()
    c = 0
    
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(pruned_train, 1): # label is +1 or -1
            words = words.copy()
            words.append('<bias>')
            sent = make_vector(words)
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                modela += c*label*sent
            c += 1
        dev_err = test(devfile, c*model - modela)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(model), time.time() - t))

 
if __name__ == "__main__":
    print(sys.argv)
    print('Train with pruning with single count')
    train(sys.argv[1], sys.argv[2], 10, 1)
    
    
    print('Train with pruning with 2 counts')
    train(sys.argv[1], sys.argv[2], 10, 2)

