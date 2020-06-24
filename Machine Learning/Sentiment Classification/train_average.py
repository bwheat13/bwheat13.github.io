#!/usr/bin/env python

from __future__ import division

import sys
import time
from svector import svector
import pandas as pd

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
            
def train(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    c = 0
    
    modela = svector()
    
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(read_from(trainfile), 1): # label is +1 or -1
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
    
    w = c*model - modela 
    w_sorted = sorted(w.items(), key=lambda x:x[1])
    print()
    print('top negative:')
    for i in range(20):
        print(w_sorted[i])
        
    print()
    print('top positive:')
    for i in range(1,21):
        print(w_sorted[-i])
        
        
    rows = []
    for i, (label, words) in enumerate(read_from(devfile), 1): 
        rows.append([label,  w.dot(make_vector(words)), ' '.join(words)])
    
    results = pd.DataFrame(rows, columns=['label', 'score', 'sentence'])
    
    print()
    print('5 negative examples with predicted to be positive:')
    print(results.query('label == -1').sort_values('score').tail().to_string())
    
    print()
    print('5 positive examples with predicted to be negative:')
    print(results.query('label == 1').sort_values('score').head().to_string())
    
    
    
def train_cache(trainfile, devfile, epochs=5):
    t = time.time()
    best_err = 1.
    model = svector()
    c = 0
    
    modela = svector()
    
    train_data = list(read_from(trainfile))
    dev_data = list(read_from(devfile))
    
    for it in range(1, epochs+1):
        updates = 0
        for i, (label, words) in enumerate(train_data, 1): # label is +1 or -1
            words = words.copy()
            words.append('<bias>')
            sent = make_vector(words)
            
            if label * (model.dot(sent)) <= 0:
                updates += 1
                model += label * sent
                modela += c*label*sent
            c += 1
        dev_err = test_cache(dev_data, c*model - modela)
        best_err = min(best_err, dev_err)
        print("epoch %d, update %.1f%%, dev %.1f%%" % (it, updates / i * 100, dev_err * 100))
    print("best dev err %.1f%%, |w|=%d, time: %.1f secs" % (best_err * 100, len(c*model - modela), time.time() - t))
        

if __name__ == "__main__":
    print('Train with averaging')
    train(sys.argv[1], sys.argv[2], 10)
    
    print()
    print('Train with averaging and caching')
    train_cache(sys.argv[1], sys.argv[2], 10)
