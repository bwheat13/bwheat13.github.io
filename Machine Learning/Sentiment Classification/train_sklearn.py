# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:10:33 2019

@author: blose
"""

from __future__ import division

import sys
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB
vectorizer = CountVectorizer(max_features=7200)
import numpy as np

  
def train(trainfile, devfile, epochs=5):
    t = time.time()
    
    y_train = []
    sents = []
    for line in open(trainfile):
        label, sent = line.strip().split("\t", 1)
        y_train.append(1 if label=="+" else 0)
        sents.append(sent)
        
        
    
    X_train = vectorizer.fit_transform(sents)
    y_train = np.array(y_train)
    
    
    cl = GaussianNB()
    cl.fit(X_train.toarray(), y_train)
    
    
    y_dev = []
    sents_dev = []
    for line in open(devfile):
        label, sent = line.strip().split("\t", 1)
        y_dev.append(1 if label=="+" else 0)
        sents_dev.append(sent)
    
    
    X_dev = vectorizer.transform(sents_dev)
    y_dev = np.array(y_dev)

    y_pred = cl.predict(X_dev.toarray())
    best_err = np.mean(y_pred != y_dev)
    print("best dev err %.1f%%, time: %.1f secs" % (best_err * 100, time.time() - t))
    
    
if __name__ == "__main__":
    print('Train with sklearn')
    train(sys.argv[1], sys.argv[2], 10)
    