# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 20:43:18 2019

@author: blose
"""

#%%
import numpy as np
from tqdm import tqdm
import time



def read_data(filename):
    lines = open(filename).read().split('\n')
    
    data = []
    for line in lines[:-1]:
        data.append(line.split(', '))
            
    data = np.array(data, dtype='object')
    
    return data


def inner_prod(x, w):
    return w[0] + sum([i*j for i,j in zip(x,w[1:])])

def predict(x,w):
    return 1 if inner_prod(x, w) > 0 else -1


def train_perceptron(X_train, y_train, X_dev, y_dev, epochs):
    m, n = X_train.shape
    w = np.array([0 for i in range(n+1)])
    
    for epoch in range(epochs):
        updates = 0
        for i in range(m):
            pred = inner_prod(X_train[i], w)
            if y_train[i]*pred <= 0:
                updates += 1
                w[0] = w[0] + y_train[i]
                w[1:] = w[1:] + y_train[i]*X_train[i]
                
        y_pred = np.zeros(X_dev.shape[0])
        for i in range(X_dev.shape[0]):
            y_pred[i] = predict(X_dev[i], w)
        
        print('epoch', epoch, 'updates', updates, \
              '('+str(np.round(updates/m*100,2))+'%)', 'dev_err', 
              np.round(np.mean(y_pred != y_dev)*100,2), '(+:'+str(np.round(100*(y_pred > 0).mean(),2))+'%)')
                
    return w


def train_perceptron_average(X_train, y_train, X_dev, y_dev, epochs):
    m, n = X_train.shape
    w = np.array([0 for i in range(n+1)])   
    ws = np.array([0 for i in range(n+1)])
    for epoch in range(epochs):
        updates = 0
        for i in range(m):
            pred = inner_prod(X_train[i], w)
            if y_train[i]*pred <= 0:
                updates += 1
                w[0] = w[0] + y_train[i]
                w[1:] = w[1:] + y_train[i]*X_train[i]
            ws = ws + w
        y_pred = np.zeros(X_dev.shape[0])
        for i in range(X_dev.shape[0]):
            y_pred[i] = predict(X_dev[i], ws)
        
        print('epoch', epoch, 'updates', updates, \
              '('+str(np.round(updates/m*100,2))+'%)', 'dev_err', 
              np.round(np.mean(y_pred != y_dev)*100,2), '(+:'+str(np.round(100*(y_pred > 0).mean(),2))+'%)')
                
    return ws



def knn(X_train, y_train, X_test, n_neighbors = 3, metric='euclidian'):
    y_pred = []
    
    if metric =='euclidian':
        dist = lambda A,b: np.sqrt(((A - b)**2).sum(axis=1))
    elif metric =='manhatan':
        dist = lambda A,b: np.abs(A - b).sum(axis=1)
    
    
    
    for row in tqdm(range(X_test.shape[0])):
        dists = dist(X_train, X_test[row,:])
        indx = np.argsort(dists)
        most = y_train[indx[:n_neighbors]]
        
        target0 = (most == 0).sum()
        target1 = (most == 1).sum()
        
        if target0 >= target1:
            y_pred.append(0)
        else:
            y_pred.append(1)
        
    return np.array(y_pred)

#%%
train = read_data('hw1-data/income.train.txt.5k')
dev = read_data('hw1-data/income.dev.txt')

mapping = {}
encoded = []
k = 0
for col in range(train.shape[1]):
    items = np.unique(train[:,col])
    thiscol = np.zeros((train.shape[0], items.shape[0]))
    for i, item in enumerate(items):
        mapping[k] = (item, col)
        k += 1
        thiscol[train[:,col] == item, i] = 1
    encoded.append(thiscol) 
encoded = np.concatenate(encoded, axis=1)

X_train = encoded[:, :-2]
y_train = (-1)**encoded[:, -2]


dev_encoded = np.zeros((dev.shape[0], encoded.shape[1]))
for key, val in mapping.items():
    for i in range(dev.shape[1]):
        dev_encoded[dev[:,i] == val[0], key] = 1  
            

X_dev = dev_encoded[:, :-2]
y_dev = (-1)**dev_encoded[:, -2] 


w = train_perceptron(X_train, y_train, X_dev, y_dev, 5)
ws = train_perceptron_average(X_train, y_train, X_dev, y_dev, 5)


indx = np.argsort(ws[1:])
for i in indx[:5]:
    print(ws[i+1], mapping[i])
    
    
indx = np.argsort(ws[1:])
for i in indx[-5:]:
    print(ws[i+1], mapping[i])
    
    
    
print('Bias:', ws[0])


#3.2
start = time.time()
y_pred = y_pred = knn(X_train, y_train, X_dev, k)
print('KNN Runtime:', time.time()-start)


start = time.time()
ws= train_perceptron_average(X_train, y_train, X_dev, y_dev, 5)
print('Perceptron Runtime:', time.time()-start)

# 4.1
sorted_index = np.argsort(-y_train)
w = train_perceptron(X_train[sorted_index], y_train[sorted_index], X_dev, y_dev, 5)
ws = train_perceptron_average(X_train[sorted_index], y_train[sorted_index], X_dev, y_dev, 5)


# 4.2 (a)
X_train2 = np.concatenate((X_train, train[:,[0,7]].astype(int)), axis=1)
X_dev2 = np.concatenate((X_dev, dev[:,[0,7]].astype(int)), axis=1)
ws = train_perceptron_average(X_train2, y_train, X_dev2, y_dev, 5)


# 4.2 (b)
num = train[:,[0,7]].astype(int)
num = num - num.mean(axis=0)
X_train3 = np.concatenate((X_train, num), axis=1)

num = dev[:,[0,7]].astype(int)
num = num - num.mean(axis=0)
X_dev3 = np.concatenate((X_dev, num), axis=1)
ws = train_perceptron_average(X_train3, y_train, X_dev3, y_dev, 5)


# 4.2 (c)
num = train[:,[0,7]].astype(int)
num = (num - num.mean(axis=0))/num.std(axis=0)
X_train4 = np.concatenate((X_train, num), axis=1)

num = dev[:,[0,7]].astype(int)
num = (num - num.mean(axis=0))/num.std(axis=0)
X_dev4 = np.concatenate((X_dev, num), axis=1)
ws = train_perceptron_average(X_train4, y_train, X_dev4, y_dev, 5)


# 4.2 (d)
combs_train = np.zeros((train.shape[0], 2*5))
combs_dev = np.zeros((dev.shape[0], 2*5))
k = 0
for sex in np.unique(train[:,6]):
    for race in np.unique(train[:,5]):
        combs_train[(train[:,6] == sex) & (train[:,5] == race), k] = 1
        combs_dev[(dev[:,6] == sex) & (dev[:,5] == race), k] = 1
        k += 1
              
X_train5 = np.concatenate((X_train, combs_train), axis=1)
X_dev5 = np.concatenate((X_dev, combs_dev), axis=1)
ws = train_perceptron_average(X_train5, y_train, X_dev5, y_dev, 5)

            
test = read_data('hw1-data/income.test.blind')

X_test = np.zeros((test.shape[0], X_train.shape[1]))
for key, val in list(mapping.items())[:-2]:
    X_test[test[:,val[1]] == val[0], key] = 1
    
ws = train_perceptron_average(X_train4, y_train, X_dev4, y_dev, 1)
y_test = []
for i in range(X_test.shape[0]):
    y_test.append(predict(X_test[i], ws))

y_test = np.array(y_test)
print('Test set positive rate:', (y_test > 0).mean())

target = np.array(['<=50K' if x == -1 else '>50K' for x in y_test])

test = np.concatenate((test, target.reshape(-1,1)), axis=1)

with open('income.test.predicted', 'w') as myfile:
    for i in range(test.shape[0]):
        myfile.write(', '.join(test[i,:])+'\n')