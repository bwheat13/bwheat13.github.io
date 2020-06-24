# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:25:52 2019

@author: blose
"""
#%%
import numpy as np
from tqdm import tqdm

def read_train(filename):
    lines = open(filename).readlines()
    data = map(lambda s: s.strip().split(", "), lines)
    
    mapping = {}
    new_data = []
    for row in data: 
        new_row = []
        for j, x in enumerate(row): 
            feature = (j, x)
            if feature not in mapping:
                mapping[feature] = len(mapping) # insert a new feature into index
            new_row.append(mapping[feature])
        new_data.append(new_row)
    
    new_data
    
    bindata = np.zeros((len(new_data), len(mapping)))
    bindata
    for i, row in enumerate(new_data):
        for x in row: 
            bindata[i][x] = 1
    
    X_train = np.delete(bindata, 7, 1)
    y_train = bindata[:,7]
    
    return X_train, y_train, mapping


def read_data(filename):
    lines = open(filename).read().split('\n')
    
    data = []
    for line in lines[:-1]:
        data.append(line.split(', '))
            
    data = np.array(data, dtype='object')
    
    return data


#%%
train = read_data('hw1-data/income.train.txt.5k')
dev = read_data('hw1-data/income.dev.txt')

ntrain = train.shape[0]


alldata = np.concatenate((train, dev))

encoded = np.zeros((alldata.shape[0],0))
mapping = {}
for col in range(alldata.shape[1]):
    if alldata[0,col].isdecimal():
        thiscol = alldata[:,col].astype(float)
        encoded = np.concatenate((encoded, thiscol.reshape(-1,1)), axis=1)
    else:
        items = np.unique(alldata[:,col])[:-1]
        thiscol = np.zeros((alldata.shape[0], items.shape[0]))
        for i, item in enumerate(items):
            mapping[item] = (i + encoded.shape[1], col)
            thiscol[alldata[:,col] == item, i] = 1
        encoded = np.concatenate((encoded, thiscol), axis=1)    
            
X_train = encoded[:ntrain, :-1]
y_train = 1 - encoded[:ntrain, -1]

X_dev = encoded[ntrain:, :-1]
y_dev = 1 - encoded[ntrain:, -1] 


dists = []
for row in range(X_dev.shape[0]):
    dists.append(np.abs(np.delete(X_train, [0,46], 1) - np.delete(X_dev[row,:], [0,46])).sum(axis=1).max())
np.max(dists)


#train = pd.read_csv('hw1-data/income.train.txt.5k', header=None)
#dev = pd.read_csv('hw1-data/income.dev.txt', header=None)
#alldata = pd.concat((train, dev))
#encoded = pd.get_dummies(alldata, drop_first=True).values
#X_train = encoded[:ntrain, :-1]
#y_train = 1 - encoded[:ntrain, -1]
#X_dev = encoded[ntrain:, :-1]
#y_dev = 1 - encoded[ntrain:, -1] 
#
#knn = KNeighborsClassifier(1)
#knn.fit(X_train, y_train)
#1- knn.score(X_dev, y_dev)

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
        
# Euclidian
for k in [1,3,5,7,9,99,999,9999]:
    y_pred = knn(X_train, y_train, X_dev, k)
    print('k =', k, 'Test: , error:', np.mean(y_pred != y_dev), ', positive rate:', y_pred.mean())

    y_pred = knn(X_train, y_train, X_train, k)
    print('k =', k, 'Train: , error:', np.mean(y_pred != y_train), ', positive rate:', y_pred.mean())
    
    
    
# Manhattan
for k in [1,3,5,7,9,99,999,9999]:
    y_pred = knn(X_train, y_train, X_dev, k, metric='manhatan')
    print('k =', k, 'Test: , error:', np.mean(y_pred != y_dev), ', positive rate:', y_pred.mean())

    y_pred = knn(X_train, y_train, X_train, k, metric='manhatan')
    print('k =', k, 'Train: , error:', np.mean(y_pred != y_train), ', positive rate:', y_pred.mean())
    
    
    


#%%
train = read_data('hw1-data/income.train.txt.5k')
dev = read_data('hw1-data/income.dev.txt')

ntrain = train.shape[0]

alldata = np.concatenate((train, dev))

encoded = np.zeros((alldata.shape[0],0))
for col in range(alldata.shape[1]):
    items = np.unique(alldata[:,col])[:-1]
    thiscol = np.zeros((alldata.shape[0], items.shape[0]))
    for i, item in enumerate(items):
        thiscol[alldata[:,col] == item, i] = 1
    encoded = np.concatenate((encoded, thiscol), axis=1)    
            
X_train = encoded[:ntrain, :-1]
y_train = 1 - encoded[:ntrain, -1]

X_dev = encoded[ntrain:, :-1]
y_dev = 1 - encoded[ntrain:, -1] 

# Euclidian
for k in [1,3,5,7,9,99,999,9999]:
    y_pred = knn(X_train, y_train, X_dev, k)
    print('k =', k, 'Test: , error:', np.mean(y_pred != y_dev), ', positive rate:', y_pred.mean())

    y_pred = knn(X_train, y_train, X_train, k)
    print('k =', k, 'Train: , error:', np.mean(y_pred != y_train), ', positive rate:', y_pred.mean())
    
    
# Manhattan
for k in [1,3,5,7,9,99,999,9999]:
    y_pred = knn(X_train, y_train, X_dev, k, metric='manhatan')
    print('k =', k, 'Test: , error:', np.mean(y_pred != y_dev), ', positive rate:', y_pred.mean())

    y_pred = knn(X_train, y_train, X_train, k, metric='manhatan')
    print('k =', k, 'Train: , error:', np.mean(y_pred != y_train), ', positive rate:', y_pred.mean())
    
    
    
    
    
    
train = read_data('hw1-data/income.train.txt.5k')
dev = read_data('hw1-data/income.dev.txt')

test = read_data('hw1-data/income.test.blind')

ntrain = train.shape[0]


alldata = np.concatenate((train, dev))
encoded = np.zeros((alldata.shape[0],0))
mapping = {}
for col in range(alldata.shape[1]):
    if alldata[0,col].isdecimal():
        thiscol = alldata[:,col].astype(float)
        encoded = np.concatenate((encoded, thiscol.reshape(-1,1)), axis=1)
    else:
        items = np.unique(alldata[:,col])[:-1]
        thiscol = np.zeros((alldata.shape[0], items.shape[0]))
        for i, item in enumerate(items):
            mapping[item] = (i + encoded.shape[1], col)
            thiscol[alldata[:,col] == item, i] = 1
        encoded = np.concatenate((encoded, thiscol), axis=1)    
            
X_train = encoded[:ntrain, :-1]
y_train = 1 - encoded[:ntrain, -1]

X_dev = encoded[ntrain:, :-1]
y_dev = 1 - encoded[ntrain:, -1] 


X_test = np.zeros((test.shape[0], X_train.shape[1]))

for item, val in list(mapping.items())[:-1]:
    X_test[test[:,val[1]] == item, val[0]] = 1
    
X_test[:,[0,46]] = test[:,[0,7]].astype(float)

y_test = knn(X_train, y_train, X_test, k, metric='manhatan')

print('Test set positive rate:', y_test.mean())

target = np.array(['<=50K' if x == 0 else '>50K' for x in y_test])

test = np.concatenate((test, target.reshape(-1,1)), axis=1)

with open('income.test.predicted', 'w') as myfile:
    for i in range(test.shape[0]):
        myfile.write(', '.join(test[i,:])+'\n')