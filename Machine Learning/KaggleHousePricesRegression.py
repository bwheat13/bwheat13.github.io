# -*- coding: utf-8 -*-
"""
Created on Thu May  2 21:45:05 2019

@author: blose
"""

import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV
from sklearn.metrics import mean_squared_log_error, mean_squared_error

#%%
datafolder = 'input'


train = pd.read_csv(os.path.join(datafolder, 'my_train.csv'))
dev = pd.read_csv(os.path.join(datafolder, 'my_dev.csv'))
test = pd.read_csv(os.path.join(datafolder, 'test.csv'))

ntrain= train.shape[0]
nvalid = dev.shape[0]

# total number of unique values (7227)
train.drop(columns=['SalePrice', 'Id']).nunique(dropna=False).sum()

# total number of unique values
train.drop(columns=['SalePrice', 'Id']).nunique().to_csv('value_per_column.csv')
train.drop(columns=['SalePrice', 'Id']).nunique()

#%% Part 2 

X_all = pd.concat((train, dev, test), sort=False).drop(columns=['SalePrice', 'Id']).astype(str)
X_dummy = pd.get_dummies(X_all)

X_train = X_dummy.iloc[:ntrain,:]
X_valid = X_dummy.iloc[ntrain:ntrain+nvalid,:]
X_test = X_dummy.iloc[ntrain+nvalid:,:]

y_train = np.log(train['SalePrice'])
y_valid = np.log(dev['SalePrice'])

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_valid)
mean_squared_error(y_valid, y_pred)**0.5

coef = pd.Series(lr.coef_.ravel(), index=X_train.columns).sort_values()
coef.head(10)
coef.tail(10)

lr.intercept_
np.exp(lr.intercept_)


y_test = lr.predict(X_test)

sub = pd.read_csv(os.path.join(datafolder, 'sample_submission.csv'))
sub['SalePrice'] = np.exp(y_test)
sub.to_csv('mysubmission_part2.csv', index=False)


#%% Part 3 - Smarter binarization

X_all = pd.concat((train, dev, test), sort=False).drop(columns=['SalePrice', 'Id'])
X_dummy = pd.get_dummies(X_all)

X_train = X_dummy.iloc[:ntrain,:].fillna(-1)
X_valid = X_dummy.iloc[ntrain:ntrain+nvalid,:].fillna(-1)
X_test = X_dummy.iloc[ntrain+nvalid:,:].fillna(-1)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_valid)
mean_squared_error(y_valid, y_pred)**0.5

coef = pd.Series(lr.coef_.ravel(), index=X_train.columns).sort_values()
coef.head(10)
coef.tail(10)

lr.intercept_
np.exp(lr.intercept_)

y_test = lr.predict(X_test) 

sub['SalePrice'] = np.exp(y_test)   
sub.to_csv('mysubmission_part3.csv', index=False)   



#%% Part 4 a
ridge = RidgeCV(alphas=10**np.arange(-5,2,0.5))
ridge.fit(X_train, y_train)
y_pred = ridge.predict(X_valid)
mean_squared_error(y_valid, y_pred)**0.5




#%% Part 4 b
X_all = pd.concat((train, dev, test), sort=False).drop(columns=['SalePrice', 'Id'])
X_all['LotArea2'] = X_all['LotArea']**2
X_all['GrLivArea2'] = X_all['GrLivArea']**2
X_dummy = pd.get_dummies(X_all)

X_train = X_dummy.iloc[:ntrain,:].fillna(-1)
X_valid = X_dummy.iloc[ntrain:ntrain+nvalid,:].fillna(-1)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_valid)
mean_squared_error(y_valid, y_pred)**0.5
# 0.12194483860617403


#%% Part 4 c
X_all = pd.concat((train, dev, test), sort=False).drop(columns=['SalePrice', 'Id'])
X_all['LotArea2'] = X_all['LotArea']**2
X_all['GrLivArea2'] = X_all['GrLivArea']**2
X_all['Years_since_remodeled'] = X_all['YearRemodAdd'] - X_all['YearBuilt']
X_dummy = pd.get_dummies(X_all)

X_train = X_dummy.iloc[:ntrain,:].fillna(-1)
X_valid = X_dummy.iloc[ntrain:ntrain+nvalid,:].fillna(-1)
X_test = X_dummy.iloc[ntrain+nvalid:,:].fillna(-1)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_valid)

y_test = lr.predict(X_test)

mean_squared_error(y_valid, y_pred)**0.5

sub['SalePrice'] = np.exp(y_test)
sub.to_csv('mysubmission_part5.csv', index=False)

#%% Part 4 d (best) 
X_all = pd.concat((train, dev, test), sort=False).drop(columns=['SalePrice', 'Id'])
X_all['LotArea2'] = X_all['LotArea']**2
X_all['GrLivArea2'] = X_all['GrLivArea']**2
X_dummy = pd.get_dummies(X_all)

X_train = X_dummy.iloc[:ntrain,:].fillna(-1)
X_valid = X_dummy.iloc[ntrain:ntrain+nvalid,:].fillna(-1)
X_test = X_dummy.iloc[ntrain+nvalid:,:].fillna(-1)

y_train = np.log(train['SalePrice'])[X_train['GrLivArea'] < 4600]
X_train = X_train.loc[X_train['GrLivArea'] < 4600,:]

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_valid)

y_test = lr.predict(X_test)

mean_squared_error(y_valid, y_pred)**0.5
# 0.12029517779102447

sub['SalePrice'] = np.exp(y_test)
sub.to_csv('mysubmission_part4.csv', index=False)



