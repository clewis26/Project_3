# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 01:22:32 2016

@author: ttw
"""

import numpy as np
import theano.tensor as T
import pandas as pd 
from theano import function
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# import train_x data
data_in=np.fromfile('train_x.bin', dtype='uint8')
# Convert to float32 for GPU
data_in=np.asarray(data_in, dtype=np.float32)

#import test x data
data_in_tst=np.fromfile('test_x.bin', dtype='uint8')
data_in_tst=np.asarray(data_in_tst, dtype=np.float32)

# find mean and std_dev for normalization across test and train x inputs
my_mean = np.mean(np.concatenate((data_in,data_in_tst)), dtype='float32')
my_std = np.std(np.concatenate((data_in,data_in_tst)),  dtype='float32')

# define theano variables, 1 vector and 2 scalar
x = T.fvector()
y,z = T.fscalars(2)

# normalizing equation
my_norm = ((x-y)/z)

# compile equation as a theano function
calc_norm= function([x,y,z],my_norm)

# call theano function to normalize both test and train x inputs
data_in=calc_norm(data_in,my_mean,my_std)
data_in_tst==calc_norm(data_in_tst,my_mean,my_std)
del my_mean, my_std

# reshape to 100000x3600 matrix
data_in = data_in.reshape((100000,3600))
data_in_tst = data_in_tst.reshape((20000,3600))
     
# Read the train y data for training model
data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in data_out.values.tolist() for val in sublist]
data_out=np.asarray(data_out, dtype=np.float32)            
del val, sublist            

# split training data into train and validation set, by default 75% train 25% validation
###############delete the 2 lines below if you want to train on entire set and predict on the given test set
x_train, x_valid, y_train, y_valid = train_test_split(data_in,data_out, random_state=2)
del data_in, data_out

# Declare LogReg function from scikit learn and pass paramters
logreg = LogisticRegression(solver='sag', multi_class='multinomial', random_state=1, verbose=10)

# train data on logreg
############### use logreg.fit(data_in,data_out) if you want to train on entire training set
logreg.fit(x_train, y_train)

# predict data on validation set
###################### use y_pred = logreg.predict(data_in_tst) to predict y values for test set
y_pred = logreg.predict(x_valid)

# print accuracy score
############ delete below line if predicting on test set
print logreg.score(x_valid, y_valid)