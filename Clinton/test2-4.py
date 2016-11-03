# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 17:50:52 2016

@author: ttw
"""

import gzip
import six.moves.cPickle as pickle

import numpy as np
import theano.tensor as T
from theano import function
from sklearn.model_selection import train_test_split
import pandas as pd

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=np.asarray(data_in, dtype=np.float32)

data_in_tst=np.fromfile('test_x.bin', dtype='uint8')
data_in_tst=np.asarray(data_in_tst, dtype=np.float32)

my_mean = np.mean(np.concatenate((data_in, data_in_tst)), dtype='float32')
my_std = np.std(np.concatenate((data_in, data_in_tst)),  dtype='float32')

x = T.fvector()
y = T.fscalar()
z = T.fscalar()

my_norm = ((x-y)/z)

calc_norm= function([x,y,z],my_norm)

data_in=calc_norm(data_in,my_mean,my_std)
data_in_tst==calc_norm(data_in_tst,my_mean,my_std)
del my_mean, my_std

data_in = data_in.reshape((100000,3600))
data_in_tst = data_in_tst.reshape((20000,3600))

     
data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in data_out.values.tolist() for val in sublist]
del val, sublist            

x_train, x_valid, y_train, y_valid = train_test_split(data_in,data_out, test_size=10000, random_state=2)
del data_in, data_out

train_set =(x_train,y_train)
del x_train, y_train
valid_set = (x_valid,y_valid)
del x_valid
test_set = (data_in_tst, np.concatenate((y_valid,y_valid)))
del data_in_tst, y_valid

