# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 17:01:34 2016

@author: ttw
"""

import numpy as np
import theano.tensor as T
from theano import function
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
from sklearn.linear_model import LogisticRegression

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=np.asarray(data_in, dtype=np.float32)
#
#data_in_tst=np.fromfile('test_x.bin', dtype='uint8')
#data_in_tst=np.asarray(data_in_tst, dtype=np.float32)

in_mean = data_in
my_mean = np.mean(in_mean, dtype='float32')
my_std = np.std(in_mean,  dtype='float32')


x = T.fvector()
y = T.fscalar()
z = T.fscalar()

my_norm = ((x-y)/z)

calc_norm= function([x,y,z],my_norm)

data_in=calc_norm(data_in,my_mean,my_std)
#data_in_tst==calc_norm(data_in_tst,my_mean,my_std)
del my_mean, my_std

data_in = data_in.reshape((100000,3600))
#data_in_tst = data_in_tst.reshape((20000,3600))

     
data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in data_out.values.tolist() for val in sublist]
data_out=np.asarray(data_out, dtype=np.float32)            
del val, sublist            

data_in= data_in[:10000,:]
data_out=data_out[:10000]

x_train, x_valid, y_train, y_valid = train_test_split(data_in,data_out, random_state=2)

logreg = LogisticRegression(C=1, solver='sag', multi_class='multinomial', random_state=1, verbose=10)

parameters = { 'C':[1,1.1,1.3,1.5,1.7,1.9]}
clf = GridSearchCV(logreg, parameters, verbose=10)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_valid)

print clf.score(x_valid, y_valid)