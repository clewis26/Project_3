# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 01:23:05 2016

@author: ttw
"""
import numpy as np
import pandas as pd
from sklearn import model_selection, neural_network

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=data_in.reshape((100000,60,60))
#
#data_in=data_in/255.
#
#data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
#data_out = [val for sublist in data_out.values.tolist() for val in sublist]
#
#del val,sublist        
#
#x_train, x_test, y_train, y_test = model_selection.train_test_split(data_in, data_out, random_state=2)
#
#del data_in, data_out
#
#mlp = neural_network.MLPClassifier(hidden_layer_sizes=(100,100,),activation='logistic',solver='adam', alpha = 0.0001, batch_size='auto', learning_rate='constant', max_iter=200, tol=1e-4, learning_rate_init=0.001, verbose=10,random_state=1)
#
#mlp.fit(x_train,y_train)
#
#print("Training set score: %f" % mlp.score(x_train, y_train))
#print("Test set score: %f" % mlp.score(x_test, y_test))