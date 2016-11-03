# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 10:53:32 2016

@author: ttw
"""

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import theano.tensor as T
from theano import function
#from sklearn.model_selection import train_test_split
import pandas as pd
from keras.utils.np_utils import to_categorical

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

data_out = to_categorical(data_out, 19)
#x_train, x_valid, y_train, y_valid = train_test_split(data_in,data_out, test_size=10000, random_state=2)
#del data_in,data_out


model = Sequential()

model.add(Dense(output_dim=150, input_dim=3600))
model.add(Activation("relu"))

model.add(Dense(output_dim=150))
model.add(Activation("relu"))

model.add(Dense(output_dim=19))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

model.fit(data_in, data_out, nb_epoch=100, batch_size=32)

#y_pred = model.predict_classes(x_valid, batch_size=32)
classes = model.predict_classes(data_in_tst, batch_size=32)

my_df = pd.DataFrame(classes)
my_df.columns=['Prediction']
my_df.to_csv(r'test_out1.csv', index_label="Id")

