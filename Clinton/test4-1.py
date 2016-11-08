# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 20:03:35 2016

@author: ttw
"""

from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
import numpy as np
import theano.tensor as T
from theano import function
import pandas as pd
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=2)

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

data_in = data_in.reshape((100000,1,60,60))
data_in_tst = data_in_tst.reshape((20000,1,60,60))
     
data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in data_out.values.tolist() for val in sublist]
del val, sublist            

data_out = to_categorical(data_out, 19)

model = Sequential()

model.add(Convolution2D(16, 3, 3,init='he_normal', border_mode='same', input_shape=(1,60, 60)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, 3, 3,init='he_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(16, 3, 3,init='he_normal', border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

model.add(Dense(200, init='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(100, init='he_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=19, init='he_normal'))
model.add(Activation("softmax"))

adm = Adam(lr=0.0001)
model.compile(loss='categorical_crossentropy', optimizer=adm, metrics=['accuracy'])

model.fit(data_in, data_out, nb_epoch=30, batch_size=32, verbose=1, validation_split=0.1, callbacks=[early_stopping])

#classes = model.predict_classes(data_in_tst, batch_size=50)
#
#my_df = pd.DataFrame(classes)
#my_df.columns=['Prediction']
#my_df.to_csv(r'test_out4.csv', index_label="Id")
#model.save('test_out4.h5')

