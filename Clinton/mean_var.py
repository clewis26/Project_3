# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 17:19:47 2016

@author: ttw
"""

import numpy as np
import theano.tensor as T
from theano import function

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=np.asarray(data_in, dtype=np.float32)

my_mean = np.mean(data_in, dtype='float32')
my_std = np.std(data_in,  dtype='float32')

x = T.fvector()
y = T.fscalar()
z = T.fscalar()

my_norm = ((x-y)/z)

calc_norm= function([x,y,z],my_norm)

data_in=calc_norm(data_in,my_mean,my_std)

