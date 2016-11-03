# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 01:22:32 2016

@author: ttw
"""

import numpy as np
import theano
import theano.tensor as T
import pandas as pd 
from theano import function
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

N = 100000                                # training sample size
feats = 3600                             # number of input variables

## generate a dataset: D = (input_values, target_class)
#D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))
training_steps = 100

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=np.asarray(data_in, dtype=np.float32)

data_in_tst=np.fromfile('test_x.bin', dtype='uint8')
data_in_tst=np.asarray(data_in_tst, dtype=np.float32)

in_mean = np.concatenate((data_in,data_in_tst))
my_mean = np.mean(in_mean, dtype='float32')
my_std = np.std(in_mean,  dtype='float32')

x = T.fvector()
y,z = T.fscalars(2)

my_norm = ((x-y)/z)

calc_norm= function([x,y,z],my_norm)

data_in=calc_norm(data_in,my_mean,my_std)
data_in_tst==calc_norm(data_in_tst,my_mean,my_std)
del my_mean, my_std

data_in = data_in.reshape((100000,3600))
data_in_tst = data_in_tst.reshape((20000,3600))
     
data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in data_out.values.tolist() for val in sublist]
data_out=np.asarray(data_out, dtype=np.float32)            
del val, sublist            

x_train, x_valid, y_train, y_valid = train_test_split(data_in,data_out, random_state=2)

# Declare Theano symbolic variables
x = T.dmatrix("x")
y = T.dvector("y")

# initialize the weight vector w randomly
#
# this and the following bias variable b
# are shared so they keep their values
# between training iterations (updates)
w = theano.shared(np.random.randn(feats), name="w")

# initialize the bias term
b = theano.shared(0., name="b")

print("Initial model:")
print(w.get_value())
print(b.get_value())

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))   # Probability that target = 1
prediction = p_1 > 0.5                    # The prediction thresholded
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Cross-entropy loss function
cost = xent.mean() + 0.01 * (w ** 2).sum()# The cost to minimize
gw, gb = T.grad(cost, [w, b])             # Compute the gradient of the cost
                                          # w.r.t weight vector w and
                                          # bias term b
                                          # (we shall return to this in a
                                          # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 0.1 * gw), (b, b - 0.1 * gb)))
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
     print i     
     pred, err = train(x_train,y_train)

print("Final model:")
print(w.get_value())
print(b.get_value())
print accuracy_score(y_valid, predict(x_valid))