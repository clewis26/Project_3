import numpy as np
import theano.tensor as T
from theano import function

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=np.asarray(data_in, dtype=np.float32)

data_in_tst=np.fromfile('test_x.bin', dtype='uint8')
data_in_tst=np.asarray(data_in_tst, dtype=np.float32)

in_mean = np.concatenate((data_in,data_in_tst))
my_mean = np.mean(in_mean, dtype='float32')
my_std = np.std(in_mean,  dtype='float32')


x = T.fvector()
y = T.fscalar()
z = T.fscalar()

my_norm = ((x-y)/z)

calc_norm= function([x,y,z],my_norm)

data_in=calc_norm(data_in,my_mean,my_std)

