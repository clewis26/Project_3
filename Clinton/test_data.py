# -*- coding: utf-8 -*-
"""
Created on Thu Nov 03 18:05:06 2016

@author: ttw
"""
import numpy as np
import scipy.misc # to visualize only
data_in_tst=np.fromfile('test_x.bin', dtype='uint8')
data_in_tst=np.asarray(data_in_tst, dtype=np.float32)


data_in_tst = data_in_tst.reshape((20000,60,60))
scipy.misc.imshow(data_in_tst[0]) # to visualize only     