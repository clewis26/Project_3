# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 12:15:44 2016

@author: ttw
"""

import numpy as np
import pandas as pd
from sklearn import model_selection, svm

data_in=np.fromfile('train_x.bin', dtype='uint8')
data_in=data_in.reshape((100000,3600))

data_in=data_in/255.
data_in= data_in[:10000]

data_out = pd.read_csv(r'train_y.csv',header=None, skiprows=1, usecols=[1])
data_out = [val for sublist in data_out.values.tolist() for val in sublist]
data_out=data_out[:10000]            

del val,sublist        

x_train, x_test, y_train, y_test = model_selection.train_test_split(data_in, data_out, random_state=2)

del data_in, data_out

logreg = svm.LinearSVC(verbose=10, random_state=1)

logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

print logreg.score(x_test, y_test)