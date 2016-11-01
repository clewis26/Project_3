# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 23:24:20 2016

@author: ttw
"""

import theano
a = theano.tensor.vector() # declare variable
b = theano.tensor.vector() # declare variable
out = a**2 + b**2 + 2*a*b               # build symbolic expression
f = theano.function([a,b], out)   # compile function
print(f([1, 2],[4,5]))