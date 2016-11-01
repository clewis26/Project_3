
# coding: utf-8

# In[3]:

import numpy as np
import csv

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

class MLP(object):
    def __init__(self, sizes):
        self.depth = len(sizes)
        self.layers = sizes
        self.weights = [np.random.randn(y, x)
                       for x, y in zip(sizes[:-1], sizes[1:])]
        self.z = []
        self.activations = []
        self.yHat = []
        for _ in range(len(self.weights)):
            self.z.append([])
            self.activations.append([])
        
    def feedforward(self, a):
        i = 0
        for w in self.weights:
            currentZ = np.dot(w, a)
            self.z[i].append(currentZ.tolist())
            a = sigmoid(currentZ)
            self.activations[i].append(a)
            i+=1
        return a
        
    def costFunctionPrime(self, trainX, labels):
        predictions = np.asarray(self.yHat)
        activations = np.asarray(self.activations[0])
 
        delta3 = np.multiply(-(labels-predictions), sigmoidPrime(np.asarray(self.z[-1])))
        dJdW2 = np.dot(activations.T, delta3)
        
        delta2 = np.dot(delta3, self.weights[1])*sigmoidPrime(np.asarray(self.z[-2]))
        dJdW1 = np.dot(trainX.T, delta2)
        
        return dJdW1, dJdW2
        
        
                
    def training(self, trainX, labels, stepSize):
        for i in range(1000):
            del self.yHat[:]
            for j in range(len(self.weights)):
                del self.z[j][:]
                del self.activations[j][:]
            print(i,'th round')
            for item in trainX:
                self.yHat.append(self.feedforward(item))
            
            for item in self.yHat:
                for i in range(len(item)):
                    item[i] = 

            cost = 0.5*sum((labels-np.asarray(self.yHat))**2)
            print('cost:', cost)
            dJdW1, dJdW2 = self.costFunctionPrime(trainX, labels)
            self.weights[0] = self.weights[0] - stepSize*dJdW1.T
            self.weights[1] = self.weights[1] - stepSize*dJdW2.T
         evaluate(test_x, test_y)

     def evaluate(self, test_x, test_y):
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)
        #dgafd
     
"""
def normalize(data):
    fullData = [value.tolist() for value in data]
    fullData = [item for sublist in fullData for item in sublist]
    fullData = np.asarray(fullData)
    mu = np.mean(fullData)
    stdev = np.std(fullData)
    return mu, stdev
"""
"""
mlp = MLP([3600,300,19])
x = np.fromfile('', dtype='uint8')
x = x.reshape((100000,1,3600))
trainX = [row[0] for row in x]

mean = 112.389565483
stdev = 63.2726888553

fullData = [value.tolist() for value in trainX]
fullData = [item for sublist in fullData for item in sublist]
del trainX[:]

#outFile = open('normalized_data.csv', 'w')
#writer = csv.writer(outFile)

finalData = [] 
for i in range(0, 360000, 3600):
    chunk = fullData[i:i + 3600]
    for j in range(len(chunk)):
        chunk[j] = (chunk[j] - mean)/stdev
    finalData.append(chunk)
     

print('done')
outFile.close()

del fullData[:]
finalData = np.asarray(finalData)
#print(finalData)

labels = []
labelMap = np.matrix(np.identity(19), copy=False)
labelMap = labelMap.tolist()

with open('train_y.csv') as inFile:
    for i, row in enumerate(csv.reader(inFile)):
        if i == 0:
            continue
        labels.append(labelMap[int(row[1])])   

del labelMap[:]

"""

import os
import struct
import numpy as np

def read(dataset = "training"):
    x = []
    y = []
    if dataset is "training":
        fname_img = 'train-images.idx3-ubyte'
        fname_lbl = 'train-labels.idx1-ubyte'
    elif dataset is "testing":
        fname_img = 't10k-images.idx3-ubyte'
        fname_lbl = 't10k-labels.idx1-ubyte'

    # Load everything in some numpy arrays
    with open(fname_lbl, 'rb') as flbl:
        magic, num = struct.unpack(">II", flbl.read(8))
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    # Create an iterator which returns each image in turn
    for i in range(len(lbl)):
        pair = get_img(i)
        x.append(pair[1])
        y.append(pair[0])

    return x, y

train_x, train_y = read(dataset = "training")
test_x, test_y = read(dataset = "testing")
train_x = [item.tolist() for item in train_x]

labelMap = np.matrix(np.identity(10), copy=False)
labelMap = labelMap.tolist()
print(labelMap)
labels = []

print(len(train_y))
for i in train_y:
    labels.append(labelMap[i])


fullData = []
for item in train_x:
    tempRow = []
    for row in item:
        for i in range(len(row)):
            row[i] = row[i]/255.0
        tempRow.extend(row)
    fullData.append(tempRow)
    
fullData = np.asarray(fullData)
print(fullData[1])

mlp = MLP([784,50,10])
mlp.training(fullData, labels, 0.001)

