
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
            
            cost = 0.5*sum((labels-np.asarray(self.yHat))**2)
            print('cost:', cost)
            dJdW1, dJdW2 = self.costFunctionPrime(trainX, labels)
            self.weights[0] = self.weights[0] - stepSize*dJdW1.T
            self.weights[1] = self.weights[1] - stepSize*dJdW2.T

    #def validate():
        
        

def normalize(data):
    fullData = [value.tolist() for value in data]
    fullData = [item for sublist in fullData for item in sublist]
    fullData = np.asarray(fullData)
    mu = np.mean(fullData)
    stdev = np.std(fullData)
    return mu, stdev


mlp = MLP([3600,300,19])
x = np.fromfile('train_x.bin', dtype='uint8')
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

mlp.training(finalData, labels[:100], 0.01)


