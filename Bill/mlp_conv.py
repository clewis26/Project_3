
# coding: utf-8

# In[3]:

import numpy as np
import csv, pickle, random

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
        
        delta3 = np.multiply(-(labels-predictions), sigmoidPrime(np.asarray(self.z[-1])))
        activations2 = np.asarray(self.activations[1])
        dJdW3 = np.dot(activations2.T, delta3)

        delta2 = np.dot(delta3, self.weights[2])*sigmoidPrime(np.asarray(self.z[-2]))
        activations = np.asarray(self.activations[0])
        dJdW2 = np.dot(activations.T, delta2)
        
        delta = np.dot(delta2, self.weights[1])*sigmoidPrime(np.asarray(self.z[-3]))
        dJdW1 = np.dot(trainX.T, delta)
        
        return dJdW1, dJdW2, dJdW3
        
    def training(self, trainX, labels, test_x, test_y, stepSize=0.000001, epochs=1000, batchsize=4000):
        highestAccuracy = 0
        trainingIter = 0
        historicalGrad = []
        #momentum = 0.0001
        eps = 0.0000001
        decay_rate = 0.99

        cache1 = cache2 = cache3 = None
        
        combined = list(zip(trainX, labels))
        random.shuffle(combined)

        miniBatches = [combined[k:k+batchsize] for k in range(0, len(combined), batchsize)]
        
        for i in range(epochs):
            del self.yHat[:]
            for j in range(len(self.weights)):
                del self.z[j][:]
                del self.activations[j][:]
            print('round', i)
            #print(self.weights[0][:20])
            for miniBatch in miniBatches:
                del self.yHat[:]
                for j in range(len(self.weights)):
                    del self.z[j][:]
                    del self.activations[j][:]
                    
                trainX, labels = zip(*miniBatch)
                #print(len(trainX), len(labels))
                trainX = np.asarray(trainX)
                for item in trainX:
                    self.yHat.append(self.feedforward(item))
                
                dJdW1, dJdW2, dJdW3 = self.costFunctionPrime(trainX, labels)

                if cache1 == None:
                    cache1 = dJdW1**2
                    cache2 = dJdW2**2
                    cache3 = dJdW3**2
                else:
                    cache1 = decay_rate * cache1 + (1 - decay_rate) * dJdW1**2
                    self.weights[0] += - stepSize * dJdW1.T / (np.sqrt(cache1.T) + eps)

                    cache2 = decay_rate * cache2 + (1 - decay_rate) * dJdW2**2
                    self.weights[1] += - stepSize * dJdW2.T / (np.sqrt(cache2.T) + eps)

                    cache3 = decay_rate * cache3 + (1 - decay_rate) * dJdW3**2
                    self.weights[2] += - stepSize * dJdW3.T / (np.sqrt(cache3.T) + eps)
                
                #self.weights[0] = self.weights[0] - (momentum*self.weights[0] + stepSize*dJdW1.T)
                #self.weights[1] = self.weights[1] - (momentum*self.weights[1] + stepSize*dJdW2.T)
                #self.weights[2] = self.weights[2] - (momentum*self.weights[2] + stepSize*dJdW3.T)

            #print(self.weights[2][-1])
            testAccuracy = self.evaluate(test_x, test_y)/len(test_x)
            print('accuracy: ', testAccuracy )
            #cost = 0.5*sum((labels-np.asarray(self.yHat))**2)
            #print(cost)
            if testAccuracy > highestAccuracy:
                highestAccuracy = testAccuracy
                trainingIter = i
        return highestAccuracy, trainingIter

    def evaluate(self, test_x, test_y):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in zip(test_x, test_y)]
        return sum(int(x == y) for (x, y) in test_results)
     
"""
def normalize(data):
    fullData = [value.tolist() for value in data]
    fullData = [item for sublist in fullData for item in sublist]
    fullData = np.asarray(fullData)
    mu = np.mean(fullData)
    stdev = np.std(fullData)
    return mu, stdev

mlp = MLP([3600,50,30,19])
x = np.fromfile('', dtype='uint8')
x = x.reshape((100000,1,3600))
trainX = [x[i][0] for i in range(1000)]

mean = 112.389565483
stdev = 63.2726888553

fullData = [value.tolist() for value in trainX]
fullData = [item for sublist in fullData for item in sublist]
del trainX[:]

outFile = open('normalized_data.p', 'wb')

finalData = [] 
for i in range(0, len(fullData), 3600):
    chunk = fullData[i:i + 3600]
    for j in range(len(chunk)):
        chunk[j] = (chunk[j] - mean)/stdev
    finalData.append(chunk)

print('done')
pickle.dump(finalData, outFile)
outFile.close()

del fullData[:]
finalData = np.asarray(finalData)
"""

import numpy
#from PIL import Image
from scipy import signal

def max_pool(x):
    imageArray = []
    for i in range(0,len(x),2):
        row1 = x[i].tolist()
        row2 = x[i+1].tolist()
        for j in range(0,len(row1),2):
            maximum = max([row1[j], row1[j+1], row2[j], row2[j+1]])
            if maximum != 0:
                imageArray.append(1)
            else:
                imageArray.append(0)
    #imageArray = numpy.asarray(imageArray)
    #imageArray = imageArray.reshape(30,30)
    return imageArray

def convolution():
    allData = []
    #from scipy.misc import * # to visualize only
    x = numpy.fromfile('train_x.bin', dtype='uint8')
    x = x.reshape((100000,60,60))

    #scipy.misc.imshow(x[0].reshape(60,60)).show()
    scharr = numpy.array([[-1,-1,-1],
                       [-1,8,-1],
                       [-1,-1,-1]])
    for i in range(50000):
        x[i] = numpy.maximum(x[i], 230)
        grad = signal.convolve2d(x[i], scharr, boundary='symm', mode='same')
        grad = numpy.maximum(grad, 0)
        #grad = max_pool(grad)
        grad = grad.reshape(1,3600)
        grad = grad[0].tolist()
        for i in range(len(grad)):
            if grad[i] > 10:
                grad[i] = 1
            else:
                grad[i] = 0
        allData.append(grad)

    return allData

labels = []
labelMap = np.matrix(np.identity(19), copy=False)
labelMap = labelMap.tolist()

with open('train_y.csv') as inFile:
    for i, row in enumerate(csv.reader(inFile)):
        if i == 0:
            continue
        if i>50000:
            break
        labels.append(labelMap[int(row[1])])   

del labelMap[:]

finalData = convolution()
#print(finalData[1])
kFold = 5
for k in range(0, len(finalData), int(len(finalData)/kFold)):
    increment = int(len(finalData)/kFold)
    fullTestX = finalData[k:k+increment]
    labels_test = labels[k:k+increment]
    
    fullTrainX = finalData[:k] + finalData[k+increment:]
    labels_train = labels[:k] + labels[k+increment:]

    fullTestX = np.asarray(fullTestX)
    labels_test = np.asarray(labels_test)
    #fullTrainX = np.asarray(fullTrainX)
    #labels_train = np.asarray(labels_train)

    mlp = MLP([3600,50,50,19])

    accuracy, iteration = mlp.training(fullTrainX, labels_train, fullTestX, labels_test, stepSize=0.0001, epochs=4000)
    outFile = open('mlp_' + str(accuracy) + '_' + str(iteration) + '.p', 'wb')
    pickle.dump(mlp, outFile)
    outFile.close()
    
