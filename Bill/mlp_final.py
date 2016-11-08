
# coding: utf-8

# In[3]:

import numpy as np
import csv, pickle, random
from matplotlib import pyplot as plt

#helper functions for calculating sigmoid and sigmoid prime
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoidPrime(z):
    return np.exp(-z)/((1+np.exp(-z))**2)

#multi-layer perceptron class
class MLP(object):
    #initialize layer weights at random and other variables of the network
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

    #method for calculating activation at each layer and the prediction at the last layer    
    def feedforward(self, a):
        i = 0
        for w in self.weights:
            currentZ = np.dot(w, a)
            self.z[i].append(currentZ.tolist())
            a = sigmoid(currentZ)
            self.activations[i].append(a)
            i+=1
        return a

    #method of backpropagation for calculating the gradient of the cost function i.e error surface    
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

    #method for training the network i.e mini-batch gradient descent, and evaluation of the model by
    #cross validation     
    def training(self, train_x, train_y, test_x, test_y, stepSize=0.01, epochs=1000, batchsize=50):
        lowestLoss = 1
        trainingIter = 0
        historicalGrad = []
        momentum = 0.001
        eps = 0.0000001
        decay_rate = 0.999
        combined = list(zip(train_x, train_y))
        cache1 = cache2 = cache3 = None
        allTrainLoss = []
        allTestLoss = []
        for i in range(epochs):
            random.shuffle(combined)
            miniBatches = [combined[k:k+batchsize] for k in range(0, len(combined), batchsize)]
            
            del self.yHat[:]
            for j in range(len(self.weights)):
                del self.z[j][:]
                del self.activations[j][:]

            print('round', i)

            for miniBatch in miniBatches:
                del self.yHat[:]
                for j in range(len(self.weights)):
                    del self.z[j][:]
                    del self.activations[j][:]
                    
                trainX, labels = zip(*miniBatch)
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

            print(self.weights[2][-1])
            
            trainingLoss = 1 - self.evaluate(train_x, train_y)/len(train_x)
            testLoss = 1 - self.evaluate(test_x, test_y)/len(test_x)
            
            allTrainLoss.append(trainingLoss)
            allTestLoss.append(testLoss)

            print('training loss', trainingLoss)
            print('testing loss: ', testLoss)

            if testLoss < lowestLoss:
                lowestLoss = testLoss
                trainingIter = i

        self.graphIterationLoss()
        return highestAccuracy, trainingIter

    #method for predicting label on the validation cases
    def evaluate(self, test_x, test_y):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in zip(test_x, test_y)]
        return sum(int(x == y) for (x, y) in test_results)

    def graphIterationLoss(allTrainLoss, allTestLoss):
        plt.figure(figsize=(8, 6), dpi=80)
        plt.subplot(1, 1, 1)
        plt.ylabel('Loss')
        plt.xlabel('Iteration')
        plt.plot(allTrainLoss)
        plt.plot(allTestLoss)
        plt.show()
        

import numpy
from scipy import signal
from scipy.stats import threshold

def loaddata():
    allData = []
    x = numpy.fromfile('train_x.bin', dtype='uint8')
    x = x.reshape((100000,60,60))

    for i in range(10000):
        grad = threshold(x[i], 240)
        #grad = numpy.maximum(grad, 0)
        #grad = max_pool(grad)
        grad = grad.reshape(1,3600)
        grad = grad[0].tolist()
        for i in range(len(grad)):
            if grad[i] > 200:
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
        if i>10000:
            break
        labels.append(labelMap[int(row[1])])   

del labelMap[:]
print(labels[0])
finalData = loaddata()
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

    mlp = MLP([3600,30,30,19])

    accuracy, iteration = mlp.training(fullTrainX, labels_train, fullTestX, labels_test, stepSize=0.001, epochs=10, batchsize=100)
    outFile = open('mlp_' + str(accuracy) + '_' + str(iteration) + '.p', 'wb')
    pickle.dump(mlp, outFile)
    outFile.close()
    
