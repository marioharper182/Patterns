__author__ = 'Mario'

from pylab import rand
import numpy as np

class Perceptron:
    def __init__(self, elements):
        self.w = rand(elements)*2-1 # Sets random weights
        self.wLength = elements
        self.learningRate = 0.05 # Set this to one for the naive Perceptron
        self.y = [] # instantiate a y misclassified list
        self.b =.0001

    def updateWeights(self,x,iterError):
        """
           w(t+1) = w(t) + learningRate*(d-r)*x
        """
        for i in range(self.wLength):
            self.w[i] += self.learningRate*iterError*x[i]

    def updateWeightsBatchRelax(self, x, iterError):
        for i in range(self.wLength):
            self.w[i] += self.w[i] + self.learningRate*np.sum((self.b-np.transpose(self.w))/np.linalg.norm(self.y)**2 * self.y)

    def response(self,x):
        y = 0
        for i in range(len(x)):
            y += x[i]*self.w[i]
        if y >= 0:
            return 1
        else:
            return -1

    def train(self,data,index):
        """
        This is the actual engine of the perceptron
        """
        learned = False
        iteration = 0
        size = len(data)
        while not learned:
            globalError = 0.0
            for i in range(size): # for each sample
                r = self.response(data[i])
                if index[i] != r: # if we have a wrong response
                    self.y.append(data[i])
                    iterError = index[i] - r # desired response - actual response
                    self.updateWeights(data[i],iterError)
                    globalError += abs(iterError)
                iteration += 1
                if globalError == 0.0 or iteration >= 100: # stop criteria
                    # print 'iterations',iteration
                    learned = True # stop learning

Perceptron(3)