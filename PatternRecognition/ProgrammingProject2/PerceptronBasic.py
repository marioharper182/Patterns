__author__ = 'Mario'

from pylab import rand,plot,show,norm

class Perceptron:
    def __init__(self, elements):
        """ perceptron initialization """
        self.w = rand(elements)*2-1 # weights
        self.wLength = elements
        self.learningRate = 0.05

    def response(self,x):
        """ perceptron output """
        y = 0
        for i in range(len(x)):
            y += x[i]*self.w[i]
        if y >= 0:
            return 1
        else:
            return -1

    def updateWeights(self,x,iterError):
        """
        updates the weights status, w at time t+1 is
           w(t+1) = w(t) + learningRate*(d-r)*x
        where d is desired output and r the perceptron response
        iterError is (d-r)
        """
        for i in range(self.wLength):
            self.w[i] += self.learningRate*iterError*x[i]

    def train(self,data,index):
        """
        trains all the vector in data.
        Every vector in data must have three elements,
        the third element (x[2]) must be the label (desired output)
        """
        learned = False
        iteration = 0
        size = len(data)
        while not learned:
            globalError = 0.0
            for i in range(size): # for each sample
                r = self.response(data[i])
                if index[i] != r: # if we have a wrong response
                    iterError = index[i] - r # desired response - actual response
                    self.updateWeights(data[i],iterError)
                    globalError += abs(iterError)
                iteration += 1
                if globalError == 0.0 or iteration >= 100: # stop criteria
                    # print 'iterations',iteration
                    learned = True # stop learning

Perceptron(3)