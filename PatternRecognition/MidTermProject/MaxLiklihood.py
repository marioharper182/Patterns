__author__ = 'Mario'

__author__ = 'Mario'

import numpy as np
from scipy.stats import multivariate_normal as norm
import pandas as pd
import matplotlib.pyplot as plt

dataTraining = pd.read_table('./Data/iris_training.txt',delim_whitespace=True, header=None)
dataTest = pd.read_table('./Data/iris_test.txt',delim_whitespace=True, header=None)
#
# dataTraining = pd.read_table('./Data/wine_uci_train.txt',delim_whitespace=True, header=None)
# dataTest = pd.read_table('./Data/wine_uci_test.txt',delim_whitespace=True, header=None)

n = len(dataTraining)
nClassifiers = len(dataTraining.loc[0])
nTypesIris = 3
def readIrisTraining():
    pass
    # print dataIris[:3]
    # print "this is: ", dataIris[0]
    # print n

def gaussian(x, mu, cov):
    # cov = np.cov([dataIris[dataIris[0]==1][1],dataIris[dataIris[0]==1][2]],
    #              dataIris[dataIris[0]==1][3],dataIris[dataIris[0]==1][4])
    # print cov
    # sigma1 = np.average(sigma)
    # x = np.linspace(-4*sigma1,4*sigma1,4)
    normPDF = norm.pdf(x,mu,cov)
    return normPDF

def estimatedParameters(xArray):
    mu = []
    sigma = []
    for i in range(1,nClassifiers):
        mu.append(xArray[i].mean())
        sigma.append((1.0/n)*(np.sum((xArray[i]-mu[i-1])**2)))
    return mu, sigma

readIrisTraining()
parametersMu = []
parametersSigma = []
parametersCov = []
indexTable = dataTraining[dataTraining[0]]
for i in range(1,nTypesIris+1):
    mu, sigma = estimatedParameters(dataTraining[dataTraining[0]==i])
    parametersMu.append(mu)
    parametersSigma.append(sigma)
    covList = []
    for j in range(1, nClassifiers):
        covList.append(dataTraining[dataTraining[0]==i][1])
    cov = np.cov([dataTraining[dataTraining[0]==i][1],dataTraining[dataTraining[0]==i][2],
                  dataTraining[dataTraining[0]==i][3],dataTraining[dataTraining[0]==i][4]])
    cov = np.cov(np.array(covList))
    parametersCov.append(cov)

parametersLen = len(parametersMu)
SuccessList = []
for i in range(0, len(dataTest)):
    tempXMaximum = 0
    maxDistribution = 0
    x = dataTest.loc[i]
    for j in range(0,parametersLen):

        tempX = (gaussian(x[1:], parametersMu[j], parametersCov[j]))
        if tempX > tempXMaximum:
            tempXMaximum = tempX
            maxDistribution = j

    if dataTest[0][i] == maxDistribution+1:
        SuccessList.append(1)
    else:
        SuccessList.append(0)

# print(SuccessList)
print(np.average(SuccessList))
# np.cov([dataIris[dataIris[0]==1][1],dataIris[dataIris[0]==1][2],dataIris[dataIris[0]==1][3],dataIris[dataIris[0]==1][4]])