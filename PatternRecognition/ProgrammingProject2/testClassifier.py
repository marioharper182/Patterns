__author__ = 'Mario'

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
import pylab as pl
from PerceptronBasic import Perceptron
from pylab import norm


digits = load_digits()

# pl.gray()
# pl.matshow(digits.images[2])
# pl.show()

digitVariables, digitsIndex = digits.data, digits.target
digitClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(digitVariables,digitsIndex).predict(digitVariables)
digitIndexClass0 = [1 if i == 0 else 0 for i in digitsIndex]
digitIndexClass1 = [1 if i == 1 else 0 for i in digitsIndex]
digitIndexClass2 = [1 if i == 2 else 0 for i in digitsIndex]
digitIndexClass3 = [1 if i == 3 else 0 for i in digitsIndex]
digitIndexClass4 = [1 if i == 4 else 0 for i in digitsIndex]
digitIndexClass5 = [1 if i == 5 else 0 for i in digitsIndex]
digitIndexClass6 = [1 if i == 6 else 0 for i in digitsIndex]
digitIndexClass7 = [1 if i == 7 else 0 for i in digitsIndex]
digitIndexClass8 = [1 if i == 8 else 0 for i in digitsIndex]
digitIndexClass9 = [1 if i == 9 else 0 for i in digitsIndex]



# This loads and processes the wine data set into the OneVsRest Classifier
dataTraining = pd.read_table('./Data/wine_uci_train.txt',delim_whitespace=True, header=None)
dataTest = pd.read_table('./Data/wine_uci_test.txt',delim_whitespace=True, header=None)
# Normalize the final column
wineMax = np.max(dataTraining[13])
wineMin = np.min(dataTraining[13])
dataTraining[13] = (dataTraining[13] - wineMin)/(wineMax-wineMin)
dataTest[13] = (dataTest[13] - wineMin)/(wineMax-wineMin)
# Creating the classifier variables as a matrix, as well as the classification identifier
trainMatrix = dataTraining.as_matrix()
index = [i[0] for i in trainMatrix]
trainMatrix = [np.delete(i, 0) for i in trainMatrix]
# Create the test Matrix
testMatrix = dataTest.as_matrix()
indexTest = [i[0] for i in testMatrix]
testMatrix = [np.delete(i, 0) for i in testMatrix]

wineClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainMatrix,index).predict(trainMatrix)

indexClass1 = [1 if i == 1 else 0 for i in index]
indexClass2 = [1 if i == 2 else 0 for i in index]
indexClass3 = [1 if i == 3 else 0 for i in index]
winePerceptronTrainer = Perceptron(13)
winePerceptronTrainer.train(trainMatrix, indexClass1)

indexTestClass1 = [1 if i == 1 else 0 for i in index]
indexTestClass2 = [1 if i == 2 else 0 for i in index]
indexTestClass3 = [1 if i == 3 else 0 for i in index]
indexTestClass = []
# for j in range(1,4):
#     indexTestClass.append([1 if i == j else 0 for i in index])
indexTestClass.append(indexTestClass1)
indexTestClass.append(indexTestClass2)
indexTestClass.append(indexTestClass3)
sizeTest = len(testMatrix)
errorInTest = 0
success = 0
for j in range(3):
    errorInTest = 0
    success = 0
    for i in range(sizeTest):
        r = winePerceptronTrainer.response(testMatrix[i])
        if r == indexTestClass[j][i]:
            if indexTestClass[j][i] == 1:
                success += 1
            else:
                errorInTest += 1
    print success, errorInTest

n = norm(winePerceptronTrainer.w)
ww = winePerceptronTrainer.w/n



# print errorInTest
# print success

print "Finished"