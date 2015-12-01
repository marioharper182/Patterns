__author__ = 'Mario'

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
import pylab as pl


digits = load_digits()

# pl.gray()
# pl.matshow(digits.images[2])
# pl.show()

wineVariables, wineIndex = digits.data, digits.target
digitClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(wineVariables,wineIndex).predict(wineVariables)


# This loads and processes the wine data set into the OneVsRest Classifier
dataTraining = pd.read_table('./Data/wine_uci_train.txt',delim_whitespace=True, header=None)
dataTest = pd.read_table('./Data/wine_uci_test.txt',delim_whitespace=True, header=None)
# Normalize the final column
wineMax = np.max(dataTraining[13])
wineMin = np.min(dataTraining[13])
dataTraining[13] = (dataTraining[13] - wineMin)/(wineMax-wineMin)
# Creating the classifier variables as a matrix, as well as the classification identifier
matrix = dataTraining.as_matrix()
index = [i[0] for i in matrix]
matrix = [np.delete(i, 0) for i in matrix]

wineClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(matrix,index).predict(matrix)



print "Finished"