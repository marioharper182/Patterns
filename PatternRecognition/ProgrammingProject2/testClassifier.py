__author__ = 'Mario'

import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron as pla
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import pylab as pl
from PerceptronBasic import Perceptron
from pylab import norm
from sklearn.ensemble import AdaBoostClassifier


digitTraining = pd.read_table('./Data/zip_train_0_2.txt',delim_whitespace=True, header=None)
digitTest = pd.read_table('./Data/zip_test_0_2.txt',delim_whitespace=True, header=None)

digitTrainingMatrix = digitTraining.as_matrix()
indexDigitTraining = np.array([i[0] for i in digitTrainingMatrix])
digitTrainingMatrix = np.array([np.delete(i,0) for i in digitTrainingMatrix])
digitTestMatrix = digitTest.as_matrix()
indexDigitTest = np.array([i[0] for i in digitTestMatrix])
digitTestMatrix = np.array([np.delete(i,0) for i in digitTestMatrix])

digitClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(digitTrainingMatrix,indexDigitTraining).predict(digitTestMatrix)
digitClassifier2 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(digitTrainingMatrix, indexDigitTraining).predict(digitTestMatrix)
# digitClassifier2 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(digitTrainingMatrix,indexDigitTraining).predict(digitTestMatrix)
# digitVariables, digitsIndex = digits.data, digits.target
# digitClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(digitVariables,digitsIndex).predict(digitVariables)
# digitIndexClass0 = [1 if i == 0 else 0 for i in digitsIndex]
# digitIndexClass1 = [1 if i == 1 else 0 for i in digitsIndex]
# digitIndexClass2 = [1 if i == 2 else 0 for i in digitsIndex]

MissClassifiedDigitData = []
for i in range(len(digitClassifier)):
    if digitClassifier[i] != indexDigitTest[i]:
        MissClassifiedDigitData.append(1)
print("Number of missclassified Digits One Vs Many is:", np.sum(MissClassifiedDigitData))
print("Number of correctly Classified Digits is:", len(digitClassifier) - np.sum(MissClassifiedDigitData))

MissClassifiedDigitData2 = []
for i in range(len(digitClassifier2)):
    if digitClassifier2[i] != indexDigitTest[i]:
        MissClassifiedDigitData2.append(1)
print("Number of missclassified Digits One Vs One is:", np.sum(MissClassifiedDigitData2))
print("Number of correctly Classified Digits is:", len(digitClassifier2) - np.sum(MissClassifiedDigitData2))

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
index = np.array([i[0] for i in trainMatrix])
trainMatrix = np.array([np.delete(i, 0) for i in trainMatrix])
# Create the test Matrix
testMatrix = dataTest.as_matrix()
indexTest = np.array([i[0] for i in testMatrix])
testMatrix = np.array([np.delete(i, 0) for i in testMatrix])

wineClassifier = OneVsRestClassifier(LinearSVC(random_state=0)).fit(trainMatrix,index).predict(testMatrix)
wineClassifier2 = OneVsOneClassifier(LinearSVC(random_state=0)).fit(trainMatrix,index).predict(testMatrix)

MissClassifiedWineData = []
for i in range(len(wineClassifier)):
    if wineClassifier[i] != indexTest[i]:
        MissClassifiedWineData.append(1)
print("Number of missclassified Wine Data One Vs Many is:", np.sum(MissClassifiedWineData))
print("Number of correctly classified Wine Data is:", len(wineClassifier) - np.sum(MissClassifiedWineData))

MissClassifiedWineData2 = []
for i in range(len(wineClassifier2)):
    if wineClassifier2[i] != indexTest[i]:
        MissClassifiedWineData2.append(1)
print("Number of missclassified Wine Data One Vs One is:", np.sum(MissClassifiedWineData2))
print("Number of correctly classified Wine Data is:", len(wineClassifier2) - np.sum(MissClassifiedWineData2))

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
    # print success, errorInTest

n = norm(winePerceptronTrainer.w)
ww = winePerceptronTrainer.w/n

winePercep = pla(n_iter=5).fit(trainMatrix, indexTestClass3).predict(testMatrix)
# print(winePercep)

clf = QuadraticDiscriminantAnalysis()
clf.fit(digitTrainingMatrix, indexDigitTraining)
A = clf.predict(digitTestMatrix)

failRateQuadratic = 0
for i in range(len(A)):
    if A[i] != indexDigitTest[i]:
        failRateQuadratic += 1

print "Number of misclassifications in the Quadratic is: ", failRateQuadratic
print i

from sklearn.cross_validation import cross_val_score, cross_val_predict
clf = AdaBoostClassifier(n_estimators=50)
scores = cross_val_score(clf, digitTrainingMatrix, indexDigitTraining)
clf.fit(digitTrainingMatrix, indexDigitTraining)
digitAdaBoost = clf.predict(digitTestMatrix)

digitAdaFail = 0
for i in range(len(digitAdaBoost)):
    if digitAdaBoost[i] != indexDigitTest[i]:
        digitAdaFail += 1
print "Number of failures from AdaBoost on Digits Data is, ", digitAdaFail

clf2 = AdaBoostClassifier()
clf.fit(trainMatrix, index)
wineAdaBoost = clf.predict(testMatrix)

wineAdaFail = 0
for i in range(len(wineAdaBoost)):
    if wineAdaBoost[i] != indexTest[i]:
        wineAdaFail += 1

print "Number of failures from AdaBoost on Wine Data is: ", wineAdaFail

# print errorInTest
# print success

print "Finished"