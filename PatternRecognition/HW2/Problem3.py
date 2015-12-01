__author__ = 'Mario'

import Data
import numpy as np

# TODO: This mean needs to be computed in a loop that 'leave's out' the sample in question
omega1mean = [np.average(Data.omega1[i]) for i in range(3)]
omega2mean = [np.average(Data.omega2[i]) for i in range(3)]
omega3mean = [np.average(Data.omega3[i]) for i in range(3)]

def LeaveOneOut():
    # Creates arrays of classifications based on the distance
    omega1List, omega2List, omega3List = [], [], []
    for i in range(10):
        omega1List.append(CheckClosest(Data.omega1[i]))
        omega2List.append(CheckClosest(Data.omega2[i]))
        omega3List.append(CheckClosest(Data.omega3[i]))
    # Return the probability of what we classified every item in our training set as
    return ComputeAccuracy( (omega1List, omega2List, omega3List) )

def CheckClosest(points):
    # This function checks to see what the closest distribution is
    dist1 = np.sqrt(np.sum(np.subtract(points, omega1mean)**2))
    dist2 = np.sqrt(np.sum(np.subtract(points, omega2mean)**2))
    dist3 = np.sqrt(np.sum(np.subtract(points, omega3mean)**2))
    if (min(dist1, dist2, dist3) == dist1):
        return 'omega1'
    if (min(dist1, dist2, dist3) == dist2):
        return 'omega2'
    if (min(dist1, dist2, dist3) == dist3):
        return 'omega3'

def ComputeAccuracy(list):
    # Check to see if what we got is accurate
    probOmega1, probOmega2, probOmega3 = list
    print(list)
    count1, count2, count3 = 0,0,0
    for i in probOmega1:
        if i == 'omega1':
            count1+=1
    for i in probOmega2:
        if i == 'omega2':
            count2+=1
    for i in probOmega3:
        if i == 'omega3':
            count3+=1
    return count1/10.0, count2/10.0, count3/10.0

A, B, C = LeaveOneOut()
print 'The accuracy of prediction for omega1: ', A
print 'The accuracy of prediction for omega2: ', B
print 'The accuracy of prediction for omega3: ', C
