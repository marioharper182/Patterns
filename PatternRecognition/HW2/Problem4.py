__author__ = 'Mario'

import Data
import numpy as np
from numpy.linalg import inv

# Calculate the mean of each category
omega1mean = [np.average(Data.omega1[i]) for i in range(3)]
omega2mean = [np.average(Data.omega2[i]) for i in range(3)]
omega3mean = [np.average(Data.omega3[i]) for i in range(3)]
# Calculate the covariance matrix
covOmega1 =  np.cov(np.array(Data.omega1).T)
covOmega2 =  np.cov(np.array(Data.omega2).T)
covOmega3 =  np.cov(np.array(Data.omega3).T)
# Priors
p1 = 1.0/3
p2 = 1.0/3
p3 = 1.0/3

def MahalanobisDistance(sample):
    # This will calculate our Mahalanobis Distance using our estimated Covariance
    omega1Mahalanobis = np.sqrt(np.dot(np.dot(np.subtract(sample,omega1mean).T,inv(covOmega1)),
                                       np.subtract(sample,omega1mean)))
    omega2Mahalanobis = np.sqrt(np.dot(np.dot(np.subtract(sample,omega2mean).T,inv(covOmega2)),
                                       np.subtract(sample,omega2mean)))
    omega3Mahalanobis = np.sqrt(np.dot(np.dot(np.subtract(sample,omega3mean).T,inv(covOmega3)),
                                       np.subtract(sample,omega3mean)))
    return omega1Mahalanobis, omega2Mahalanobis, omega3Mahalanobis

# Calculate the 4 given points
point1 = MahalanobisDistance([1,2,1])
point2 = MahalanobisDistance([5,3,2])
point3 = MahalanobisDistance([0,0,0])
point4 = MahalanobisDistance([1,0,0])

print(point1)
print(point2)
print(point3)
print(point4)
