__author__ = 'Mario'

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

rv = norm(loc=5.8, scale=1.79)

# print rv.pdf(5.8)
# X = np.linspace(2, 10)
#
# plt.plot(rv.pdf(X))
# plt.show()

def bayesParam(x):
    if x<4:
        return .295
    else:
        return 20.02*(x**-3 - 10**-3)

def getNorm(x):
    return rv.pdf(x)

def Compare(x):
    uniform = bayesParam(x)
    norm = getNorm(x)
    if uniform>norm:
        print("Uniform, Class 2")
    if norm > uniform:
        print("Norm, Class 1")

def findSlope(X1, X2, Y1, Y2):
    return (Y2-Y1)/(X2-X1)

def findDistance(X1, X2, Y1, Y2):
    return np.sqrt((X2-X1)**2+(Y2-Y1)**2)

omega1X = [-9.5, -16.5]
omega1Y = [-12.5, -13.0]
omega2X = [-7.5, -7.0]
omega2Y = [12.0, 6.0]
omega3X = [4.8, 5.0]
omega3Y = [-8.0, -10.0]
omega4X = [2.0, 3.0]
omega4Y = [3.0, 2.0]

plt.plot(omega1X, omega1Y, 'bo', omega2X, omega2Y, 'go', omega3X, omega3Y, 'ro',
         omega4X, omega4Y, 'mo', -3,-2, 'co')
plt.show()


XVar = np.var([-9.5, -16.5, -7.5, -7.0, 4.8, 5.0, 2.0, 3.0])
YVar = np.var([-12.5, -13.0, 12.0, 6.0, -8.0, -10.0, 3.0, 2.0])

XVar2 = np.var([-7.5, -7.0, 2.0, 3.0])
YVar2 = np.var([12.0, 6.0, 3.0, 2.0])

print XVar2, YVar2
print np.mean([-12.5, -13.0, 12.0, 6.0, -8.0, -10.0, 3.0, 2.0])

print ("Complete")
print( "None ")