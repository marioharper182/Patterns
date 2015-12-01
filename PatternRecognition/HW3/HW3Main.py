__author__ = 'Mario'

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0.001, 5, 100)
exp1 = 1*np.exp(X*-1)
exp2 = 2*np.exp(-2*X)

plt.plot(exp1,X,exp2,X)
plt.show()