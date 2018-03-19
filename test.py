import numpy as np
import random

import matplotlib.pyplot as plt


from jamboree import *


#===============================

N = 10

# Two arrays of random inputs
x = 2 * np.random.random((N,2)) - 1.

# Sum of random inputs
y = (x[:,0] + x[:,1]) / 4. + 0.5
y = np.matrix(y).T

Nt = 1000

#===============================

Tran1 = np.random.random((2,2))
Tran2 = np.random.random((2))

bias1 = np.random.random((2))
bias2 = np.random.random((1))

eta = 0.1

#===============================

total_errors, Tran1, bias1, Tran2, bias2 = trainruns_nnet_3layer(x,y,2,eta,Nt)

plt.plot(total_errors)
plt.show()
