import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from q4_cross_validation_error import q4_cross_validation_error
from checking4b import checking4b

#checking
checking4b()

# Load data
S = spio.loadmat('autompg.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']
del S

# Try different lambda with a linear model
lambdavec = 10**np.linspace(-5, 7, 7, True)
error = q4_cross_validation_error(X, Y, lambdavec, 'linear', 10)
# Plot the results
fig = plt.figure(figsize=(10,10))
plt.subplot(2,1,1)
plt.plot(np.log10(lambdavec), error, '-db')
plt.ylabel('squared error per sample')
plt.title('cross validation error for regularized least square with b^l(x)')
plt.xlabel(r'$log_{10} \lambda$')
plt.grid()

# Try different lambda with a quadratic model
lambdavec = 10**np.linspace(-5, 7, 7, True)
error = q4_cross_validation_error(X, Y, lambdavec, 'quadratic', 10)
# Plot the results
plt.subplot(2,1,2)
plt.plot(np.log10(lambdavec), error, '-db')
plt.ylabel('squared error per sample')
plt.title('cross validation error for regularized least square with b^q(x)')
plt.xlabel(r'$log_{10} \lambda$')

plt.grid()
plt.show()

fig.savefig('q4b.png', dpi = 300)

