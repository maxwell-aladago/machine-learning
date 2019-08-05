import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from q4_cross_validation_error import q4_cross_validation_error
from q4_test_error import q4_test_error
from checking4f import checking4f

#checking
checking4f()

# Load data
S = spio.loadmat('autompg.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']
Xtest = S['testsetX']
Ytest = S['testsetY']

# Try different lambda with the linear model.
lambdavec = 10**np.linspace(-5, 7, 7, True)
error_cross_validation = q4_cross_validation_error(X, Y, lambdavec, 'linear', 10)
error_test = q4_test_error(X, Y, Xtest, Ytest, lambdavec, 'linear')
fig = plt.figure(figsize=(10,10))
plt.subplot(2,1,1);
plt.plot(np.log10(lambdavec), error_cross_validation, '-db')
plt.plot(np.log10(lambdavec), error_test, '-*r')
plt.legend(['cross validation error', 'test error'])
plt.ylabel('squared error per sample')
plt.title('regularized least square with b^l(i)')
plt.xlabel(r'$log_{10} \lambda$')
plt.grid()


# Try different lambda with the quadratic model.
lambdavec = 10**np.linspace(-5, 7, 7, True)
error_cross_validation = q4_cross_validation_error(X, Y, lambdavec, 'quadratic', 10)
error_test = q4_test_error(X, Y, Xtest, Ytest, lambdavec, 'quadratic')
plt.subplot(2,1,2);
plt.plot(np.log10(lambdavec), error_cross_validation, '-db')
plt.plot(np.log10(lambdavec), error_test, '-*r')
plt.legend(['cross validation error', 'test error'])
plt.ylabel('squared error per sample')
plt.title('regularized least square with b^q(i)')
plt.xlabel(r'$log_{10} \lambda$')
plt.grid()

plt.show()

fig.savefig('q4f.png', dpi = 300)

