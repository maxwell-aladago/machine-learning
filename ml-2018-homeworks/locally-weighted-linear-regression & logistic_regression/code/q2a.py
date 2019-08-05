import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from q2_initialize import q2_initialize
from q2_train import q2_train
from q2_predict import q2_predict
from q2_error import q2_error
from checking2a import checking2a
# This script requires the following functions to be implemented:
# q2_initialize
# q2_predict
# q2_loglik
# q2_gradient
# q2_train
# q2_error



#checking
checking2a()

# load data
S = spio.loadmat('parkinsons.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY'][:].astype(np.float)
Xtest = S['testsetX']
Ytest = S['testsetY'][:].astype(np.float)
del S

# add constant feature set to 1 in order to implement the bias term
m = X.shape[0]
X = np.hstack((np.ones((m,1)), X))
m = Xtest.shape[0]
Xtest = np.hstack((np.ones((m,1)), Xtest))


alpha = 1e-6 # learning rate / step size
tol = 6.0 # tolerance on the norm of the gradient to decide when to stop

# initialize weights
theta_init = q2_initialize(X, Y, 'heuristic')

[theta, n_iter, loglik] = q2_train(X, Y, theta_init, alpha, tol)

pred_Y = q2_predict(X, theta)[0]
train_error = q2_error(Y, pred_Y)

pred_Ytest = q2_predict(Xtest, theta)[0]
test_error = q2_error(Ytest, pred_Ytest)

print('Number of iterations: ' + str(n_iter))
print('Misclassification rate on the training set: ' + str(train_error*100) + '%')
print('Misclassification rate on the test set: ' + str(test_error*100) + '%')

fig = plt.figure(figsize=(10,10))
plt.plot(np.linspace(1, n_iter, n_iter, True), loglik, 'o-')
plt.ylabel('Log likelihood');
plt.title('# training iterations: ' + str(n_iter) + ' --- training error rate: ' + str(train_error*100) + '% --- testing error rate: ' + str(test_error*100) + '%')
plt.xlabel('Number of iterations');
plt.grid()
fig.savefig('q2a.png', dpi = 300)
plt.show()
