import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from q2_initialize import q2_initialize
from q2_train import q2_train
from q2_train_line_search import q2_train_line_search
from q2_predict import q2_predict
from q2_error import q2_error
from checking2b import checking2b
# This script requires the following functions to be implemented:
# q2_initialize
# q2_predict
# q2_loglik
# q2_gradient
# q2_train
# q2_error
# q2_train_line_search

#checking
checking2b()

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


alpha = 10**-6 # learning rate for gradient ascent
large_alpha = 10**-4
tol = 6 # tolerance on the norm of the gradient to decide when to stop

theta_init = q2_initialize(X, Y, 'heuristic');
[theta1, n_iter1, loglik1] = q2_train(X, Y, theta_init, alpha, tol);

pred_Y = q2_predict(X, theta1)[0]
train_error1 = q2_error(Y, pred_Y)

pred_Ytest = q2_predict(Xtest, theta1)[0]
test_error1 = q2_error(Ytest, pred_Ytest)

print('Gradient ascent using fixed alpha=' + str(alpha))
print('Number of iterations: ' + str(n_iter1))
print('Misclassification rate on the training set: ' + str(train_error1*100) + '%')
print('Misclassification rate on the test set: ' + str(test_error1*100) + '%')


[theta2, n_iter2, loglik2] = q2_train_line_search(X, Y, theta_init, large_alpha, tol)

pred_Y = q2_predict(X, theta2)[0]
train_error2 = q2_error(Y, pred_Y)

pred_Ytest = q2_predict(Xtest, theta2)[0]
test_error2 = q2_error(Ytest, pred_Ytest)

print('\nGradient ascent using line search')
print('Number of iterations: ' + str(n_iter2))
print('Misclassification rate on the training set: ' + str(train_error2*100) + '%')
print('Misclassification rate on the test set: ' + str(test_error2*100) + '%')

fig = plt.figure(figsize=(10,10))
plt.plot(np.linspace(1, n_iter1, n_iter1, True), loglik1, 'bo--')
plt.plot(np.linspace(1, n_iter2, n_iter2, True), loglik2, 'rs--')
plt.ylabel('Log likelihood');
#plt.title('# training iterations: ' + str(n_iter) + ' --- training error rate: ' + str(train_error*100) + '% --- testing error rate: ' + str(test_error*100) + '%')
plt.xlabel('Number of iterations');
plt.legend(['fixed step','line search'])
plt.title('# training iterations w/ fixed alpha: ' + str(n_iter1) + ' --- # training iterations w/ line search: ' + str(n_iter2))
plt.grid()
fig.savefig('q2b.png', dpi = 300)
plt.show()

