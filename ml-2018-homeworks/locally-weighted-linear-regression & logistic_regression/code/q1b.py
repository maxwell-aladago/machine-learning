import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from q1_test_error import q1_test_error
from checking1b import checking1b

# dependent functions
# q1_W
# q1_train
# q1_predict
# q1_test_error
# q1_features
# q1_mse

checking1b()

# Load data
S = spio.loadmat('autompg.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']
Xtest = S['testsetX']
Ytest = S['testsetY']


# tau hyperparameter values
tau_vec = 10.0**np.array([2.0, 3.0, 5.0, 6.0])

# perform full evaluation on test set
test_errors = q1_test_error(X, Y, Xtest, Ytest, tau_vec)

# plotting test errors versus different tau values
fig = plt.figure(figsize=(10,10))
plt.plot(np.log10(tau_vec), test_errors, '-*r')
plt.ylabel('mean squared error')
plt.title('locally weighted least square with b^l(x)')
plt.xlabel(r'$log_{10} \tau$')
plt.grid()
fig.savefig('q1b.png', dpi = 300)
plt.show()


