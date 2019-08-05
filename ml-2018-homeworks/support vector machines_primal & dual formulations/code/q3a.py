import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from q3_test_error import q3_test_error

# load the data
S = spio.loadmat('parkinsons.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']
Xt = S['testsetX']
Yt = S['testsetY']

# set k parameters
k = np.uint8(np.linspace(1, 13, 7))

error = q3_test_error(X, Y, Xt, Yt, k)

fig = plt.figure(figsize=(10,10))

plt.plot(k, error, 'bo-')
plt.ylabel('misclassification rate')
plt.title('kNN test error on parkinsons dataset')
plt.xlabel('k')
plt.grid()
fig.savefig('q3a.png', dpi = 300)
plt.show()
