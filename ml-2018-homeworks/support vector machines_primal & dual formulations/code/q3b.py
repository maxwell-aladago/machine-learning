import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from q3_leave_one_out_error import q3_leave_one_out_error

# load the data
S = spio.loadmat('parkinsons.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']


# set k parameters
k = np.uint8(np.linspace(1, 13, 7))

error = q3_leave_one_out_error(X, Y, k)

fig = plt.figure(figsize=(10,10))

plt.plot(k, error, 'bo-')
plt.ylabel('misclassification rate')
plt.title('kNN leave-one-out error on parkinsons dataset')
plt.xlabel('k')
plt.grid()
fig.savefig('q3b.png', dpi = 300)
plt.show()
