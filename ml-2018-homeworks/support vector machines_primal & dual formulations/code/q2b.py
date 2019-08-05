import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt

# load the Matlab file
S = spio.loadmat('iris_subset.mat', squeeze_me=True)
trainsetX = S['trainsetX']
trainsetY = S['trainsetY']

# identify the positive and negative examples
positive_idx = np.where(trainsetY == 1)[0]
negative_idx = np.where(trainsetY == -1)[0]

# plot
fig = plt.figure(figsize=(10,10))


plt.plot(trainsetX[positive_idx,0], trainsetX[positive_idx,1], 'ro');
plt.plot(trainsetX[negative_idx,0], trainsetX[negative_idx,1], 'bx');
plt.xlabel(r'$x_1$');
plt.ylabel(r'$x_2$');
plt.legend(['positive examples', 'negative examples']);

# save the plot (Note: do not remove this line of code)
plt.grid()
fig.savefig('q2b.png', dpi = 300)
plt.show()
