import scipy.io as spio
import matplotlib.pyplot as plt
from q2_train_svm_dual import q2_train_svm_dual
from q2_visualize_nonlinear_svm_model import q2_visualize_nonlinear_svm_model

# load the Matlab file
S = spio.loadmat('iris_subset.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']


# train the SVM model (linear case)
mode = 'linear'
C = 100
[svs, alphas] = q2_train_svm_dual(X, Y, C, mode)
print('Linear model. # SVs: ' + str(len(svs)))
# plot
fig = plt.figure(figsize=(10,10))
plt.subplot(1, 2, 1)
q2_visualize_nonlinear_svm_model(X, Y, svs, alphas, mode)
plt.title('Linear model. # SVs: ' + str(len(svs)))

# train the SVM model (polynomial case)
mode = 'polynomial'
C = 100
[svs, alphas] = q2_train_svm_dual(X, Y, C, mode)
print('Polynomial model. # SVs: ' + str(len(svs)))
# plot
plt.subplot(1, 2, 2)
q2_visualize_nonlinear_svm_model(X, Y, svs, alphas, mode)
plt.title('Polynomial model. # SVs: ' + str(len(svs)))

# save the plot (Note: do not remove this line of code)
fig.savefig('q2h.png', dpi = 300)
plt.show()

