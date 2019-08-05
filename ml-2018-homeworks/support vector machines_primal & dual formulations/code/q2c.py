import scipy.io as spio
import matplotlib.pyplot as plt
from q2_train_svm_primal_separable import q2_train_svm_primal_separable
from q2_visualize_linear_svm_model import q2_visualize_linear_svm_model

# load the Matlab file
S = spio.loadmat('iris_subset.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']

# train SVM, and report the hyperplane coefficients, bias, and number of SVs
[w, b, svs] = q2_train_svm_primal_separable(X, Y)
print('w = [' + str(w[0]) + ' ' + str(w[1]) + '],    b = ' + str(b))
print('Number of SVs: ' + str(len(svs)))

# visualize the trained SVM model: training data, decision boundary, margins, support vectors.

fig = plt.figure(figsize=(10,10))
q2_visualize_linear_svm_model(X, Y, w, b, svs);
plt.title('w = [' + str(w[0]) + ' ' + str(w[1]) + '],    b = ' + str(b) + '\n Number of SVs: ' + str(len(svs)))

# save the plot (Note: do not remove this line of code)
fig.savefig('q2c.png', dpi = 300)
plt.show()
