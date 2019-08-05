import numpy as np
import scipy.io as spio
import matplotlib.pyplot as plt
from q2_train_svm_primal import q2_train_svm_primal
from q2_visualize_linear_svm_model import q2_visualize_linear_svm_model
from q2_predict_svm_primal import q2_predict_svm_primal

# load the Matlab file
S = spio.loadmat('iris_subset.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']
Xtest = S['testsetX']
Ytest = S['testsetY']


# Try a bunch a hyperparameters for "C"
C_list = np.array([0.1, 1, 10, 100])

fig = plt.figure(figsize=(10,10))
count = 1
for C in C_list:

    # train the SVM (slack version), for a particular hyperparameter C
    [w, b, svs] = q2_train_svm_primal(X, Y, C)
    
    # visualize the model in a subplot
    plt.subplot(1, len(C_list), count)
    q2_visualize_linear_svm_model(X, Y, w, b, svs)
    plt.title('C=' + str(C))
    
    # calculate and print the training and test error
    pred_train_labels = q2_predict_svm_primal(X, w, b)[0]
    train_error = 100*np.sum(Y != pred_train_labels)/len(Y)
    pred_test_labels = q2_predict_svm_primal(Xtest, w, b)[0]
    test_error = 100*np.sum(Ytest != pred_test_labels)/len(Ytest)
    plt.title('C=' + str(C) + '\n #svs=' + str(len(svs)) + '\n tr error=' + "{:.3f}".format(train_error) + '%\n test error=' + "{:.3f}".format(test_error) + '%')       
    
    count = count + 1


# save the plot (Note: do not remove this line of code)
fig.savefig('q2e.png', dpi = 300)
plt.show()


