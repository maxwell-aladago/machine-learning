import numpy as np
import matplotlib.pyplot as plt
from draw_line import draw_line

# Do not modify this function, it has already been coded for you

def q2_visualize_linear_svm_model(X, Y, w, b, svs):
# Visualize the model of a linear SVM, given the parameters of the primal.
#
# INPUT:
#  X: [m x n] matrix, where each row is a d-dimensional input example
#  Y: [m x 1] vector, where the i-th element is the correct output value for the i-th input example. 
#  w: [n x 1] vector, containing the learned model parameters (coefficients of the hyperplane).
#  b: [1 x 1] scalar value, indicating the bias of the hyperplane.
#  svs: [nsv x 1] the indices of the training examples that are support vectors.

# ******************************************************************
# ****************** DO NOT EDIT THIS FUNCTION *********************
# ******************************************************************

    # visualize the training data
    # identify the positive and negative examples
    positive_idx = np.where(Y == 1)[0]
    negative_idx = np.where(Y == -1)[0]
    plt.plot(X[positive_idx,0], X[positive_idx,1], 'ro', lineWidth=2)
    plt.plot(X[negative_idx,0], X[negative_idx,1], 'bx', lineWidth=2)
    plt.legend(['positive examples', 'negative examples'])

    # visualize the decision boundary, and the pos/neg margin
    
    min_x = np.min(X[:,0])
    max_x = np.max(X[:,0])
    
    draw_line(w, b, min_x, max_x, 'k-');
    draw_line(w, b+1,  min_x, max_x, 'g--');
    draw_line(w, b-1,  min_x, max_x, 'g--');

    # draw the Support Vectors
    for i in range(len(svs)):
        if Y[svs[i]]==1:
            plt.plot(X[svs[i],0], X[svs[i],1], 'mo', lineWidth=2)
        else:
            plt.plot(X[svs[i],0], X[svs[i],1], 'mx', lineWidth=2)



