import numpy as np
from q2_calculate_bias_svm_dual import q2_calculate_bias_svm_dual
from q2_predict_svm_dual import q2_predict_svm_dual
import matplotlib.pyplot as plt

# Do not modify this function, it has already been coded for you

def q2_visualize_nonlinear_svm_model(X, Y, svs, alphas, mode):
# Visualize the model of a linear SVM, given the parameters of the primal.
#
# INPUT:
#  X: [m x n] matrix, where each row is a d-dimensional input example
#  Y: [m x 1] vector, where the i-th element is the correct output value for the i-th input example. 
#  svs: [nsv x 1] the indices of the training examples that are support vectors
#  alphas: [nsv x 1] the alpha coefficients associated to the support vectors
#  mode: the type of kernel; it is a string that can be either 'linear' or 'polynomial'

# ******************************************************************
# ****************** DO NOT EDIT THIS FUNCTION *********************
# ******************************************************************

    # calculate the bias from the learned model
    bias = q2_calculate_bias_svm_dual(X, Y, svs, alphas, mode)
    
    positive_idx = np.where(Y == 1)[0]
    negative_idx = np.where(Y == -1)[0]
    plt.plot(X[positive_idx,0], X[positive_idx,1], 'ro', lineWidth=2)
    plt.plot(X[negative_idx,0], X[negative_idx,1], 'bx', lineWidth=2)
    plt.legend(['positive examples', 'negative examples'])
    plt.xlabel('x_1');
    plt.ylabel('x_2');

    min_x = np.floor(np.min(X[:,0]))
    max_x = np.ceil(np.max(X[:,0]))
    min_y = np.floor(np.min(X[:,1]))
    max_y = np.ceil(np.max(X[:,1]))

    x1_idx = np.arange(min_x, max_x, 0.01)
    x2_idx = np.arange(min_y, max_y, 0.01)
    nsamples1 = len(x1_idx)
    nsamples2 = len(x2_idx)
    des_map = np.zeros((nsamples1, nsamples2))
    
    for i in range(nsamples1):
        for  j in range(nsamples2):
            xtest = [x1_idx[i], x2_idx[j]]
            des_map[i,j] = q2_predict_svm_dual(xtest, X, Y, svs, alphas, bias, mode)[1]

    curve = [];
    for i in range(nsamples1):
        idx = np.argmin(np.abs(des_map[i,:]))
        
        if abs(des_map[i,idx])<=1e-2:
            a = np.array([x1_idx[i], x2_idx[idx]]) 
            if len(curve)==0:
                curve = np.reshape(a, (1,2))
            else:
                curve = np.vstack((curve, np.reshape(a, (1,2))))                    

    plt.plot(curve[:,0], curve[:,1], 'g-', LineWidth=2)

    curve = []
    for i in range(nsamples1):
        idx = np.argmin(np.abs(des_map[i,:]-1.0))
    
        if abs(des_map[i,idx]-1.0)<=1e-2:
            a = np.array([x1_idx[i], x2_idx[idx]]) 
            if len(curve)==0:
                curve = np.reshape(a, (1,2))
            else:
                curve = np.vstack((curve, np.reshape(a, (1,2))))                    

    if len(curve)>0:
        plt.plot(curve[:,0], curve[:,1], 'c--', LineWidth=2)

    curve = []
    for i in range(nsamples1):
        idx = np.argmin(np.abs(des_map[i,:]+1.0))
    
        if abs(des_map[i,idx]+1.0)<=1e-2:
            a = np.array([x1_idx[i], x2_idx[idx]]) 
            if len(curve)==0:
                curve = np.reshape(a, (1,2))
            else:
                curve = np.vstack((curve, np.reshape(a, (1,2))))                    

    if len(curve)>0:
        plt.plot(curve[:,0], curve[:,1], 'c--', LineWidth=2)


