import numpy as np
import math
from q1_predict import q1_predict
from q1_mse import q1_mse

def q1_cross_validation_error(X, Y, tau_vec, N):
    
# Calculates the cross-validation errors for different values of tau_vec, given the
# training set X, Y.
#
# ** Implementation notes **
# - As discussed in class, you should first randomly permute the examples, before starting the
#   cross-validation stage. Here we did it for you: we created Xr and Yr which are obtained from
#   X and Y by permuting examples. You should use Xr and Yr in your code (not X and Y)
# - Do not change/initialize/reset the Python pseudo-number generator.
#
# INPUT
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input example
#  Y: a numpy.ndarray vector of size [m x 1] and type 'float', where the 
#     i-th element is the correct output value for the i-th input example. 
#  tau_vec: a numpy.ndarray vector of size [k x 1] and type 'float'
#             containing the set of regularization hyperparameter values
#  N: `int' representing the number of folds for the cross-validation stage
#
# OUTPUT
#  error: a numpy.ndarray vector of size [k x 1] and type 'float'
#         containing the cross-validation error (i.e., the average of the mean 
#         squared errors over the N validation sets) for each value in lambdavec.
#

# ********  DO NOT TOUCH THE FOLLOWING 5 LINES  ********************
    np.random.seed(0)
    m = X.shape[0]
    idxperm = np.random.permutation(m)-1
    Xr = X[idxperm,:]
    Yr = Y[idxperm]
# ******************************************************************
    
    # insert your code here
    # make sure to use Xr and Yr in your code, NOT X and Y (read Implemantiation notes in the header)

    error = np.zeros(tau_vec.shape[0])
    fold_size = math.ceil(Xr.shape[0]/N)
    m = Xr.shape[0]

    for run in range(N):
        fold_start_index = run * fold_size
        fold_end_index = fold_start_index + fold_size

        if fold_end_index > m: fold_end_index = m

        Xtrain = np.concatenate((Xr[0:fold_start_index], Xr[fold_end_index:m]), axis = 0)
        Ytrain = np.concatenate((Yr[0:fold_start_index], Yr[fold_end_index:m]), axis = 0)
        Xval = Xr[fold_start_index:fold_end_index]
        Yval = Yr[fold_start_index:fold_end_index]

        for i in range(tau_vec.shape[0]):
            tau_i = tau_vec[i]
            pred_Y = np.zeros(Yval.shape[0])
            for j in range(Xval.shape[0]):
                pred_Y[j] = q1_predict(Xtrain, Ytrain, Xval[j], tau_i)
            error[i] += q1_mse(Yval, pred_Y)

    error = error/N
    return error
