import numpy as np

def q1_features(X, mode):
# Given the data matrix X (where each row X[i,:] is an example), the function
# computes the feature matrix B, where row B[i,:] represents the feature vector 
# associated to example X[i,:]. The features should be either linear or quadratic
# functions of the inputs, depending on the value of the input argument 'mode'.
# Please make sure to implement the features according to the *exact* order
# specified in the text of the homework assignment.
#
# INPUT:
#  X: a numpy.ndarray matrix of size [m x d] and type 'float' where each row 
#     is a d-dimensional input example
#  mode: specifies the type of features; 
#        it is a 'str' that can be either 'linear' or 'quadratic'.
#
# OUTPUT:
#  B: a numpy.ndarray matrix of size [m x n] and type 'float', with each row 
#     containing the feature vector of an example
  
    m = X.shape[0]
    d = X.shape[1]
    
    if mode == 'linear':        
        B = np.hstack((np.ones((m, 1)), X))        
    elif mode == 'quadratic':
        B = []
    else:
        print('Error, only linear and quadratic forms are supported');
        return []
    
    return B
