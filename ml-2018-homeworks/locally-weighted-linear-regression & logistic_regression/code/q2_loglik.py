import numpy as np
from q2_predict import q2_predict

def q2_loglik(Xtrain, Ytrain, theta):
    # Computes the log likelihood value for training data (Xtrain, Ytrain) and parameter theta

    # INPUT
    #  Xtrain  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #            is a n-dimensional input example (assume it already contains the constant feature set to 1)
    #  Ytrain  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #            i-th element is the correct labelfor the i-th input example.
    #  theta   : a numpy.ndarray vector of size [n x 1] and type 'float'  containing
    #            the model parameters
    # OUTPUT
    #  lik     : float, the computed log likelihood

    prob_Y = q2_predict(Xtrain, theta)[1]
    prob_Y_0  = prob_Y - np.spacing(1)  # doing this helps np.log(1-prob(y)) to work. 
    lik = np.sum((Ytrain *  np.log(prob_Y)) + ((1-Ytrain)*np.log(1-prob_Y_0)))

    return lik
