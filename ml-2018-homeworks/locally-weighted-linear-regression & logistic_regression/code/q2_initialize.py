import numpy as np

def q2_initialize(Xtrain, Ytrain, opt):
    # Initializes the weights for training logistic regression

    # INPUT
    #  Xtrain  : a numpy.ndarray matrix of size [m x n] and type 'float' where each row
    #            is a n-dimensional input example (assume it already contains the constant feature set to 1)
    #  Ytrain  : a numpy.ndarray vector of size [m x 1] and type 'float', where the
    #            i-th element is the correct labelfor the i-th input example.
    #  opt     : it is a `str', it can be either 'random' or 'heuristic' which allows to
    #            choose the initialization between randomly of heuristic

    # OUTPUT
    #  theta   : a numpy.ndarray vector of size [n x 1] and type 'float'
    #            containing the initialized parameter vector
    #
    # HINTS
    #  We provide the code for random initialization and ask you to implement
    #  the case of 'heuristic’, which we have discussed in class.


    m = Xtrain.shape[0]
    n = Xtrain.shape[1]
    if opt == 'random':
        # random initialization
        # ********  DO NOT TOUCH THE FOLLOWING 2 LINES  ********************
        np.random.seed(0)
        theta = np.random.normal(0, 1, n) # generate initial values
        # ******************************************************************    
    elif opt == 'heuristic':
        # "heuristic" initialization
        # insert your code here
        Ytrain = Ytrain.astype('float64')
        Ytrain[Ytrain==1] = .95
        Ytrain[Ytrain==0] = .05
        b = np.log((1-Ytrain)/Ytrain) * -1
        theta = np.linalg.lstsq(Xtrain, b, rcond=None)[0]
    else:
        print('Error, only random or heuristic initializations are supported')
        return []

    return theta
