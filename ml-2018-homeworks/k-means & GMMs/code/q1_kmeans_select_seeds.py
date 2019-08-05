import numpy as np
from q1_dist2 import q1_dist2

def q1_kmeans_select_seeds(X, K, mode):
    # Returns an initial set of centroids (i.e. a set of seeds) for the K-means algorithm.
    #
    # INPUT:
    #  X : a numpy.ndarray matrix of size [m x n] and type 'float', where each row
    #      is an n-dimensional input example
    #  K: 'int', indicating the number of centroids (i.e. hyperparameter "K" in K-means)
    #  mode: 'str', indicating the type of initilization.
    #        It can be either 'random' or 'diverse_set'.
    #
    # OUTPUT:
    #  seeds_idx: a numpy.ndarray vector of size [K x 1] and type 'int',
    #             containing the indices of the examples to be used as
    #             initial centroids.
    #             seeds_idx[i] should be an integer number between 0 and m-1.

    m = X.shape[0]
    n = X.shape[1]
    if mode == 'random':
        # random initialization
        idxperm = np.random.permutation(m)
        seeds_idx = idxperm[:K]
    elif mode == 'diverse_set':
        # always use first value as the starting centroid; deterministic
        seeds_idx = np.array([0], dtype='int')
        centroids = X[seeds_idx]

        # find remaining k-1 centroid using max of min distances of examples from
        # selected centroids
        for i in range(1, K):
            D = q1_dist2(X, centroids)
            i_th_id = np.argmax(np.min(D, axis=1))
            seeds_idx = np.hstack((seeds_idx, i_th_id))
            centroids = X[seeds_idx]

    else:
        print('Error, only random and diverse_set modes are supported')
        return []

    return seeds_idx
