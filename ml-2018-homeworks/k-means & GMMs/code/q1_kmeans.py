import numpy as np
from q1_dist2 import q1_dist2

def q1_kmeans(X, K, seeds_idx):
    # Executes K-means clustering algorithm, using euclidean distances.
    #
    # INPUT:
    #  X : a numpy.ndarray matrix of size [m x n] and type 'float', where each row
    #      is an n-dimensional input example
    #  K: 'int', indicating the number of centroids (i.e. hyperparameter "K" in K-means)
    #  seeds_idx: a numpy.ndarray vector of size [K x 1] and type 'int',
    #             containing the indices of the examples to be used as
    #             initial centroids.
    #             seeds_idx[i] is an integer number between 0 and m-1.
    #
    # OUTPUT:
    #  labels: a numpy.ndarray vector of size [m x 1] and type 'int',
    #          containing the labels that the K-means algorithm assigned to the examples.
    #          labels[i] is an element of {0, ..., K-1}, and it indicates the
    #          cluster ID associated to the i-th example
    #  means: a numpy.ndarray matrix of size [K x n] and type 'float',
    #         containing the K n-dimensional cluster centroids.
    #  distortions: a numpy.ndarray vector of size [num_iterations x 1] and type 'float'.
    #               Each element contains the  distortion at a particular iteration, i.e.
    #               the sum of the squared Euclidean distances between the examples
    #               and their associated centroids.

    # insert your code here
    means = X[seeds_idx]
    labels = np.argmin(q1_dist2(X, means), axis=1)

    distortion = 0

    # starting distortion
    for i in range(K):
        distortion = distortion + np.sum(q1_dist2(X[labels == i], means[i]))

    distortions = np.array([distortion])
    prev_distortion = distortion + 100  # just to enter the loop
    while np.linalg.norm(distortion - prev_distortion) >= 1e-6:

        prev_distortion = distortion
        # expectation step: assign labels to examples
        labels = np.argmin(q1_dist2(X, means), axis=1)

        # maximization step: recompute means. Also update distortion value
        distortion = 0
        for j in range(K):
            masks = np.array(labels == j, dtype='int')
            sum_i = sum(masks)
            means[j] = np.sum(masks[:, np.newaxis] * X, axis=0)/sum_i
            distortion = distortion + np.sum(q1_dist2(X[labels == j], means[j]))
        distortions = np.hstack((distortions, distortion))

    return (labels, means, distortions)

