import numpy as np

def q1_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles):
    # Reconstructs an image starting from the VQ model.
    #
    # INPUT:
    #  prototypes: a numpy.ndarray matrix of size [K x n] and type 'float',
    #              containing the K n-dimensional cluster centroids (i.e., the learned
    #              K prototypical tiles.
    #  tilesize: 'int', indicating the size (width and height) of the tiles.
    #  tileidx: a numpy.ndarray vector of size [m x 1] and type 'int',
    #           containing the labels that the K-means algorithm assigned to the examples.
    #           tileidx[i] is an element of {0, ..., K-1}, and it indicates the
    #           cluster/prototype ID associated to the i-th example/tile.
    #           Note that tileidx stores the tiles in column order
    #           (see comments in file q5_splitimgintiles.m)
    #  num_x_tiles: 'int', indicating the number of tiles along the x axis.
    #  num_y_tiles: 'int', indicating the number of tiles along the y axis.
    #
    # OUTPUT:
    #  recI : a numpy.ndarray matrix of size [r x c] and type 'float' corresponding to
    #         the reconstructed image.


    # insert your code here

    recI = np.zeros((num_y_tiles * tilesize, num_x_tiles*tilesize))
    count = 0
    for i in range(num_x_tiles):
        for j in range(num_y_tiles):
            prototype = prototypes[tileidx[count]]
            recI[j * tilesize:(j + 1)*tilesize, i * tilesize:(i + 1)*tilesize] = prototype.reshape(tilesize, tilesize).T
            count += 1
    return recI

