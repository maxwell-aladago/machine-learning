import numpy as np

def q1_splitimgintiles(I, tilesize):
    # Split the input image into [tilesize x tilesize] tiles.
    #
    # INPUT:
    #  I : a numpy.ndarray matrix of size [r x c] and type 'float' corresponding to
    #      a grayscale image. Entry I[y,x] contains the intensity value of the pixel
    #      at row 'y' and column 'x'
    #  tilesize: 'int', indicating the size (width and height) of the tiles.
    #
    # OUTPUT:
    #  num_x_tiles: 'int', indicating the number of tiles along the x axis.
    #  num_y_tiles: 'int', indicating the number of tiles along the y axis.
    #  X : a numpy.ndarray matrix of size [(num_x_tiles*num_y_tiles) x (tilesize*tilesize)]
    #      and type 'float'. Each each row is the vectorized version of a square tile.
    #      The i-th row correponds to the i-th tile of the image. The tiles are
    #      stored in column order, i.e. top to bottom, left to right.
    #      Note that we vectorize each tile in a column-wise fashion.
    #
    # ******************************************************************
    # ****************** DO NOT EDIT THIS FUNCTION *********************
    # ******************************************************************


    r = I.shape[0]
    c = I.shape[1]

    num_x_tiles = np.int(np.floor(c/tilesize))
    num_y_tiles = np.int(np.floor(r/tilesize))

    num_pix_in_tile = np.int(tilesize**2)

    X = np.zeros((num_pix_in_tile, num_x_tiles*num_y_tiles))

    count = 0
    for j in range(num_x_tiles):
        for i in range(num_y_tiles):
            X[:, count] = np.reshape(I[i*tilesize:(i+1)*tilesize, j*tilesize:(j+1)*tilesize], num_pix_in_tile, 1)
            count = count + 1;

    X = np.transpose(X)

    return (num_x_tiles, num_y_tiles, X)
 
 