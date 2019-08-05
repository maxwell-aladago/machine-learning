import sys
import numpy as np
import scipy.io as spio
from q1_dist2 import q1_dist2
from q1_kmeans_select_seeds import q1_kmeans_select_seeds
from q1_kmeans import q1_kmeans
from q1_reconstructimgfromVQ import q1_reconstructimgfromVQ


def checking1a():
    check_data = spio.loadmat('q1a_checker.mat', squeeze_me=True)

    c = 0
    try:
        X1 = check_data['X1']
        X2 = check_data['X2']
        Xa = check_data['Xa']
        Xb = np.float_(check_data['Xb'])
        Ka = check_data['Ka']
        Kb = check_data['Kb']
        seeds_idx = check_data['seeds_idx']-1
        mode = check_data['mode']
        prototypes = np.transpose(check_data['prototypes'])
        tilesize = check_data['tilesize']
        tileidx = check_data['tileidx']-1
        num_x_tiles = check_data['num_x_tiles']
        num_y_tiles = check_data['num_y_tiles']
        output_size1 = check_data['output_size1']
        output_size2 = check_data['output_size2'][0]
        output_size3a = check_data['output_size3a'][0]
        output_size3b = check_data['output_size3b']
        output_size3c = check_data['output_size3c'][0]
        output_size4 = check_data['output_size4']

        D = q1_dist2(X1,np.reshape(X2, (1,X2.size)))
        if np.linalg.norm(np.shape(D) - output_size1):
            print('q1_dist2, WRONG OUTPUT:', np.shape(D), ', expected:', output_size1, '\n')
            c += 1
        else:
            print("q1_dist2 successfully passed output size test!")

        seeds_id = q1_kmeans_select_seeds(Xa,Ka,mode)
        if np.linalg.norm(np.shape(seeds_id) - output_size2):
            print('q1_kmeans_select_seeds, WRONG OUTPUT:', np.shape(seeds_id), ', expected:', output_size2, '\n')
            c += 1
        else:
            print("q1_kmeans_select_seeds successfully passed output size test!")

        labels,means,distortions = q1_kmeans(Xb,Kb,seeds_idx)
        clocal = 0
        if np.linalg.norm(np.shape(labels) - output_size3a):
            print('q1_kmeans: labels, WRONG OUTPUT:', np.shape(labels), ', expected:', output_size3a, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(means) - output_size3b):
            print('q1_kmeans: means, WRONG OUTPUT:', np.shape(means), ', expected:', output_size3b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(distortions) - output_size3c):
            print('q1_kmeans: distortions, WRONG OUTPUT:', np.shape(distortions), ', expected:', output_size3c, '\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q1_kmeans successfully passed output size test!")

        recI = q1_reconstructimgfromVQ(prototypes,tilesize,tileidx,num_x_tiles,num_y_tiles)
        if np.linalg.norm(np.shape(recI) - output_size4):
            print('q1_reconstructimgfromVQ, WRONG OUTPUT:', np.shape(recI), ', expected:', output_size4, '\n')
            c += 1
        else:
            print("q1_reconstructimgfromVQ successfully passed output size test!")

    except:
        c += 1
        print("cannot execute one of the functions required for q1a\n")

    if c != 0:
        print("Exiting q1a due to error\n")
        sys.exit()


def main():
    checking1a()


if __name__ == '__main__':
    main()

