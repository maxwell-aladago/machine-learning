import sys
import numpy as np
import scipy.io as spio
from q1_error import q1_error
from q1_nb_train import q1_nb_train
from q1_nb_predict import q1_nb_predict

def checking1a():
    check_data = spio.loadmat('q1a_checker.mat', squeeze_me=True)

    c = 0
    try:
        Y = check_data['Y']
        pred_Y = check_data['pred_Y']
        Xtrain = check_data['Xtrain']
        Ytrain = check_data['Ytrain']
        X = check_data['X']
        phi_y0 = check_data['phi_y0']
        phi_y1 = check_data['phi_y1']
        phi_prior = check_data['phi_prior']

        output_size1 = check_data['output_size1'][0]
        output_size2a = check_data['output_size2a'][0]
        output_size2b = check_data['output_size2b'][0]
        output_size2c = check_data['output_size2c'][0]
        output_size3 = check_data['output_size3'][0]

        error = q1_error(Y,pred_Y)
        clocal = 0
        if np.linalg.norm(np.shape(error) - output_size1):
            print('q1_error, WRONG OUTPUT:', np.shape(error), ', expected:', output_size1, '\n')
            c += 1
            clocal+=1
        if type(error) != float:
            print('q1_error: error, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if clocal == 0:
            print("q1_error successfully passed output size test!")

        phi_y0,phi_y1,phi_prior = q1_nb_train(Xtrain, Ytrain);
        clocal = 0
        if np.linalg.norm(np.shape(phi_y0) - output_size2a):
            print('q1_nb_train: phi_y0, WRONG OUTPUT:', np.shape(phi_y0), ', expected:', output_size2a, '\n')
            c += 1
            clocal += 1
        if np.linalg.norm(np.shape(phi_y1) - output_size2b):
            print('q1_nb_train: phi_y1, WRONG OUTPUT:', np.shape(phi_y1), ', expected:', output_size2b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(phi_prior) - output_size2c):
            print('q1_nb_train: phi_prior, WRONG OUTPUT:', np.shape(phi_prior), ', expected:', output_size2c, '\n')
            c += 1
            clocal += 1
        if type(phi_prior) != float:
            print('q1_nb_train: phi_prior, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if clocal == 0:
            print("q1_nb_train successfully passed output size test!")

        pred_y = q1_nb_predict(X,phi_y0,phi_y1,phi_prior);
        if np.linalg.norm(np.shape(pred_y) - output_size3):
            print('q1_nb_predict, WRONG OUTPUT:', np.shape(pred_y), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q1_nb_predict successfully passed output size test!")

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

