import sys
import numpy as np
import scipy.io as spio
from q1_top_words import q1_top_words
from q1_nb_train import q1_nb_train
from q1_nb_predict import q1_nb_predict

def checking1b():
    check_data = spio.loadmat('q1b_checker.mat', squeeze_me=True)

    c = 0
    try:
        Xtrain = check_data['Xtrain']
        Ytrain = check_data['Ytrain']
        X = check_data['X']
        phi_y0 = check_data['phi_y0']
        phi_y1 = check_data['phi_y1']
        phi_prior = check_data['phi_prior']
        phi_y0a = check_data['phi_y0a']
        phi_y1a = check_data['phi_y1a']
        phi_prior1 = check_data['phi_prior1']
        k = check_data['k']

        output_size1a = check_data['output_size1a'][0]
        output_size1b = check_data['output_size1b'][0]
        output_size1c = check_data['output_size1c'][0]
        output_size2 = check_data['output_size2'][0]
        output_size3 = check_data['output_size3']

        phi_y0,phi_y1,phi_prior = q1_nb_train(Xtrain, Ytrain);
        clocal = 0
        if np.linalg.norm(np.shape(phi_y0) - output_size1a):
            print('q1_nb_train: phi_y0, WRONG OUTPUT:', np.shape(phi_y0), ', expected:', output_size1a, '\n')
            c += 1
            clocal += 1
        if np.linalg.norm(np.shape(phi_y1) - output_size1b):
            print('q1_nb_train: phi_y1, WRONG OUTPUT:', np.shape(phi_y1), ', expected:', output_size1b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(phi_prior) - output_size1c):
            print('q1_nb_train: phi_prior, WRONG OUTPUT:', np.shape(phi_prior),', expected:', output_size1c, '\n')
            c += 1
            clocal += 1
        if type(phi_prior) != float:
            print('q1_nb_train: phi_prior, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if clocal == 0:
            print("q1_nb_train successfully passed output size test!")

        pred_y = q1_nb_predict(X,phi_y0,phi_y1,phi_prior);
        if np.linalg.norm(np.shape(pred_y) - output_size2):
            print('q1_nb_predict, WRONG OUTPUT:', np.shape(pred_y), ', expected:', output_size2, '\n')
            c += 1
        else:
            print("q1_nb_predict successfully passed output size test!")

        word_idx = q1_top_words(phi_y0a,phi_y1a,phi_prior1,k)
        if np.linalg.norm(np.shape(word_idx) - output_size3):
            print('q1_top_words, WRONG OUTPUT:', np.shape(word_idx), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q1_top_words successfully passed output size test!")


    except:
        c += 1
        print("cannot execute one of the functions required for q1b\n")

    if c != 0:
        print("Exiting q1b due to error\n")
        sys.exit()


def main():
    checking1b()


if __name__ == '__main__':
    main()

