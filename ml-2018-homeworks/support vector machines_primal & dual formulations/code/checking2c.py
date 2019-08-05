import sys
import numpy as np
import scipy.io as spio
from q2_train_svm_primal_separable import q2_train_svm_primal_separable

def checking2c():
    check_data = spio.loadmat('q2c_checker.mat', squeeze_me=True)

    c = 0
    try:
        X = check_data['X']
        Y = check_data['Y']
        output_size1a = check_data['output_size1a'][0]
        output_size1b = check_data['output_size1b'][0]
        output_size1c = check_data['output_size1c'][0]

        w,b,svs = q2_train_svm_primal_separable(X,Y);
        clocal = 0
        if np.linalg.norm(np.shape(w) - output_size1a):
            print('q2_train_svm_primal_separable: w, WRONG OUTPUT:', np.shape(w), ', expected:', output_size1a, '\n')
            c += 1
            clocal += 1
        if np.linalg.norm(np.shape(b) - output_size1b):
            print('q2_train_svm_primal_separable: b, WRONG OUTPUT:', np.shape(b), ', expected:', output_size1b, '\n')
            c += 1
            clocal += 1
        if type(b) != float:
            print('q2_train_svm_primal_separable: b, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(svs) - output_size1c):
            print('q2_train_svm_primal_separable: svs, WRONG OUTPUT:', np.shape(svs),', expected:', output_size1c, '\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q2_train_svm_primal_separable successfully passed output size test!")


    except:
        c += 1
        print("cannot execute one of the functions required for q2c\n")

    if c != 0:
        print("Exiting q2c due to error\n")
        sys.exit()


def main():
    checking2c()


if __name__ == '__main__':
    main()

