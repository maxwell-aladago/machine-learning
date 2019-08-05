import sys
import numpy as np
import scipy.io as spio
from q2_train_svm_primal import q2_train_svm_primal
from q2_predict_svm_primal import q2_predict_svm_primal

def checking2e():
    check_data = spio.loadmat('q2e_checker.mat', squeeze_me=True)

    c = 0
    try:
        X1 = check_data['X1']
        X2 = check_data['X2']
        Y = check_data['Y']
        C = check_data['C']
        w1 = check_data['w']
        b1 = check_data['b']
        output_size1a = check_data['output_size1a'][0]
        output_size1b = check_data['output_size1b'][0]
        output_size1c = check_data['output_size1c'][0]
        output_size2a = check_data['output_size2a'][0]
        output_size2b = check_data['output_size2b'][0]

        w,b,svs = q2_train_svm_primal(X1,Y,C);
        clocal = 0
        if np.linalg.norm(np.shape(w) - output_size1a):
            print('q2_train_svm_primal: w, WRONG OUTPUT:', np.shape(w), ', expected:', output_size1a, '\n')
            c += 1
            clocal += 1
        if np.linalg.norm(np.shape(b) - output_size1b):
            print('q2_train_svm_primal: b, WRONG OUTPUT:', np.shape(b), ', expected:', output_size1b, '\n')
            c += 1
            clocal += 1
        if type(b) != float:
            print('q2_train_svm_primal: b, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(svs) - output_size1c):
            print('q2_train_svm_primal: svs, WRONG OUTPUT:', np.shape(svs),', expected:', output_size1c, '\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q2_train_svm_primal successfully passed output size test!")

        label,Y = q2_predict_svm_primal(X2,w1,b1);
        clocal = 0
        if np.linalg.norm(np.shape(label) - output_size2a):
            print('q2_predict_svm_primal: label, WRONG OUTPUT:', np.shape(label), ', expected:', output_size2a, '\n')
            c += 1
            clocal += 1
        if np.linalg.norm(np.shape(Y) - output_size2b):
            print('q2_predict_svm_primal: Y, WRONG OUTPUT:', np.shape(Y), ', expected:', output_size2b, '\n')
            c += 1
            clocal += 1
        if clocal == 0:
            print("q2_predict_svm_primal successfully passed output size test!")

    except :
        c += 1
        print("cannot execute one of the functions required for q2e\n")

    if c != 0:
        print("Exiting q2e due to error\n")
        sys.exit()


def main():
    checking2e()


if __name__ == '__main__':
    main()

