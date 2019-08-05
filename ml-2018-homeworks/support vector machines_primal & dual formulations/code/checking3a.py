import sys
import numpy as np
import scipy.io as spio
from q3_predict import q3_predict
from q3_test_error import q3_test_error

def checking3a():
    check_data = spio.loadmat('q3a_checker.mat', squeeze_me=True)

    c = 0
    try:
        Y = check_data['Y']
        pred_Y = check_data['pred_Y']
        Xtrain1 = check_data['Xtrain1']
        Ytrain1 = check_data['Ytrain1']
        Xtrain2 = check_data['Xtrain2']
        Ytrain2 = check_data['Ytrain2']
        xtest = check_data['xtest']
        Xtest = check_data['Xtest']
        Ytest = check_data['Ytest']
        k1 = check_data['k1']
        k2 = check_data['k2']

        output_size2 = check_data['output_size2'][0]
        output_size3 = check_data['output_size3'][0]

        pred_y = q3_predict(Xtrain1, Ytrain1, xtest, k1);
        clocal=0
        if np.linalg.norm(np.shape(pred_y) - output_size2):
            print('q3_predict: pred_y, WRONG OUTPUT:', np.shape(pred_y), ', expected:', output_size2, '\n')
            c += 1
            clocal += 1

        if type(pred_y) != np.uint8:
            print('q3_predict: pred_y, WRONG OUTPUT: expected a uint8\n')
            c += 1
            clocal += 1
        if clocal == 0:
            print("q3_predict successfully passed output size test!")

        errors = q3_test_error(Xtrain2,Ytrain2,Xtest,Ytest,k2);
        if np.linalg.norm(np.shape(errors) - output_size3):
            print('q3_test_error, WRONG OUTPUT:', np.shape(errors), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q3_test_error successfully passed output size test!")

    except:
        c += 1
        print("cannot execute one of the functions required for q3a\n")

    if c != 0:
        print("Exiting q3a due to error\n")
        sys.exit()


def main():
    checking3a()


if __name__ == '__main__':
    main()

