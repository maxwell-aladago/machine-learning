import sys
import numpy as np
import scipy.io as spio
from q3_predict import q3_predict
from q3_leave_one_out_error import q3_leave_one_out_error

def checking3b():
    check_data = spio.loadmat('q3b_checker.mat', squeeze_me=True)

    c = 0
    try:
        Y = check_data['Y']
        pred_Y = check_data['pred_Y']
        Xtrain1 = check_data['Xtrain1']
        Ytrain1 = check_data['Ytrain1']
        Xtrain2 = check_data['Xtrain2']
        Ytrain2 = check_data['Ytrain2']
        xtest = check_data['xtest']
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

        errors = q3_leave_one_out_error(Xtrain2,Ytrain2,k2);
        if np.linalg.norm(np.shape(errors) - output_size3):
            print('q3_leave_one_out_error, WRONG OUTPUT:', np.shape(errors), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q3_leave_one_out_error successfully passed output size test!")

    except:
        c += 1
        print("cannot execute one of the functions required for q3b\n")

    if c != 0:
        print("Exiting q3b due to error\n")
        sys.exit()


def main():
    checking3b()


if __name__ == '__main__':
    main()

