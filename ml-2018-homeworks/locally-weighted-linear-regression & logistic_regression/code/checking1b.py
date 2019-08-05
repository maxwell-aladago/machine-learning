import sys
import numpy as np
import scipy.io as spio
from q1_W import q1_W
from q1_train import q1_train
from q1_predict import q1_predict
from q1_test_error import q1_test_error
from q1_features import q1_features
from q1_mse import q1_mse

def checking1b():
    check_data = spio.loadmat('q1b_checker.mat', squeeze_me=True)

    c = 0
    try:
        xtest1 = check_data['xtest1']
        xtest2 = check_data['xtest2']
        xtest3 = check_data['xtest3']
        xtest2 = check_data['xtest2']
        tau1 = check_data['tau1']
        tau2 = check_data['tau2']
        tau3 = check_data['tau3']
        tau4 = check_data['tau4']
        Xtrain = check_data['Xtrain']
        Ytrain = check_data['Ytrain']
        Xtest = check_data['Xtest']
        Ytest = check_data['Ytest']
        pred_Y = check_data['pred_Y']
        correct_Y = check_data['correct_Y']
        mode = check_data['mode']
        X1 = check_data['X1']
        X2 = check_data['X2']
        X3 = check_data['X3']
        X4 = check_data['X4']
        Y1 = check_data['Y1']
        Y2 = check_data['Y2']
        output_size1 = check_data['output_size1']
        output_size2 = check_data['output_size2'][0]
        output_size3 = check_data['output_size3'][0]
        output_size4 = check_data['output_size4'][1]
        output_size5 = check_data['output_size5']
        output_size6 = check_data['output_size6'][0]

        W = q1_W(X1, xtest1, tau1)
        if np.linalg.norm(np.shape(W) - output_size1):
            print('q1_W, WRONG OUTPUT:', np.shape(W), ', expected:', output_size1, '\n')
            c += 1
        else:
            print("q1_W successfully passed output size test!")

        theta = q1_train(X2, Y1, xtest2, tau2);
        if np.linalg.norm(np.shape(theta) - output_size2):
            print('q1_train, WRONG OUTPUT:', np.shape(theta), ', expected:', output_size2, '\n')
            c += 1
        else:
            print("q1_train successfully passed output size test!")

        pred_y = q1_predict(X3, Y2, xtest3,tau3 );
        if np.linalg.norm(np.shape(pred_y) - output_size3):
            print('q1_predict, WRONG OUTPUT:', np.shape(theta), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q1_predict successfully passed output size test!")

        error = q1_test_error(Xtrain, Ytrain, Xtest, Ytest, tau4);

        if np.linalg.norm(np.shape(error) - output_size4):
            print('q1_test_error, WRONG OUTPUT:', np.shape(error), ', expected:', output_size5, '\n')
            c += 1
        else:
            print("q1_test_error successfully passed output size test!")

        B = q1_features(X4,mode);

        if np.linalg.norm(np.shape(B) - output_size5):
            print('q1_features, WRONG OUTPUT:', np.shape(B), ', expected:', output_size5, '\n')
            c += 1
        else:
            print("q1_features successfully passed output size test!")

        err = q1_mse(pred_Y,correct_Y);
        if np.linalg.norm(np.shape(err) - output_size6):
            print('q1_mse, WRONG OUTPUT:', np.shape(err), ', expected:', output_size6, '\n')
            c += 1
        else:
            print("q1_mse successfully passed output size test! \n")

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


