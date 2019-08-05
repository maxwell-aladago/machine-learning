import sys
import numpy as np
import scipy.io as spio
from q4_features import q4_features
from q4_mse import q4_mse
from q4_train import q4_train
from q4_predict import q4_predict
from q4_cross_validation_error import q4_cross_validation_error



def checking4b():
    check_data = spio.loadmat('q4b_checker.mat',squeeze_me=False)

    c = 0
    try:
        mode1 = check_data['mode1']
        mode2 = check_data['mode2']
        mode3 = check_data['mode3']
        mode4 = check_data['mode4']
        lambda1 = check_data['lambda1']
        lambda2 = check_data['lambda2'].flatten()
        pred_Y = check_data['pred_Y']
        correct_Y = check_data['correct_Y']
        theta = check_data['theta']
        X1 = check_data['X1']
        X2 = check_data['X2']
        X3 = check_data['X3']
        X4 = check_data['X4']
        Y2 = check_data['Y2']
        Y3 = check_data['Y3']
        N = int(check_data['N'])
        output_size1 = check_data['output_size1'][0]
        output_size2 = check_data['output_size2'][0][0]
        output_size3 = check_data['output_size3'][0]
        output_size4 = check_data['output_size4'][0]
        output_size5 = check_data['output_size5'][0][1]

        B = q4_features(X1,mode1)
        if np.linalg.norm(np.shape(B) - output_size1):
            print('q4_features, WRONG OUTPUT:', np.shape(B), ', expected:',output_size1,'\n')
            c+=1
        else:
            print("q4_features successfully passed output size test!")

        err = q4_mse(pred_Y, correct_Y)
        if np.linalg.norm(np.shape(err) - output_size2):
            print('q4_mse, WRONG OUTPUT:', np.shape(err), ', expected:',output_size2,'\n')
            c+=1
        else:
            print("q4_mse successfully passed output size test!")

        theta = q4_train(X2,Y2,lambda1,mode2)
        if np.linalg.norm(np.shape(theta) - output_size3):
            print('q4_train, WRONG OUTPUT:', np.shape(theta), ', expected:',output_size3,'\n')
            c+=1
        else:
            print("q4_train successfully passed output size test!")

        pred_Y = q4_predict(theta, X3,mode3)
        if np.linalg.norm(np.shape(pred_Y) - output_size4):
            print('q4_predict, WRONG OUTPUT:', np.shape(pred_Y), ', expected:',output_size4,'\n')
            c+=1
        else:
            print("q4_predict successfully passed output size test!")

        error = q4_cross_validation_error(X4,Y3,lambda2,mode4,N)
        if np.linalg.norm(np.shape(error) - output_size5):
            print('q4_cross_validation_error, WRONG OUTPUT:', np.shape(error), ', expected:',output_size5,'\n')
            c+=1
        else:
            print("q4_cross_validation_error successfully passed output size test! \n")

    except:
        c+=1
        print("cannot execute one of the functions required for q4b\n")

    if c != 0:
        print("Exiting q4b due to error\n")
        sys.exit()



def main():
    checking4b()


if __name__ == '__main__':
    main()

