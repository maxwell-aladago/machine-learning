import sys
import numpy as np
import scipy.io as spio
from q2_kernel import q2_kernel
from q2_calculate_bias_svm_dual import q2_calculate_bias_svm_dual
from q2_train_svm_dual import q2_train_svm_dual
from q2_predict_svm_dual import q2_predict_svm_dual

def checking2h():
    check_data = spio.loadmat('q2h_checker.mat', squeeze_me=True)

    c = 0
    try:
        X1 = check_data['X1']
        X2 = check_data['X2']
        Y1 = check_data['Y1']
        Y2 = check_data['Y2']
        mode1 = check_data['mode1']
        mode2 = check_data['mode2']
        mode3 = check_data['mode3']
        mode4 = check_data['mode4']
        x1 = check_data['x1']
        x2 = check_data['x2']
        svs1 = check_data['svs1']
        svs2 = check_data['svs2']
        alphas1 = check_data['alphas1']
        alphas2 = check_data['alphas2']
        C = check_data['C']
        xtest = check_data['xtest']
        Xtrain = check_data['Xtrain']
        Ytrain = check_data['Ytrain']
        bias = check_data['bias']
        output_size1 = check_data['output_size1'][0]
        output_size2 = check_data['output_size2'][0]
        output_size3a = check_data['output_size3a'][0]
        output_size3b = check_data['output_size3b'][0]
        output_size4a = check_data['output_size4a'][0]
        output_size4b = check_data['output_size4b'][0]

        d = q2_kernel(x1,x2,mode1);
        clocal = 0
        if np.linalg.norm(np.shape(d) - output_size1):
            print('q2_kernel: WRONG OUTPUT:', np.shape(d), ', expected:', output_size1, '\n')
            c += 1
            clocal += 1
        if type(d) != float:
            print('q2_kernel: d, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if clocal==0:
            print("q2_kernel successfully passed output size test!")

        bias = q2_calculate_bias_svm_dual(X1,Y1,svs1,alphas1,mode2);
        clocal = 0
        if np.linalg.norm(np.shape(bias) - output_size2):
            print('q2_calculate_bias_svm_dual: WRONG OUTPUT:', np.shape(bias), ', expected:', output_size2, '\n')
            c += 1
            clocal += 1
        if type(bias) != float:
            print('q2_calculate_bias_svm_dual: b, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if clocal==0:
            print("q2_calculate_bias_svm_dual successfully passed output size test!")


        svs, alphas = q2_train_svm_dual(X2, Y2, C, mode3);
        if np.linalg.norm(np.shape(svs)-output_size3a):
            print('q2_train_svm_dual: svs, WRONG OUTPUT:', np.shape(svs), ', expected:', output_size3a, '\n')
            c += 1

        if np.linalg.norm(np.shape(alphas) - output_size3b):
            print('q2_train_svm_dual: alphas, WRONG OUTPUT:', np.shape(alphas), ', expected:', output_size3b, '\n')
            c += 1

        else:
            print("q2_train_svm_dual successfully passed output size test!")

        label, s = q2_predict_svm_dual(xtest, Xtrain, Ytrain, svs2, alphas2, bias,mode4);
        clocal = 0
        if np.linalg.norm(np.shape(label) - output_size4a):
            print('q2_predict_svm_dual: label, WRONG OUTPUT:', np.shape(label), ', expected:', output_size4a, '\n')
            c += 1
            clocal += 1
        if type(label) != np.int16:
            print('q2_predict_svm_dual: label, WRONG OUTPUT: expected a int\n')
            c += 1
            clocal += 1
        if np.linalg.norm(np.shape(s) - output_size4b):
            print('q2_predict_svm_dual: s, WRONG OUTPUT:', np.shape(s), ', expected:', output_size4b, '\n')
            c += 1
            clocal += 1
        if type(s) != float:
            print('q2_predict_svm_dual: s, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if clocal == 0:
            print("q2_predict_svm_dual successfully passed output size test!")
    except :
        c += 1
        print("cannot execute one of the functions required for q2h\n")

    if c != 0:
        print("Exiting q2h due to error\n")
        sys.exit()


def main():
    checking2h()


if __name__ == '__main__':
    main()

