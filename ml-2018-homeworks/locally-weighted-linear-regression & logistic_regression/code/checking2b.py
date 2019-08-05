import sys
import numpy as np
import scipy.io as spio
from q2_initialize import q2_initialize
from q2_predict import q2_predict
from q2_loglik import q2_loglik
from q2_gradient import q2_gradient
from q2_train import q2_train
from q2_error import q2_error
from q2_train_line_search import q2_train_line_search

def checking2b():
    check_data = spio.loadmat('q2b_checker.mat', squeeze_me=True)

    c = 0
    try:
        Xtrain1 = check_data['Xtrain1']
        Xtrain2 = check_data['Xtrain2']
        Xtrain3 = check_data['Xtrain3']
        Xtrain4 = check_data['Xtrain4']
        Xtrain5 = check_data['Xtrain5']
        Ytrain1 = check_data['Ytrain1']
        Ytrain2 = check_data['Ytrain2']
        Ytrain3 = check_data['Ytrain3']
        Ytrain4 = check_data['Ytrain4']
        Ytrain5 = check_data['Ytrain5']
        opt = check_data['opt']
        X = check_data['X']
        theta1 = check_data['theta1']
        theta2 = check_data['theta2']
        theta3 = check_data['theta3']
        theta_init = check_data['theta_init']
        alpha = check_data['alpha']
        tol = check_data['tol']
        theta_init1 = check_data['theta_init1']
        alpha1 = check_data['alpha1']
        tol1 = check_data['tol1']
        Y = check_data['Y']
        pred_Y = check_data['pred_Y']
        output_size1 = check_data['output_size1'][0]
        output_size2 = check_data['output_size2'][0]
        output_size3 = check_data['output_size3'][0]
        output_size4 = check_data['output_size4'][0]
        output_size5 = check_data['output_size5'][0]
        output_size6a = check_data['output_size6a'][0]
        output_size6b = check_data['output_size6b'][0]
        output_size6c = check_data['output_size6c'][0]
        output_size7c = check_data['output_size7c'][0]

        theta = q2_initialize(Xtrain1, Ytrain1, opt)
        if np.linalg.norm(np.shape(theta) - output_size1):
            print('q2_initialize, WRONG OUTPUT:', np.shape(theta), ', expected:', output_size1, '\n')
            c += 1
        else:
            print("q2_initialize successfully passed output size test!")

        pred_Y, prob_Y = q2_predict(X, theta1);
        clocal = 0
        if np.linalg.norm(np.shape(pred_Y) - output_size2):
            print('q2_predict: pred_Y, WRONG OUTPUT:', np.shape(pred_Y), ', expected:', output_size2, '\n')
            c += 1
            clocal +=1
        if np.linalg.norm(np.shape(prob_Y) - output_size2):
            print('q2_predict: prob_Y, WRONG OUTPUT:', np.shape(prob_Y), ', expected:', output_size2, '\n')
            c += 1
            clocal +=1
        if clocal == 0:
            print("q2_predict successfully passed output size test!")

        lik = q2_loglik(Xtrain2,Ytrain2,theta2);
        if np.linalg.norm(np.shape(lik) - output_size3):
            print('q2_loglik, WRONG OUTPUT:', np.shape(theta), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q2_loglik successfully passed output size test!")

        grad = q2_gradient(Xtrain3,Ytrain3,theta3);

        if np.linalg.norm(np.shape(grad) - output_size5):
            print('q2_gradient, WRONG OUTPUT:', np.shape(grad), ', expected:', output_size5, '\n')
            c += 1
        else:
            print("q2_gradient successfully passed output size test!")

        theta, n_iter, loglik = q2_train(Xtrain4, Ytrain4, theta_init, alpha, tol);

        clocal = 0
        if np.linalg.norm(np.shape(theta) - output_size6a):
            print('q2_train: theta, WRONG OUTPUT:', np.shape(theta), ', expected:', output_size6a, '\n')
            c += 1
            clocal +=1
        if np.linalg.norm(np.shape(n_iter) - output_size6b):
            print('q2_train: n_iter, WRONG OUTPUT:', np.shape(n_iter), ', expected:', output_size6b, '\n')
            c += 1
            clocal +=1
        if type(n_iter) != int :
            print('q2_train: n_iter, WRONG OUTPUT: expected an int\n')
            c += 1
            clocal +=1
        if np.linalg.norm(np.shape(loglik) - output_size6c):
            print('q2_train: loglik, WRONG OUTPUT:', np.shape(loglik), ', expected:', output_size6c, '\n')
            c += 1
            clocal +=1
        if clocal == 0:
            print("q2_train successfully passed output size test!")

        error = q2_error(Y,pred_Y);
        if np.linalg.norm(np.shape(error) - output_size3) and type(error) != float:
            print('q2_error, WRONG OUTPUT:', np.shape(error), ', expected:', output_size3, '\n')
            c += 1
        else:
            print("q2_error successfully passed output size test!")

        theta,n_iter,loglik = q2_train_line_search(Xtrain5, Ytrain5,theta_init1,alpha1,tol1);

        clocal = 0
        if np.linalg.norm(np.shape(theta) - output_size6a):
            print('q2_train_line_search: theta, WRONG OUTPUT:', np.shape(theta), ', expected:', output_size6a, '\n')
            c += 1
            clocal +=1
        if np.linalg.norm(np.shape(n_iter) - output_size6b):
            print('q2_train_line_search: n_iter, WRONG OUTPUT:', np.shape(n_iter), ', expected:', output_size6b, '\n')
            c += 1
            clocal +=1
        if type(n_iter) != int :
            print('q2_train: n_iter, WRONG OUTPUT: expected an int\n')
            c += 1
            clocal +=1
        if np.linalg.norm(np.shape(loglik) - output_size7c):
            print('q2_train_line_search: loglik, WRONG OUTPUT:', np.shape(loglik), ', expected:', output_size7c, '\n')
            c += 1
            clocal +=1
        if clocal == 0:
            print("q2_train_line_search successfully passed output size test! \n")


    except:
        c += 1
        print("cannot execute one of the functions required for q2b\n")

    if c != 0:
        print("Exiting q2b due to error\n")
        sys.exit()


def main():
    checking2b()


if __name__ == '__main__':
    main()


