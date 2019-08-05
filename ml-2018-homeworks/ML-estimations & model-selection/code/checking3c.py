import numpy as np
import scipy.io as spio
import sys
from q3_likelihood import q3_likelihood


def checking3c():
    check_data = spio.loadmat('q3c_checker.mat',squeeze_me = False)

    c = 0
    try:
        
        H = check_data['H']
        mu = check_data['mu']
        m = check_data['m']
        output_size1 = check_data['output_size1'][0]
        
        lik = q3_likelihood(mu, m, H)

        if np.linalg.norm(np.shape(lik) - output_size1):
            print('q3_likelihood, WRONG OUTPUT:', np.shape(lik), ', expected:', output_size1, '\n')
            c+=1
        else:
            print("q3_likelihood, successfully passed output size test! \n")            
            
    except:
        c+=1
        print("cannot execute q3_likelihood \n")

    if c != 0:
        print("Exiting q3c due to error\n")
        sys.exit()


def main():
    checking3c()


if __name__ == '__main__':
    main()
