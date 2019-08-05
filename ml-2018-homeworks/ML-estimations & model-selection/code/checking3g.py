import sys
import numpy as np
import scipy.io as spio
from q3_likelihood import q3_likelihood
from q3_posterior import q3_posterior
from q3_prior import q3_prior


def checking3g():
    check_data = spio.loadmat('q3g_checker.mat',squeeze_me=False)
    c = 0
    try:

        H = check_data['H']
        Z = check_data['Z']
        a = check_data['a']
        mu = check_data['mu']
        m = check_data['m']
        
        output_size1 = check_data['output_size1'][0]
        output_size2 = check_data['output_size2'][0]
        output_size3 = check_data['output_size3'][0]

        
        lik = q3_likelihood(mu, m, H)

        if np.linalg.norm(np.shape(lik) - output_size1):
            print('q3_likelihood, WRONG OUTPUT:', np.shape(lik), ', expected:',output_size1,'\n')
            c+=1
        else:
            print("q3_likelihood successfully passed output size test!")            
        
        prior = q3_prior(mu, a, Z);
        if np.linalg.norm(np.shape(prior) - output_size2):
            print('q3_prior, WRONG OUTPUT:', np.shape(prior), ', expected:',output_size2,'\n')
            c+=1
        else:
             print("q3_prior successfully passed output size test!")             
           
        prob = q3_posterior(mu, m, H, a, Z);       
        if np.linalg.norm(np.shape(prob) - output_size3):
            print('q3_posterior, WRONG OUTPUT:', np.shape(prob), ', expected:',output_size3,'\n')
            c+=1
        else:
           print("q3_posterior successfully passed output size test! \n")        
           
    except:
        c+=1
        print("cannot execute one of the functions required for q3g\n")

    if c != 0:
        print("Exiting q3g due to error\n")        
        sys.exit()

def main():
    checking3g()


if __name__ == '__main__':
    main()
