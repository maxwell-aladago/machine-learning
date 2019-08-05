import sys
import numpy as np
import scipy.io as spio
from q3_prior import q3_prior


def checking3d():
    check_data = spio.loadmat('q3d_checker.mat',squeeze_me = False)
    c = 0
    try:

        a = check_data['a']
        mu = check_data['mu']
        Z = check_data['Z']
        output_size2 = check_data['output_size2'][0]

        prior = q3_prior(mu, a, Z)
    
        if np.linalg.norm(np.shape(prior) - output_size2):
            print('q3_prior, WRONG OUTPUT:', np.shape(prior), ', expected:',output_size2,'\n')
            c+=1
        else:
            print("q3_prior, successfully passed output size test! \n")            
        
    except:
        c+=1
        print("cannot execute q3_prior\n")

    if c != 0:
        print("Exiting q3d due to error\n")        
        sys.exit()


def main():
    checking3d()


if __name__ == '__main__':
    main()
