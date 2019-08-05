import sys
import numpy as np
import scipy.io as spio
from q1_dist2 import q1_dist2
from q1_kmeans_select_seeds import q1_kmeans_select_seeds
from q1_kmeans import q1_kmeans
from q1_reconstructimgfromVQ import q1_reconstructimgfromVQ
from q1_gmminit import q1_gmminit
from q1_logprobgauss import q1_logprobgauss
from q1_GM_Expectation import q1_GM_Expectation
from q1_GM_Maximization import q1_GM_Maximization
from q1_GaussianMixture import q1_GaussianMixture


def checking1b():
    check_data = spio.loadmat('q1b_checker.mat', squeeze_me=True)

    c = 0
    try:
        X1 = check_data['X1']
        X2 = check_data['X2']
        Xa = check_data['Xa']
        Xb = np.float_(check_data['Xb'])
        Xc = np.float_(check_data['Xc'])
        Xd = check_data['Xd']
        Xe = check_data['Xe']
        Xf = check_data['Xf']
        x = check_data['x']
        Ka = check_data['Ka']
        Kb = check_data['Kb']
        Kc = check_data['Kc']
        seeds_idx = check_data['seeds_idx']-1
        mode = check_data['mode']
        prototypes = np.transpose(check_data['prototypes'])
        tilesize = check_data['tilesize']
        tileidx = check_data['tileidx']-1
        num_x_tiles = check_data['num_x_tiles']
        num_y_tiles = check_data['num_y_tiles']
        mus = np.transpose(check_data['mus'])
        sigmas = np.moveaxis(check_data['sigmas'], 2, 0)        
        priors = check_data['priors']
        labels = check_data['labels']-1
        mu = check_data['mu']
        sigma = check_data['sigma']        
        prob_c = np.transpose(check_data['prob_c'])
        mus_init = np.transpose(check_data['mus_init'])
        sigmas_init = np.moveaxis(check_data['sigmas_init'], 2, 0)        
        priors_init = check_data['priors_init']
        num_iterations = check_data['num_iterations']

        output_size1 = check_data['output_size1']
        output_size2 = check_data['output_size2'][0]
        output_size3a = check_data['output_size3a'][0]
        output_size3b = check_data['output_size3b']
        output_size3c = check_data['output_size3c'][0]
        output_size4 = check_data['output_size4']
        output_size5a = check_data['output_size5a']
        output_size5b = check_data['output_size5b']
        output_size5c = check_data['output_size5c']
        output_size6 = check_data['output_size6'][0]
        output_size7a = check_data['output_size7a']
        output_size7b = check_data['output_size7b'][0]
        output_size7c = check_data['output_size7c'][0]
        output_size8a = check_data['output_size8a']
        output_size8b = check_data['output_size8b']
        output_size8c = check_data['output_size8c'][0]
        output_size8d = check_data['output_size8d'][0]
        output_size8e = check_data['output_size8e'][0]
        output_size9a = check_data['output_size9a']
        output_size9b = check_data['output_size9b']
        output_size9c = check_data['output_size9c'][0]
        output_size9d = check_data['output_size9d'][0]
        output_size9e = check_data['output_size9e'][0]
        output_size9f = check_data['output_size9f'][0]
        output_size9g = check_data['output_size9g'][0]

        D = q1_dist2(X1,np.reshape(X2, (1,X2.size)))
        if np.linalg.norm(np.shape(D) - output_size1):
            print('q1_dist2, WRONG OUTPUT:', np.shape(D), ', expected:', output_size1, '\n')
            c += 1
        else:
            print("q1_dist2 successfully passed output size test!")

        seeds_id = q1_kmeans_select_seeds(Xa,Ka,mode)
        if np.linalg.norm(np.shape(seeds_id) - output_size2):
            print('q1_kmeans_select_seeds, WRONG OUTPUT:', np.shape(seeds_id), ', expected:', output_size2, '\n')
            c += 1
        else:
            print("q1_kmeans_select_seeds successfully passed output size test!")

        labels1,means,distortions = q1_kmeans(Xb,Kb,seeds_idx)
        clocal = 0
        if np.linalg.norm(np.shape(labels1) - output_size3a):
            print('q1_kmeans: labels, WRONG OUTPUT:', np.shape(labels1), ', expected:', output_size3a, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(means) - output_size3b):
            print('q1_kmeans: means, WRONG OUTPUT:', np.shape(means), ', expected:', output_size3b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(distortions) - output_size3c):
            print('q1_kmeans: distortions, WRONG OUTPUT:', np.shape(distortions), ', expected:', output_size3c, '\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q1_kmeans successfully passed output size test!")

        recI = q1_reconstructimgfromVQ(prototypes,tilesize,tileidx,num_x_tiles,num_y_tiles)
        if np.linalg.norm(np.shape(recI) - output_size4):
            print('q1_reconstructimgfromVQ, WRONG OUTPUT:', np.shape(recI), ', expected:', output_size4, '\n')
            c += 1
        else:
            print("q1_reconstructimgfromVQ successfully passed output size test!")

        mus1, sigmas1, priors1 = q1_gmminit(Xc,Kc,labels)
        clocal = 0
        if np.linalg.norm(np.shape(mus1) - output_size5a):
            print('q1_gmminit: mus, WRONG OUTPUT:', np.shape(mus1), ', expected:', output_size5a, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(sigmas1) - output_size5b):
            print('q1_gmminit: sigmas, WRONG OUTPUT:', np.shape(sigmas1), ', expected:', output_size5b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(priors1) - output_size5c):
            print('q1_gmminit: priors, WRONG OUTPUT:', np.shape(priors1), ', expected:', output_size5c, '\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q1_gmminit successfully passed output size test!")

        logprob = q1_logprobgauss(x,mu,sigma)
        if np.linalg.norm(np.shape(logprob) - output_size6):
            print('q1_logprobgauss, WRONG OUTPUT:', np.shape(logprob), ', expected:', output_size6, '\n')
            c += 1

        else:
            print("q1_logprobgauss successfully passed output size test!")

        prob_c1, free_energy_e, likelihood_e = q1_GM_Expectation(Xd,mus,sigmas,priors)
        clocal = 0
        if np.linalg.norm(np.shape(prob_c1) - output_size7a):
            print('q1_GM_Expectation: prob_c, WRONG OUTPUT:', np.shape(prob_c1), ', expected:', output_size7a, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(free_energy_e) - output_size7b):
            print('q1_GM_Expectation: free_energy_e, WRONG OUTPUT:', np.shape(free_energy_e), ', expected:', output_size7b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(likelihood_e) - output_size7c):
            print('q1_GM_Expectation: likelihood_e, WRONG OUTPUT:', np.shape(likelihood_e), ', expected:', output_size7c, '\n')
            c += 1
            clocal += 1

        if (type(free_energy_e) != float) and (type(free_energy_e) != np.float64):
            print('q1_GM_Expectation: free_energy_e, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if (type(likelihood_e) != float) and (type(likelihood_e) != np.float64):
            print('q1_GM_Expectation: likelihood_e, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q1_GM_Expectation successfully passed output size test!")

        mus1, sigmas, priors, free_energy_m, likelihood_m = q1_GM_Maximization(Xe,prob_c)
        clocal = 0
        if np.linalg.norm(np.shape(mus1) - output_size8a):
            print('q1_GM_Maximization: mus, WRONG OUTPUT:', np.shape(mus1), ', expected:', output_size8a, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(sigmas) - output_size8b):
            print('q1_GM_Maximization: sigmas, WRONG OUTPUT:', np.shape(sigmas), ', expected:', output_size8b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(priors) - output_size8c):
            print('q1_GM_Maximization: priors, WRONG OUTPUT:', np.shape(priors), ', expected:', output_size8c, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(free_energy_m) - output_size8d):
            print('q1_GM_Maximization: free_energy_m, WRONG OUTPUT:', np.shape(free_energy_m), ', expected:', output_size8d, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(likelihood_m) - output_size8e):
            print('q1_GM_Maximization: likelihood_m, WRONG OUTPUT:', np.shape(likelihood_m), ', expected:', output_size8e, '\n')
            c += 1
            clocal += 1
        if (type(free_energy_m) != float) and (type(free_energy_m) != np.float64):
            print('q1_GM_Maximization: free_energy_m, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1
        if (type(likelihood_m) != float) and (type(likelihood_m) != np.float64):
            print('q1_GM_Maximization: likelihood_m, WRONG OUTPUT: expected a float\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q1_GM_Maximization successfully passed output size test!")

        mus, sigmas, priors,likelihood_e,free_energy_e,likelihood_m,free_energy_m = q1_GaussianMixture(Xf,mus,sigmas_init,priors_init,num_iterations)
        clocal = 0
        if np.linalg.norm(np.shape(mus) - output_size9a):
            print('q1_GaussianMixture: mus, WRONG OUTPUT:', np.shape(mus), ', expected:', output_size9a, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(sigmas) - output_size9b):
            print('q1_GaussianMixture: sigmas, WRONG OUTPUT:', np.shape(sigmas), ', expected:', output_size9b, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(priors) - output_size9c):
            print('q1_GaussianMixture: priors, WRONG OUTPUT:', np.shape(priors), ', expected:', output_size9c, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(likelihood_e) - output_size9d):
            print('q1_GaussianMixture: likelihood_e, WRONG OUTPUT:', np.shape(likelihood_e), ', expected:', output_size9d, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(free_energy_e) - output_size9e):
            print('q1_GaussianMixture: free_energy_e, WRONG OUTPUT:', np.shape(free_energy_e), ', expected:', output_size9e, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(likelihood_m) - output_size9f):
            print('q1_GaussianMixture: likelihood_m, WRONG OUTPUT:', np.shape(likelihood_m), ', expected:',
                  output_size9f, '\n')
            c += 1
            clocal += 1

        if np.linalg.norm(np.shape(free_energy_m) - output_size9g):
            print('q1_GaussianMixture: free_energy_m, WRONG OUTPUT:', np.shape(free_energy_m), ', expected:',
                  output_size9g, '\n')
            c += 1
            clocal += 1

        if clocal == 0:
            print("q1_GaussianMixture successfully passed output size test!")

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

