import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from q1_splitimgintiles import q1_splitimgintiles
from q1_kmeans_select_seeds import q1_kmeans_select_seeds
from q1_kmeans import q1_kmeans
from q1_reconstructimgfromVQ import q1_reconstructimgfromVQ
from q1_gmminit import q1_gmminit
from q1_GaussianMixture import q1_GaussianMixture
from q1_GM_Expectation import q1_GM_Expectation
from checking1b import checking1b


checking1b()

face = misc.face()
I = np.array(misc.imread('dartmouthhall.png'), dtype=float)

fig = plt.figure(figsize=(10,10))
plt.subplot(1,2,1)
plt.imshow(I, cmap=plt.cm.gray)
plt.title('original image')

# split the image into tiles
tilesize = 8
[num_x_tiles, num_y_tiles, X] =  q1_splitimgintiles(I, tilesize)

# execute Kmeans
K = 4
seeds_idx = q1_kmeans_select_seeds(X, K, 'diverse_set')
[tileidx, prototypes, distortions] = q1_kmeans(X, K, seeds_idx)

# reconstruct the image
recI_kmeans = q1_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles)
ssd = np.sum((I-recI_kmeans)**2)
print('SSD using K-means: ' + str(ssd))

if (np.abs(ssd-distortions[-1])>1e-5):
    print('Error: ssd does not match the value returned in vector distortions!')
    

# initialize the GMM
[mus_init, sigmas_init, priors_init] = q1_gmminit(X, K, tileidx)

# train the GMM
num_iterations = 10
[mus, sigmas, priors, likelihood_e, free_energy_e, likelihood_m, free_energy_m ] = q1_GaussianMixture(X, mus_init, sigmas_init, priors_init, num_iterations)

# calculate the posteriors of the examples, given the trained GMM model
postprob = q1_GM_Expectation(X, mus, sigmas, priors)[0]

# calculate to which gaussians the tiles belong to, and reconstruct the image
labels = np.argmax(postprob, 1)
recI_GMM = q1_reconstructimgfromVQ(mus, tilesize, labels, num_x_tiles, num_y_tiles)
ssd_GMM = np.sum((I-recI_GMM)**2)
print('SSD using GMM: ' + str(ssd_GMM))

# visualize the reconstructed image
plt.subplot(1,2,2)
plt.imshow(recI_GMM, cmap=plt.cm.gray)
plt.title('ssd GMM = ' + str(ssd_GMM))
# save the plot (Note: do not remove this line of code)
fig.savefig('q1b.png', dpi = 300)
plt.show()


# visualize the Expectation plots
fig = plt.figure(figsize=(10,10))
plt.plot(likelihood_e, '-*b', LineWidth=2, MarkerSize=5)
plt.plot(free_energy_e, ':sr', LineWidth=2, MarkerSize=10)
plt.legend(['log likelihood', 'free energy'])
plt.xlabel( 'iteration' )
plt.title('After E step')
plt.grid(True)
fig.savefig('q1b_estep.png', dpi = 300)
plt.show()


# visualize the Maximization plots
fig = plt.figure(figsize=(10,10))
plt.plot(likelihood_m, '-*b', LineWidth=2, MarkerSize=5)
plt.plot(free_energy_m, ':sr', LineWidth=2, MarkerSize=10)
plt.legend(['log likelihood', 'free energy'])
plt.xlabel(' iteration')
plt.title('After M step')
plt.grid(True)
fig.savefig('q1b_mstep.png', dpi = 300)
plt.show()
