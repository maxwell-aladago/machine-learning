import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from q1_splitimgintiles import q1_splitimgintiles
from q1_kmeans_select_seeds import q1_kmeans_select_seeds
from q1_kmeans import q1_kmeans
from q1_reconstructimgfromVQ import q1_reconstructimgfromVQ
from checking1a import checking1a


checking1a()

# K parameters for the Kmeans
Kvalues = np.array([2, 4, 8])

# read and visualize the image
face = misc.face()
I = np.array(misc.imread('dartmouthhall.png'), dtype=float)

fig = plt.figure(figsize=(10,10))
plt.subplot(1,Kvalues.size+1,1)
plt.imshow(I, cmap=plt.cm.gray)
plt.title('original image')

# split the image into tiles
tilesize = 8
[num_x_tiles, num_y_tiles, X] =  q1_splitimgintiles(I, tilesize)

# run K-means for different numbers of centroids
count = 2
for K in Kvalues:
    # execute Kmeans
    init_mode = 'diverse_set'
    #init_mode = 'random'
    
    seeds_idx = q1_kmeans_select_seeds(X, K, init_mode)
    [tileidx, prototypes, distortions] = q1_kmeans(X, K, seeds_idx)
    
    # reconstruct the image from its VQ form, and calculate the SSD.
    recI = q1_reconstructimgfromVQ(prototypes, tilesize, tileidx, num_x_tiles, num_y_tiles);
    ssd = np.sum((I-recI)**2)
    # print(distortions[-1] - ssd)
    if (np.abs(ssd-distortions[-1])>1e-5):
        print('Error: ssd does not match the value returned in vector distortions!')
        break
    
    # visualize the reconstruction
    plt.subplot(1, Kvalues.size+1,count)
    plt.imshow(recI, cmap=plt.cm.gray)
    plt.title('K = ' + str(K) + '\n ssd = ' + str(ssd));
    print('init_mode=' + init_mode + '; SSD using K=' + str(K) + ': ' + str(ssd))
    count = count + 1

fig.savefig('q1a.png', dpi = 300)
plt.show()



