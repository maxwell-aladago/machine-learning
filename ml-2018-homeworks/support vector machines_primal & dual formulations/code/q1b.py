import scipy.io as spio
from q1_nb_train import q1_nb_train
from q1_top_words import q1_top_words

# load the spam dataset
S = spio.loadmat('spamdata.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']

k = 6;

feature_names = open('spambase_names.txt').read().splitlines()

[phi_y0, phi_y1, phi_prior] = q1_nb_train(X, Y)

word_idx = q1_top_words(phi_y0, phi_y1, phi_prior, k)

f = open('q1b_output.txt','w') 

for i in range(word_idx.shape[0]):
    print('Top ' + str(k) + ' words for class ' + str(i) + ': ' + feature_names[word_idx[i,0]], end='')
    f.write('Top ' + str(k) + ' words for class ' + str(i) + ': ' + feature_names[word_idx[i,0]]) 
    
    for j in range(1,k):
        print(', ' + feature_names[word_idx[i,j]], end='')
        f.write(', ' + feature_names[word_idx[i,j]])
        
    print('.')
    f.write('.\n')

f.close()