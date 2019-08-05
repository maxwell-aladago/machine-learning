import scipy.io as spio
from q1_error import q1_error
from q1_nb_train import q1_nb_train
from q1_nb_predict import q1_nb_predict


# load the spam dataset
S = spio.loadmat('spamdata.mat', squeeze_me=True)
X = S['trainsetX']
Y = S['trainsetY']
Xtest = S['testsetX']
Ytest = S['testsetY']


# train a Naive Bayes model
[phi_y0, phi_y1, phi_prior] = q1_nb_train(X, Y)

pred_Y = q1_nb_predict(X, phi_y0, phi_y1, phi_prior)
train_error = q1_error(Y, pred_Y)

pred_Y = q1_nb_predict(Xtest, phi_y0, phi_y1, phi_prior)
test_error = q1_error(Ytest, pred_Y)

f = open('q1a_output.txt','w') 

print('Training error: ' + str(100*train_error) + '%')
f.write('Training error: ' + str(100*train_error) + '%\n')

print('Test error: ' + str(100*test_error) + '%')
f.write('Test error: ' + str(100*test_error) + '%\n')

f.close()


