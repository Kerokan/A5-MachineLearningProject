import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import read
import display
import preprocessing
import knn
import logreg
import mvg

train_data = read.read_data('Train.txt')
test_data  = read.read_data('Test.txt')

#########################
## Normalize Data #######
normalized_data = preprocessing.normalize(train_data)

#########################
## Split Dataset ########
train_set, eval_set = preprocessing.split_train_val(normalized_data)

#########################
## PCA ##################
## Proj, P, s, U = preprocessing.PCA_return_all(train_data, train_data.shape[1] - 1)
## display.contributionPCA(s)

#########################
## KNN ##################
## k = 5
## predicted = knn.knn_all(k, train_set, eval_set)
## display.confusion_matrix(eval_set[:,-1], predicted)

#########################
## Regression Logistic ##
## (DTR, LTR), (DTE, LTE) = (train_set[:,:-1].T, train_set[:,-1].T), (eval_set[:,:-1].T, eval_set[:,-1].T)
## l = 0.01
## values = np.zeros(DTR.shape[0] + 1)
## x, f, d = scipy.optimize.fmin_l_bfgs_b(func=logreg.logreg_obj, x0=values, args=(DTR, LTR, l), approx_grad=True)
## LP = logreg.predict(x, DTE)
## display.confusion_matrix(LTE, LP)

#########################
## MVG ##################
Pc = 0.5
(DTR, LTR), (DTE, LTE) = (train_set[:,:-1].T, train_set[:,-1].T), (eval_set[:,:-1].T, eval_set[:,-1].T)
print(mvg.MVG(DTR, LTR, DTE, LTE, Pc))
print(mvg.Naive_Bayes(DTR, LTR, DTE, LTE, Pc))
print(mvg.Tied_Covariance(DTR, LTR, DTE, LTE, Pc))