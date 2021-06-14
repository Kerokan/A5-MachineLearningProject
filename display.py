import numpy as np
import matplotlib.pyplot as plt

def plot_hist(dataset):
    data, label = dataset[:,:-1], dataset[:,-1]
    labeled_0 = data[label == 0.0, :]
    labeled_1 = data[label == 1.0, :]

    caracteristics = ['Mean of the integrated profile',
        'Standard deviation of the integrated profile',
        'Excess kurtosis of the integrated profile',
        'Skewness of the integrated profile',
        'Mean of the DM-SNR curve',
        'Standard deviation of the DM-SNR curve',
        'Excess kurtosis of the DM-SNR curve',
        'Skewness of the DM-SNR curve']

    plt.figure()

    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.xlabel(caracteristics[i])
        plt.hist(labeled_0[:, i], bins = 10, density = True, alpha = 0.4, label = 'Negative')
        plt.hist(labeled_1[:, i], bins = 10, density = True, alpha = 0.4, label = 'Positive')
        plt.legend()
    
    plt.rcParams['figure.figsize'] = (20,8)
    plt.show()

def correlation(dataset):
  data = dataset[:,:-1]
  corr = np.corrcoef(data.T)
  print('CORRELATION MATRIX : \n')
  print('Axes||\t   1\t   2\t   3\t   4\t   5\t   6\t   7\t   8')
  print('======================================================================')
  for i in range(corr.shape[0]):
    to_print = ' \033[0m' + str(i + 1) + "  || "
    for j in range(corr.shape[1]):
      if corr[i,j] >= 0.5:
        to_print = to_print + '\t \033[94m' + str(corr[i,j])[0:5]
      elif corr[i,j] <= -0.5:
        to_print = to_print + '\t \033[91m' + str(corr[i,j])[0:5]
      else: 
        to_print = to_print + '\t \033[0m' + str(corr[i,j])[0:5]
    print(to_print)

def confusion_matrix(labels, predictions):

  labels = np.asarray(labels)
  predictions = np.asarray(predictions)
  
  TP, TN, FP, FN = 0, 0, 0, 0
  for i in range(labels.shape[0]):
    if (labels[i] == 1 and predictions[i] == 1):
      TP += 1
    if (labels[i] == 1 and predictions[i] == 0):
      FN += 1
    if (labels[i] == 0 and predictions[i] == 0):
      TN += 1
    if (labels[i] == 0 and predictions[i] == 1):
      FP += 1

  print('=================================')
  print('| Real\Predicted | True | False |')
  print('|      True      |  {}  |   {}  |'.format(TP, FN))
  print('|     False      |  {}  |   {}  |'.format(FP, TN))
  print('=================================')
  print('\nSensitivity: {:.3f}\nSpecificity: {:.3f}\nAccuracy: {:.3f}\n'.format(TP/(TP+FN), TN/(TN+FP), (TN+TP)/(TN+FN+TP+FP)))

def contributionPCA(s):
  print(" Axe ||   Contribution\t|| Cumulative contribution")
  contrib_sum=0
  for i in range(s.shape[0]):
    contrib = s[s.shape[0]-1-i] / s.sum() * 100
    contrib_sum = contrib_sum + contrib
    print("  {}  ||      {:.1f} % \t|| \t   {:.1f} %    ".format(i+1, contrib, contrib_sum))
