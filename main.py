import numpy as np
import matplotlib.pyplot as plt
import read
import display
import preprocessing
import knn

train_data = read.read_data('Train.txt')
test_data  = read.read_data('Test.txt')

normalized_data = preprocessing.normalize(train_data)

train_set, eval_set = preprocessing.split_train_val(normalized_data)

predicted = knn.knn_all(5, train_set, eval_set)

display.confusion_matrix(eval_set[:,-1], predicted)