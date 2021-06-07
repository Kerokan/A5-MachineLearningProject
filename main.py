import numpy as np
import matplotlib.pyplot as plt
import read
import display
import preprocessing

train_data = read.read_data('Train.txt')
test_data  = read.read_data('Test.txt')

normlized_data = preprocessing.normalize(train_data)
