import numpy as np
import random

def normalize(data):
    normalized_data = np.copy(data)
    for i in range(normalized_data.shape[1]-1):
        column = normalized_data[:,i]
        mean = column.sum() / column.shape[0]
        var = ((column-mean)**2).sum() / column.shape[0]

        column = (column-mean)/var
        normalized_data[:,i] = column
    return normalized_data

def split_train_val(train_data, proportion=0.33):

    data = train_data.copy()
    random.shuffle(data)
    n = len(data)

    train_set = data[int(n*proportion):]
    val_set   = data[:int(n*proportion)]
    print('Randomly splitted the data in two groups:', len(train_set), 'in train,', len(val_set), 'in validation')

    return np.asarray(train_set), np.asarray(val_set) 