import numpy as np

def read_data(file_name):

    file = open("Data/" + file_name, 'r')

    data = [line.split(',') for line in file]
    data = np.array(data).astype(float)

    print('Loaded {} - {} rows, {} columns (including label)'.format(file_name, data.shape[0], data.shape[1]))

    return data