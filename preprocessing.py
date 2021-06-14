import numpy as np
import scipy.linalg
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
    
def PCA(dataset,m):
    C = compute_C(dataset)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    Projection = np.dot(P.T,dataset.T)
    #print(U)
    #print(s)
    #print(P)
    return Projection

def PCA_return_all(dataset,m):
    C = compute_C(dataset)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    Projection = np.dot(P.T,dataset.T)
    return Projection, P, s, U

def compute_C(dataset):
    mu = dataset.mean(axis=0)
    mu = mu.reshape(1, mu.size)
    dataCenter = dataset - mu
    dataCenterT = dataCenter.T
    C = np.dot(dataCenterT,dataCenter)
    N = len(dataset[:,0])
    C = (1/N) * C
    return C

def compute_Sw(data,labels,n):
    Sw = np.zeros((len(data[0,:]),len(data[0,:])))
    
    for i in range(n):
        D = data[labels==i, :]
        C = compute_C(D)
        Sw += len(D[:,0]) * C
    Sw = (1/len(data[:,0]))*Sw

    return Sw


def compute_Sb(data,labels,n):
    Sb = np.zeros((len(data[0,:]),len(data[0,:])))
    mu = data.mean(axis=0)
    mu = mu.reshape(1, mu.size)

    for i in range(n):
        D = data[labels==i, :]
        muc = D.mean
        muc = muc.reshape(1, muc.size)
        a = muc - mu
        b = a.T
        Sb += len(D[:,0]) * np.dot(b,a)
    Sb = (1/len(data[:,0])) * Sb

    return Sb


def LDA1(data,labels,n,m):
    Sw = compute_Sw(data,labels,n)
    Sb = compute_Sb(data,labels,n)
    s, U = scipy.linalg.eigh(Sb,Sw)
    W = U[:, ::-1][:, 0:m]
    return W


def LDA2(data,labels,n,m):
    Sw = compute_Sw(data,labels,n)
    Sb = compute_Sb(data,labels,n)
    U, s, _ = np.linalg.svd(Sw)
    temp = 1.0/(s**0.5)
    temp = temp.reshape(temp.size, 1)
    P1 = np.dot(U*temp, U.T)
    Sbt = np.dot(np.dot(P1,Sb),P1.T)
    s2, V = np.linalg.eigh(Sbt)
    P2 = V[:, ::-1][:, 0:m]
    W = np.dot(P1.T,P2)
    return W