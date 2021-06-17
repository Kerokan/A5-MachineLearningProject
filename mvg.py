import numpy as np
import scipy

def compute_mu(dataset):
    return dataset.mean(axis=1, keepdims=True)

def compute_C(dataset):
    mu = compute_mu(dataset)
    mu = mu.reshape(mu.size, 1)
    dataCenter = dataset - mu
    dataCenterT = dataCenter.T
    C = np.dot(dataCenter,dataCenterT)
    N = len(dataset[0,:])
    C = (1/N) * C
    return C

def compute_accuracy(pred,labels):
    accuracy = 0
    for i in range(len(labels)):
        if pred[i] == labels[i]:
            accuracy += 1
    return 100*accuracy/len(labels)

def logpdf_GAUND(x, mu, C):
    constant = -0.5*len(C[0])*np.log(2*np.pi)
    logDet = -0.5*np.linalg.slogdet(C)[1]
    product = -0.5*(((x-mu).T).dot(np.linalg.inv(C)).dot((x-mu)))

    return constant + logDet + product

def compute_likelihood(x, mu, var, nbClasses):
    S = np.zeros((nbClasses,x.shape[1])) #Score matrix init.
    for i in range(len(x[0])):
        for j in range(nbClasses):
            values = x.T[i]
            values = values.reshape(values.size, 1)
            ll_samples = np.exp(logpdf_GAUND(values, mu[j], var[j])) #density of i-th record, considering it belongs to the j-th class.
            S[j,i] = ll_samples

    return S

def compute_log_likelihood(x, mu, var, nbClasses):
    S = np.zeros((nbClasses,x.shape[1])) #Score matrix init.
    for i in range(len(x[0])):
        for j in range(nbClasses):
            values = x.T[i]
            values = values.reshape(values.size, 1)
            ll_samples = logpdf_GAUND(values, mu[j], var[j]) #log-density of i-th record, considering it belongs to the j-th class.
            S[j,i] = ll_samples

    return S

def MVG(DTR, LTR, DTE, LTE, Pc):

    DTR0 = DTR[:, LTR==0]
    DTR1 = DTR[:, LTR==1]
    mu0 = compute_mu(DTR0)
    mu1 = compute_mu(DTR1)
    sig0 = compute_C(DTR0)
    sig1 = compute_C(DTR1)

    MU = [mu0,mu1]
    VAR = [sig0,sig1]

    S = compute_likelihood(DTE,MU,VAR,2)
    SJoint = S*Pc
    SMar = SJoint.sum(axis=0)
    SPost = SJoint/SMar

    predictions = SPost.argmax(axis=0)
    print(predictions)

    acc = compute_accuracy(predictions,LTE)
    return acc

def Naive_Bayes(DTR, LTR, DTE, LTE, Pc):

    DTR0 = DTR[:, LTR==0]
    DTR1 = DTR[:, LTR==1]
    mu0 = compute_mu(DTR0)
    mu1 = compute_mu(DTR1)
    sig0 = compute_C(DTR0)*np.identity(8)
    sig1 = compute_C(DTR1)*np.identity(8)

    MU = [mu0,mu1]
    VAR = [sig0,sig1]

    S = compute_log_likelihood(DTE,MU,VAR,2)
    SJoint = S + np.log(Pc)
    SMar = scipy.special.logsumexp(SJoint)
    SPost = SJoint - SMar

    predictions = SPost.argmax(axis=0)
    print(predictions)

    acc = compute_accuracy(predictions,LTE)
    return acc

def Tied_Covariance(DTR, LTR, DTE, LTE, Pc):

    DTR0 = DTR[:, LTR==0]
    DTR1 = DTR[:, LTR==1]
    mu0 = compute_mu(DTR0)
    mu1 = compute_mu(DTR1)
    sig0 = compute_C(DTR0)
    sig1 = compute_C(DTR1)

    MU = [mu0,mu1]
    VAR = [sig0,sig1]
    SIG = 0
    for i in range(2):
        SIG += VAR[i]*len([j for j in LTR if j==i])
    SIG = SIG/len(LTR)

    S = compute_log_likelihood(DTE,MU,[SIG,SIG],2)
    SJoint = S + np.log(Pc)
    SMar = scipy.special.logsumexp(SJoint)
    SPost = SJoint - SMar

    predictions = SPost.argmax(axis=0)
    print(predictions)

    acc = compute_accuracy(predictions,LTE)
    return acc