import numpy as np

def logreg_obj(v, DTR, LTR, l):
  w, b = v[0:-1], v[-1]
  sum = 0
  for i in range(DTR.shape[1]):
    z=2*LTR[i] - 1
    value = np.log1p(np.exp(-z*(np.dot(w.T, DTR[:,i])+b)))
    sum = sum + value

  sum = sum / DTR.shape[1]

  sum = sum + (l/2) * np.linalg.norm(w)
  return sum

def score(v, element):
  w, b = v[0:-1], v[-1]
  result = np.dot(w.T, element) + b
  return result

def predict(v, DTE):
  LP = np.zeros(DTE.shape[1])
  scores = 0
  for i in range(DTE.shape[1]):
    scores = score(v, DTE[:,i])
    if scores > 0:
      LP[i] = 1

  return LP