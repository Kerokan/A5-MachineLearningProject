import numpy as np

def get_distance(pointA, pointB):
  distance = 0.0
  for i in range(pointA.shape[0]):
    distance = distance + (pointA[i] - pointB[i])**2
  distance = distance**(1/2)
  return distance

def knn_point(k, data, point):
    k_nearest = np.empty((0,2))

    for data_point in data:
        dist = get_distance(data_point[:-1], point)
        if k_nearest.shape[0] < k :
            k_nearest = np.append(k_nearest, np.array([[dist, data_point[-1]]]), axis=0)
            k_nearest = np.sort(k_nearest, axis=0)
        else:
            for i in range(k):
                if(dist <= k_nearest[i,0]):
                    k_nearest = np.insert(k_nearest, i, np.array([[dist, data_point[-1]]]), axis=0)
                    k_nearest = np.delete(k_nearest, -1, axis=0)
                    break
    if(k_nearest[:,1].sum() >= k/2):
        return 1
    else: 
        return 0

def knn_all(k, ref_data, to_predict):

  print('k={}'.format(k))

  results = []
  for i in range(to_predict.shape[0]):
    results.append(knn_point(3, ref_data, to_predict[i,:-1]))

  return results