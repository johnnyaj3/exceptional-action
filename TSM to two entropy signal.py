
import numpy as np
import glob
import math
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def entropy(arr,base):
    ent = 0
    for value in arr:
        ent -= value*math.log(value,base)
    return ent

def entropy_v1(dist):
    entropy_tmp = np.zeros([dist.shape[1]])
    dist *= -1
    for i in range(dist.shape[1]-3):
        if i >= dist.shape[0]+3:
            remain = dist[dist.shape[0]-1][dist.shape[1]-64:dist.shape[1],dist.shape[1]-64:dist.shape[1]][i-dist.shape[0]]
            value = entropy(softmax(remain[i-dist.shape[0]+1:-3]), math.e)
            entropy_tmp[i] = value
        elif i in [0, 1, 2]:
            continue
        else:
            value = entropy(softmax(dist[i-3,i,i+1:i+58]),math.e)
            entropy_tmp[i] = value
    return entropy_tmp

def entropy_v2(dist):
    entropy_v2_tmp = np.zeros([dist.shape[1]])
    for i in range(dist.shape[1]-3):
        if i >= dist.shape[0]+3:
            remain = dist[dist.shape[0]-1][dist.shape[1]-64:dist.shape[1],dist.shape[1]-64:dist.shape[1]][i-dist.shape[0]]
            remain /= np.sum(remain)
            value = entropy(remain[i-dist.shape[0]+1:-3], math.e)
            entropy_v2_tmp[i] = value

        elif i in [0, 1, 2]:
            continue
        else:
            dist[i-3,i,i+1:i+58] /= np.sum(dist[i-3,i,i+1:i+58])
            value = entropy(dist[i-3,i,i+1:i+58],math.e)
            entropy_v2_tmp[i] = value
    return entropy_v2_tmp



total_ = [i for i in glob.glob('./TSM/*npz')]
total_.sort()

for i in total_:
    FILE_NAME = i.split('/')[-1].split('.')[0]
    print(FILE_NAME)
    sims_1 = np.load(i)['dist']
    sims_2 = np.load(i)['dist']
    sims_entropy_v1 = entropy_v1(sims_1)
    sims_entropy_v2 = entropy_v2(sims_2)
    print(len(sims_entropy_v1))
    print(len(sims_entropy_v2))
    del sims_1
    del sims_2
    path = './entropy/'+FILE_NAME+'_entropy.npy'
    np.save(path, sims_entropy_v1)
    path = './entropy/'+FILE_NAME+'_entropy_v2.npy'
    np.save(path, sims_entropy_v2)
    del sims_entropy_v1
    del sims_entropy_v2
