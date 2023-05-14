#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Graph data generation in accordance with PyGeometric

import numpy as np
import torch
import scipy.io as sio 
from matplotlib import pyplot as plt
import pickle
N = 100;  # no of antennas
K = 100;  # no of users
alpha = 1;
dataset = []
for iter in range(1000): # 1000 channel matrices
    print(iter)
    mat = sio.loadmat(f'F:\\dataset_{K}\\H{iter+1}.mat')
    H = mat['H'] 

    distances = np.zeros([K,K])
    for i in range(K):
        for j in range(K):
            if i!=j:
                distances[i,j] = np.log2(1 + np.abs(H[:,i] @ H[:,i].conj().T)**2/(1 + np.abs(H[:,i] @ H[:,j].conj().T)**2))
    distances = distances + np.transpose(distances)
    adj = distances>alpha*np.mean(distances)

    X = []
    Y = []
    W = []
    for i in range(K):
        for j in range(K):
            if adj[i,j]:
                if i!=j:
                    X.append(i)
                    Y.append(j)
                    W.append(distances[i,j])
    Z = [X,Y]
    Z = torch.Tensor(Z).int()
    W = torch.Tensor(W/np.max(W))
    node_features = torch.ones((K,1))
    entry = {'edge_index': Z, 'x': node_features.float(), 'edge_weight': W}
    dataset.append(entry)

with open("F:\\channel_100_1", "wb") as fp:    # naming convention: channel_{no of users}_{alpha}
    pickle.dump(dataset, fp)


# In[ ]:




