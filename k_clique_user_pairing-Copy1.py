
#imports
import torch
import torch.nn.functional as F
from torch.nn import Linear
from itertools import product
import time
from torch import tensor
from torch.optim import Adam
from torch.optim import SGD
from math import ceil
from torch.nn import Linear
from torch.distributions import categorical
from torch.distributions import Bernoulli
import torch.nn
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
from torch_geometric.utils import convert as cnv
from torch_geometric.utils import sparse as sp
from torch_geometric.data import Data
from networkx.drawing.nx_agraph import graphviz_layout
import networkx as nx
from torch.utils.data.sampler import RandomSampler
from torch.nn.functional import gumbel_softmax
from torch.distributions import relaxed_categorical
import myfuncs
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.inits import glorot, zeros
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import GINConv, GATConv, global_mean_pool, NNConv, GCNConv
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU, LeakyReLU
from torch_geometric.nn import MessagePassing
import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool
from torch_geometric.data import Batch 
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean
from torch import autograd
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.utils import softmax, add_self_loops, remove_self_loops, segregate_self_loops, remove_isolated_nodes, contains_isolated_nodes, add_remaining_self_loops
from torch_geometric.utils import dropout_adj, to_undirected, to_networkx
from torch_geometric.utils import is_undirected
from cut_utils import get_diracs
import scipy
import scipy.io
from matplotlib.lines import Line2D
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import GPUtil
from networkx.algorithms.approximation import max_clique
import pickle
from torch_geometric.nn import SplineConv, global_mean_pool, DataParallel
from torch_geometric.data import DataListLoader, DataLoader
from random import shuffle
from networkx.algorithms.approximation import max_clique
from networkx.algorithms import graph_clique_number
from networkx.algorithms import find_cliques
from torch_geometric.nn.norm import graph_size_norm
from torch_geometric.datasets import TUDataset
import visdom 
from visdom import Visdom 
import numpy as np
import matplotlib.pyplot as plt
from models import clique_MPNN
from torch_geometric.nn.norm.graph_size_norm import GraphSizeNorm
from modules_and_utils import decode_clique_final, decode_clique_final_speed
import numpy as np
import scipy.io as sio 
from matplotlib import pyplot as plt
from torch_scatter import scatter_min, scatter_max, scatter_add, scatter_mean

############################################################################################################################################
#                                                          DATASET DEFINITION
############################################################################################################################################
print("Setting up data.......")
K = 40 # no of users
with open("F:\\channel_40_1", "rb") as fp:
    dataset = pickle.load(fp)
    
dataset_scale = 1
total_samples = int(np.floor(len(dataset)*dataset_scale))
dataset = dataset[:total_samples]

num_trainpoints = int(np.floor(0.6*len(dataset)))
num_valpoints = int(np.floor(num_trainpoints/3))
num_testpoints = len(dataset) - (num_trainpoints + num_valpoints)

data_list = []
for i in range(1000):
    data_list.append(Data(x = dataset[i]["x"], num_nodes = K, edge_index = dataset[i]["edge_index"], edge_weight = dataset[i]["edge_weight"]))


traindata= data_list[0:num_trainpoints]
valdata = data_list[num_trainpoints:num_trainpoints + num_valpoints]
testdata = data_list[num_trainpoints + num_valpoints:]


batch_size = 1
train_loader = DataLoader(traindata, batch_size, shuffle=False)
for data in testdata:
    print(data)
print("Finished setting up data!")  

############################################################################################################################################
#                                                          DEFINE GNN
############################################################################################################################################

#set up random seeds 
torch.manual_seed(1)
np.random.seed(2)   
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#number of propagation layers
numlayers = 8

#size of receptive field
receptive_field = numlayers + 1

# val_losses = []
# cliq_dists = []

net =  clique_MPNN(dataset,numlayers, 128, 128,1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
lr_decay_step_size = 5
lr_decay_factor = 0.95

net.to(device).reset_parameters()
optimizer = Adam(net.parameters(), lr=0.001, weight_decay=0.00)


############################################################################################################################################
#                                                        TRAIN GNN
############################################################################################################################################
print("Starting Training....")
b_sizes = [50]
l_rates = [0.001]
depths = [8]
coefficients = [4.]
rand_seeds = [66]
widths = [64]

epochs = 100
net.train()
retdict = {}
edge_drop_p = 0.0
edge_dropout_decay = 0.90



for batch_size, learning_rate, numlayers, penalty_coeff, r_seed, hidden_1 in product(b_sizes, l_rates, depths, coefficients, rand_seeds, widths):
   
    torch.manual_seed(r_seed)

    train_loader = DataLoader(traindata, batch_size, shuffle=True)
    test_loader = DataLoader(testdata, batch_size, shuffle=False)
    val_loader =  DataLoader(valdata, batch_size, shuffle=False)

    receptive_field= numlayers + 1
    val_losses = []
    cliq_dists = []

    hidden_2 = 1

    net =  clique_MPNN(dataset,numlayers, hidden_1, hidden_2 ,1)
    net.to(device).reset_parameters()
    optimizer = Adam(net.parameters(), lr=learning_rate, weight_decay=0.00000)

    for epoch in range(epochs):
        totalretdict = {}
        count=0
        if epoch % 5 == 0:
            edge_drop_p = edge_drop_p*edge_dropout_decay
            print("Edge_dropout: ", edge_drop_p)

        if epoch % 10 == 0:
            penalty_coeff = penalty_coeff + 0.
            print("Penalty_coefficient: ", penalty_coeff)

        print("here2")
        #learning rate schedule
        if epoch % lr_decay_step_size == 0:
            for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_decay_factor * param_group['lr']

        #show currrent epoch and GPU utilizationss
        print('Epoch: ', epoch)
        GPUtil.showUtilization()



        #print("here3")


        net.train()
        for data in train_loader
            count += 1 
            optimizer.zero_grad(), 
            data = data.to(device)
            data_prime = get_diracs(data, 1, sparse = True, effective_volume_range=0.15, receptive_field = receptive_field)

            data = data.to('cpu')
            data_prime = data_prime.to(device)
#             print(data_prime)


#             retdict = net(data_prime, None, penalty_coeff)
            retdict = net(data_prime, data.edge_weight.to(device), None, penalty_coeff)

            for key,val in retdict.items():
                if "sequence" in val[1]:
                    if key in totalretdict:
                        totalretdict[key][0] += val[0].item()
                    else:
                        totalretdict[key] = [val[0].item(),val[1]]

            if epoch > 2:
                    retdict["loss"][0].backward()
                    #reporter.report()

                    torch.nn.utils.clip_grad_norm_(net.parameters(),1)
                    optimizer.step()
                    del(retdict)

        if epoch > -1:        
            for key,val in totalretdict.items():
                if "sequence" in val[1]:
                    val[0] = val[0]/(len(train_loader.dataset)/batch_size)
            del data_prime
print("Training done!")


############################################################################################################################################
#                                                          EVALUATE GNN
############################################################################################################################################

print("Starting inference....")
tbatch_size = batch_size
num_data_points = num_testpoints

batch_size = 1
test_data = testdata
test_loader = DataLoader(test_data, batch_size, shuffle=False)
net.to(device)
count = 1
bound = 4
#Evaluation on test set
net.eval()

gnn_nodes = []
gnn_edges = []
gnn_sets = {}

#set number of samples according to your execution time, for 10 samples
max_samples = 1

gnn_times = []
num_samples = max_samples
t_start = time.time()

for data in test_loader:
    num_graphs = data.batch.max().item()+1
    bestset = {}
    bestedges = np.zeros((num_graphs))
    maxset = np.zeros((num_graphs))
    print(data)
    total_samples = []
    for graph in range(num_graphs):
        curr_inds = (data.batch==graph)
        g_size = curr_inds.sum().item()
        if max_samples <= g_size: 
            samples = np.random.choice(curr_inds.sum().item(),max_samples, replace=False)
        else:
            samples = np.random.choice(curr_inds.sum().item(),max_samples, replace=True)

        total_samples +=[samples]

    data = data.to(device)
    t_0 = time.time()

    for k in range(num_samples):
        t_datanet_0 = time.time()
        data_prime = get_diracs(data.to(device), 1, sparse = True, effective_volume_range=0.15, receptive_field = 7)
  
        initial_values = data_prime.x.detach()
        data_prime.x = torch.zeros_like(data_prime.x)
        g_offset = 0
        for graph in range(num_graphs):
            curr_inds = (data_prime.batch==graph)
            g_size = curr_inds.sum().item()
            graph_x = data_prime.x[curr_inds]
            data_prime.x[total_samples[graph][k] + g_offset]=1.
            g_offset += g_size
            
        retdz = net(data_prime, data.edge_weight)
#         retdz = net(data_prime)
        
        t_datanet_1 = time.time() - t_datanet_0
#         print("data prep and fp: ", t_datanet_1)
        t_derand_0 = time.time()

        sets, set_edges, set_cardinality = decode_clique_final_speed(data_prime, data.edge_weight.to(device), (retdz["output"][0]), weight_factor =0., clique_number_bounds = bound, draw=False, beam = 1)

        t_derand_1 = time.time() - t_derand_0
#         print("Derandomization time: ", t_derand_1)

        for j in range(num_graphs):
            indices = (data.batch == j)
            if (set_cardinality[j]>maxset[j]):
                    maxset[j] = set_cardinality[j].item()
                    bestset[str(j)] = sets[indices].cpu()
                    bestedges[j] = set_edges[j].item()

    t_1 = time.time()-t_0
    print("Current batch: ", count)
#     print("Time so far: ", time.time()-t_0)
    gnn_sets[str(count)] = bestset
    
    gnn_nodes += [maxset]
    gnn_edges += [bestedges]
    gnn_times += [t_1]

    count += 1

t_1 = time.time()
total_time = t_1 - t_start
print("Average time per graph: ", total_time/(len(test_data)))

print("Inference done!")


############################################################################################################################################
#                                                          K-Means User Grouping
############################################################################################################################################

print("Performing k-means user grouping")
N = 100;
number = K;
k = bound
z = [[] for i in range(k)]
l = [[] for i in range(k)]
x = [[] for i in range(k)]
y = [[] for i in range(k)]
xm = []
ym = []
idx = []
def k_means(U, A, k, centers=None, num_iter=10):
    if centers is None:
        rnd_centers_idx = np.random.choice(np.arange(U.shape[0]), k, replace=False)
        centers = U[rnd_centers_idx]
        UUh = np.zeros([K,N,N], dtype = np.csingle)
        for i in range(K):
            r = np.count_nonzero(A[i,:,:] > 0.001)
            UUh[i,:,:] = U[i,:,N-r:N-1] @ U[i,:,N-r:N-1].conj().T
 
        VVh = np.zeros([k,N,N], dtype = np.csingle)
        for i in range(k):
            r = np.count_nonzero(A[rnd_centers_idx[i],:,:] > 0.001)
            VVh[i,:,:] = U[rnd_centers_idx[i],:,N-r:N-1] @ U[rnd_centers_idx[i],:,N-r:N-1].conj().T

    tot_dist_curr = 9999 
    tot_dist_prev = 0 
#     while np.abs(tot_dist_prev - tot_dist_curr)>1:
    for ur in range(5):
        tot_dist_prev = tot_dist_curr
        distances = (np.linalg.norm((UUh - VVh[:,np.newaxis]), axis=(2,3)))
        tot_dist_curr = np.sum(distances)
        cluster_assignments = np.argmin(distances, axis=0)
        for i in range(k):
            msk = (cluster_assignments == i)
            Z = UUh[cluster_assignments == i]
            Z = np.sum(Z, axis=0)/(np.count_nonzero(cluster_assignments == i)+1)
            LambdaW, W = np.linalg.eig((Z+Z.conj().T)/2)
            LambdaW = np.abs(LambdaW)
            idx = LambdaW.argsort()[::-1]   
            LambdaW = LambdaW[idx]
            W = W[:,idx]
            
            r = np.count_nonzero(LambdaW > 0.1)
            W = W[:,1:r]
            del LambdaW
            VVh[i,:,:] = W @ W.conj().T
    return cluster_assignments
to = 199
users_ind_kmeans = np.zeros([to,k])
users_ind_kmeans = users_ind_kmeans.astype('i')
k = bound
for iteration in range(801,801+to):
    print(iteration)

    mat = sio.loadmat(f'F:\\dataset_{number}\\U{iteration+1}.mat')
    eigen = mat['U'] 
    mat = sio.loadmat(f'F:\\dataset_{number}\\A{iteration+1}.mat')
    lamda = mat['A']
    labels = k_means(eigen,lamda,k)
    mat = sio.loadmat(f'F:\\dataset_{number}\\x{iteration+1}.mat')
    X = mat['x']
    mat = sio.loadmat(f'F:\\dataset_{number}\\y{iteration+1}.mat')
    Y = mat['y']
    z = [[] for i in range(k)]
    l = [[] for i in range(k)]
    x = [[] for i in range(k)]
    y = [[] for i in range(k)]
    for i in range(K):
        for j in range(k):
            if labels[i] == j:
                x[j].append(X[0,i])
                y[j].append(Y[0,i])
    xm = []
    for i in range(k):
        xm.append(np.mean(x[i]))
    ym = []
    for i in range(k):
        ym.append(np.mean(y[i]))
    for i in range(K):
        for j in range(k):
            if labels[i] == j:
                z[j].append(((X[0,i]-xm[j])**2+(Y[0,i]-ym[j])**2))
                l[j].append(i)
    for j in range(k):
        if np.size(z[j])%2==0:
            z[j].append(999999)

    idx = []
    for j in range(k):
        idx.append(np.argsort(z[j])[0])
    for j in range(k):
        if not l[j]:
            l[j].append(0)
        users_ind_kmeans[iteration-801][j] = l[j][idx[j]]

print(users_ind_kmeans)
   
labels = np.append(labels,[7, 7, 7, 7])
# labels = np.append(labels,[7, 7, 7, 7, 7, 7])
# x = np.append(X,[xm[0], xm[1], xm[2], xm[3], xm[4], xm[5]])
# y = np.append(Y,[ym[0], ym[1], ym[2], ym[3], ym[4], ym[5]])
x = np.append(X,[xm[0], xm[1], xm[2], xm[3]])
y = np.append(Y,[ym[0], ym[1], ym[2], ym[3]])
print(np.size(labels))

plt.scatter(x,y, c=labels)
plt.title('K-means user pairing', fontsize=11)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
# plt.scatter([xm1 xm2 xm3],[ym1 ym2 ym3], c=[5 5 5])
plt.show()
print("Finished k-means user grouping!")


############################################################################################################################################
#                                                             k-SUS
############################################################################################################################################

print("Starting semi-orthogonal user scheduling....")
# print(np.shape(users_ind_kmeans))
print(K)
users = gnn_sets[str(to)]['0']
# print((users))
mat = sio.loadmat(f'F:\\dataset_{number}\\x{801+to}.mat')
X = mat['x']
mat = sio.loadmat(f'F:\\dataset_{number}\\y{801+to}.mat')
Y = mat['y']

plt.scatter(X,Y,c=users)
plt.title('K-clique user pairing', fontsize=11)
plt.xlabel('X', fontsize=14)
plt.ylabel('Y', fontsize=14)
plt.show()

alpha = 0.01
users_ind_sus = np.zeros([to,k])
users_ind_sus = users_ind_sus.astype('i')
for p in range(801,801+to):
    T = list(range(0,K))
    P = [] 
    i = 0
    mat = sio.loadmat(f'F:\\dataset_{number}\\H{p}.mat')
    H = mat['H']
    g = H
    cond = np.size(P) < k+1
    while T and cond:
        i += 1
        
        for q in range(0,np.size(T)):
            gs = 0
            for j in range(0,i-1):
                gs += ((H[:,T[q]].T @ g[:,P[j]].conj())/np.linalg.norm(g[:,P[j]])**2) * g[:,P[j]]
            g[:,T[q]] = H[:,T[q]] - gs
        G = np.linalg.norm(g[:,T], axis = 0)
        P.append(T[np.argmax(G)])
        val = T[np.argmax(G)]
        T.remove(T[np.argmax(G)])
        Q = []
        for r in range(0,np.size(T)):
            sus = (H[:,T[r]].T @ g[:,val].conj())/(np.linalg.norm(H[:,T[r]])*np.linalg.norm(g[:,val]))
            if sus < alpha:
                Q.append(T[r])
        
        for s in range(0,np.size(Q)):
            T.remove(Q[s])
        cond = np.size(P) < k+1
        if np.size(P) == k or not T and cond:
            for idx in range(0,np.size(P)):
                users_ind_sus[p-801][idx] = P[idx]
        
print(users_ind_sus)
print("Finished semi-orthogonal user scheduling!")


############################################################################################################################################
#                                                       Waterfilling and Sum rate calculations
############################################################################################################################################
print("Starting sum rate calculations....")
from scipy.linalg import svd

def waterfilling(H,P):
    U, g, VT = svd(H)
#     print(np.shape(VT))
    alpha_low = 0 # Initial low
    alpha_high = (P + np.sum(1/g**2)) # Initial high

    stop_threshold = 1e-7 # Stop threshold

    # Iterate while low/high bounds are further than stop_threshold
    while(np.abs(alpha_low - alpha_high) > stop_threshold):
        alpha = (alpha_low + alpha_high) / 2 # Test value in the middle of low/high

        # Solve the power allocation
        p = 1/alpha - 1/g**2 
        p[p < 0] = 0 # Consider only positive power allocation

        # Test sum-power constraints
        if (np.sum(p) > P): # Exceeds power limit => lower the upper bound
            alpha_low = alpha
        else: # Less than power limit => increase the lower bound
            alpha_high = alpha
    return p

def sumrate(ebn,num):
    snr = 10**(ebn/10)
    sr_kmeans = []
    sr_clique = []
    sr_sus    = []
    dpc_kmeans= []
    dpc_clique= []
    
    users = gnn_sets[str(num-800)]['0']
    users_ind_clique = torch.argwhere(users)
    power_clique = 0
    power_dpc_clique = 0
    power_dpc_kmeans = 0
    interference_clique = 0
    capacity_clique = 0
    power_kmeans = 0
    interference_kmeans = 0
    capacity_kmeans = 0
    power_dbscan = 0
    interference_dbscan = 0
    capacity_dbscan = 0
    power_sus = 0
    interference_sus = 0
    capacity_sus = 0
    mat = sio.loadmat(f'F:\\dataset_{number}\\H{num}.mat')
    H = mat['H']
    H = H/np.linalg.norm(H, axis = 0)
    p_clique = waterfilling(H[:,np.asarray(users_ind_clique).squeeze(-1)],snr)
    count=-1
    for k in users_ind_clique:
        count+=1
        idx = -1
        power_dpc_clique += p_clique[count]*(np.reshape(H[:,k],(N,1)) @ np.reshape(H[:,k],(N,1)).conj().T)
        for j in users_ind_clique:
            idx+=1
            power_clique += p_clique[idx]*snr*np.abs(H[:,k] @ H[:,j].conj().T)**2     
            
        interference_clique = power_clique - p_clique[count]*snr*np.abs(H[:,k] @ H[:,k].conj().T)**2      
        capacity_clique += np.log2((1 + power_clique)/(1 + interference_clique))
        
    sr_clique.append(capacity_clique)
    
    p_kmeans = waterfilling(H[:,users_ind_kmeans[num-801]],snr)
    count=-1
    for k in users_ind_kmeans[num-801]:
        count+=1
        idx = -1
        power_dpc_kmeans += p_kmeans[count]*(np.reshape(H[:,k],(N,1)) @ np.reshape(H[:,k],(N,1)).conj().T)
        for j in users_ind_kmeans[num-801]:
            idx+=1
            power_kmeans += p_kmeans[idx]*snr*np.abs(H[:,k] @ H[:,j].conj().T)**2
        interference_kmeans = power_kmeans - p_kmeans[count]*snr*np.abs(H[:,k] @ H[:,k].conj().T)**2        
        capacity_kmeans += np.log2((1 + power_kmeans)/(1 + interference_kmeans))
        
    sr_kmeans.append(capacity_kmeans)

    
    p_sus = waterfilling(H[:,users_ind_sus[num-801]],snr)
    count=-1
    for k in users_ind_sus[num-801]:
        count+=1
        idx = -1
        for j in users_ind_sus[num-801]:
            idx+=1
            power_sus += p_sus[idx]*snr*np.abs(H[:,k] @ H[:,j].conj().T)**2
        interference_sus = power_sus - p_sus[count]*snr*np.abs(H[:,k] @ H[:,k].conj().T)**2        
        capacity_sus += np.log2((1 + power_sus)/(1 + interference_sus))
        
    sr_sus.append(capacity_sus)
    
    return np.asarray(sr_clique), np.asarray(sr_kmeans), np.asarray(sr_sus)

    
number = K

ebn = list(range(0,31,5))
snr = [10**(x/10) for x in ebn]
sr_kmeans = [[] for i in range(to)]
sr_clique = [[] for i in range(to)]
sr_sus    = [[] for i in range(to)]
dpc_clique = [[] for i in range(to)]
dpc_kmeans = [[] for i in range(to)]
for i in ebn:
    print(i)
    for j in range(801,801+to):
        sr1, sr2, sr3 = sumrate(i,j)      
        sr_clique[j-801].append(sr1)
        sr_kmeans[j-801].append(sr2)
        sr_sus[j-801].append(sr3)

plt.plot(ebn,np.mean(sr_kmeans, axis=0), label='k-means, K=40, k=4', marker = 'o', ms = 7, c = 'b')
plt.plot(ebn,np.mean(sr_clique, axis=0), label='k-clique, K=40, k=4', marker = '*', ms = 8, c = 'r')
plt.plot(ebn,np.mean(sr_sus, axis=0), label='k-sus, K=40, k=4', marker = 's', ms = 6, c = 'y')
plt.xlabel(r'$E_{b}/N_{0} (dB)$')
plt.ylabel("sum rate (bps/Hz)")
plt.legend()
plt.grid(True)
plt.show()

print("Finished sumrate calculations!")

############################################################################################################################################
#                                                          OTHER PLOTS
############################################################################################################################################
# Values collected from different runs of the GNN with the respective values of K's

clique = [6.52799823, 6.51964084, 6.62973278, 6.63232852, 6.62970638, 6.62598411, 6.62950228, 6.62635731, 6.62720659, 6.62298049]
kmeans = [5.80317788, 5.73147345, 5.63851779, 5.68550135, 5.68521737, 5.53902121, 5.62057428, 5.55284441, 5.50474545, 5.7004235 ]
dpc    = [7.1993973 , 7.19962125, 7.22801259, 7.17494089, 7.22361739, 7.21458155, 7.22799199, 7.18094093, 7.20266438, 7.18952159]
sus    = [5.17295826, 5.24305174, 4.97910948, 5.12277046, 5.13511962, 5.28560087, 5.44692029, 5.33897954, 5.29645529, 5.44272452]
users = list(range(10, 101, 10))
plt.plot(users,kmeans, label='k-means, k=4', marker = 'o', ms = 6, c = 'b')
plt.plot(users,clique, label='k-clique, k=4', marker = '*', ms = 8, c = 'r')
plt.plot(users,sus, label='k-sus, k=4', marker = 's', ms = 6, c = 'y')
plt.xlabel('number of users (K)')
plt.ylabel("sum rate (bps/Hz)")
plt.legend()
plt.grid(True)
plt.show()


flops_clique = [39.98, 79.96, 119.9, 159.9, 199.9, 239.8, 279.8, 319.8, 359.8, 399.8]
flops_kmeans = [107.5, 153.6, 199.8, 246.0, 292.2, 338.3, 384.5, 430.7, 476.9, 523.1]
# flops_sus    = [46.99, 88.60, 130.9, 178.8, 235.8, 282.4, 332.2, 384.4, 430.7, 470.7]
flops_sus    = [39.04, 78.12, 109.4, 155.3, 194.7, 233.8, 273.0, 311.5, 351.4, 388.5]

lat_clique = [0.006, 0.007, 0.011, 0.014, 0.017, 0.024, 0.03, 0.037, 0.044, 0.054]
lat_kmeans = [0.161, 0.162, 0.168, 0.192, 0.2, 0.239, 0.267, 0.299, 0.324, 0.336]
lat_sus    = [0.0009, 0.0010, 0.0010, 0.0015, 0.0019, 0.0025, 0.0021, 0.0035, 0.0045, 0.00385]

plt.plot(users,flops_kmeans, label='k-means, k=4', marker = 'o', ms = 7, c = 'b')
plt.plot(users,flops_clique, label='k-clique, k=4', marker = '*', ms = 8, c = 'r')
plt.plot(users,flops_sus, label='k-sus, k=4', marker = 's', ms = 6, c = 'y')
plt.xlabel('number of users (K)')
plt.ylabel("number of FLOPs $ \cdot 10^{6}$")
plt.legend()
plt.grid(True)
plt.show()


plt.semilogy(users,lat_kmeans, label='k-means, k=4', marker = 'o', ms = 8, c = 'b')
plt.semilogy(users,lat_clique, label='k-clique, k=4', marker = '*', ms = 8, c = 'r')
plt.semilogy(users,lat_sus, label='k-sus, k=4', marker = 's', ms = 6, c = 'y')
plt.xlabel('number of users (K)')
plt.ylabel("Runtime $(sec)$")
plt.legend()
plt.grid(True)
plt.show()


sr = [6.25853899, 6.19661747, 5.52504612, 5.72445559, 5.78752572, 5.63617305, 5.77002804, 5.80513293, 5.77119982, 5.75396522]
sm = [6.25082681, 6.20439943, 6.63113   , 6.63203779, 6.62999324, 6.62615638, 6.62810202, 6.62924805, 6.62601823, 6.62692901]
users = list(range(10, 101, 10))
print(users)
plt.bar(users,sm, color ='b', width = 3)
# plt.plot(ebn,np.mean(dpc_kmeans, axis=0), label='dpc_kmeans')
plt.xlabel('number of users (K)')
plt.ylabel("sumrate")
# plt.legend()
# plt.grid(True)
plt.show()


