import os
import torch
import scipy.sparse as sp
import numpy as np

from sklearn.metrics import precision_score, recall_score
from torch_geometric.datasets import Planetoid, Amazon, GitHub, FacebookPagePage, LastFMAsia, \
    DeezerEurope, WikiCS, Flickr, Twitch, Coauthor
from torch_geometric.utils import to_dense_adj, sort_edge_index, to_scipy_sparse_matrix, from_scipy_sparse_matrix
from torch_geometric.data import Data

rng = np.random.default_rng(0)


def get_dataset(dataset_dir: str, dataset_name: str, epsilon: float = None):
    """load dataset"""
    if dataset_name.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(root=dataset_dir, name=dataset_name)
    elif dataset_name.lower() in ['computers', 'photo']:
        dataset = Amazon(root=dataset_dir, name=dataset_name)
    elif dataset_name.lower() == 'github':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = GitHub(root=dataset_dir)
    elif dataset_name.lower() == 'facebook':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = FacebookPagePage(root=dataset_dir)
    elif dataset_name.lower() == 'lastfmasia':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = LastFMAsia(root=dataset_dir)
    elif dataset_name.lower() == 'deezereurope':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = DeezerEurope(root=dataset_dir)
    elif dataset_name.lower() == 'wikics':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = WikiCS(root=dataset_dir, is_undirected=True)
    elif dataset_name.lower() == 'flicker':
        dataset_dir = os.path.join(dataset_dir, dataset_name)
        dataset = Flickr(root=dataset_dir)
    elif dataset_name.lower() in ['de', 'en', 'es', 'fr', 'pt', 'ru']:
        dataset = Twitch(root=dataset_dir, name=dataset_name.upper())
    elif dataset_name.lower() in ['cs', 'physics']:
        dataset = Coauthor(root=dataset_dir, name=dataset_name)
    else:
        raise NotImplementedError

    '''split data'''
    data = dataset[0]
    num_train = int(data.num_nodes * 0.6)
    num_val = int(data.num_nodes * 0.2)
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    perm_index = torch.randperm(data.num_nodes, generator=torch.random.manual_seed(0))
    data.train_mask[perm_index[:num_train]] = True
    data.val_mask[perm_index[num_train:num_train + num_val]] = True
    data.test_mask[perm_index[num_train + num_val:]] = True

    data.true_num_edges = data.num_edges

    if epsilon is None:
        dataset.data = data
        return dataset

    '''differential privacy'''
    adj = to_scipy_sparse_matrix(data.edge_index, num_nodes=data.num_nodes).tocoo()
    adj_noisy = perturb_adj_continuous(adj, epsilon)
    edge_index = from_scipy_sparse_matrix(adj_noisy)[0]

    data.edge_index = edge_index
    dataset.data = data

    return dataset


def construct_sparse_mat(indice, N):
    cur_row = -1
    new_indices = []
    new_indptr = []

    for i, j in indice:
        if i >= j:
            continue

        while i > cur_row:
            new_indptr.append(len(new_indices))
            cur_row += 1

        new_indices.append(j)

    while N > cur_row:
        new_indptr.append(len(new_indices))
        cur_row += 1

    data = np.ones(len(new_indices), dtype=np.int64)
    indices = np.asarray(new_indices, dtype=np.int64)
    indptr = np.asarray(new_indptr, dtype=np.int64)

    mat = sp.csr_matrix((data, indices, indptr), (N, N))

    return mat + mat.T


def perturb_adj_continuous(adj, epsilon):
    n_nodes = adj.shape[0]
    n_edges = len(adj.data) // 2

    N = n_nodes

    A = sp.tril(adj, k=-1)

    eps_1 = epsilon * 0.01
    eps_2 = epsilon - eps_1
    noise = get_noise(noise_type='laplace', size=(N, N),
                      eps=eps_2, delta=1e-5, sensitivity=1)
    noise *= np.tri(*noise.shape, k=-1, dtype=np.bool)
    A += noise
    n_edges_keep = n_edges + int(
        get_noise(noise_type='laplace', size=1,
                  eps=eps_1, delta=1e-5, sensitivity=1)[0])
    a_r = A.A.ravel()

    n_splits = 50
    len_h = len(a_r) // n_splits
    ind_list = []
    for i in range(n_splits - 1):
        ind = np.argpartition(a_r[len_h * i:len_h * (i + 1)], -n_edges_keep)[-n_edges_keep:]
        ind_list.append(ind + len_h * i)

    ind = np.argpartition(a_r[len_h * (n_splits - 1):], -n_edges_keep)[-n_edges_keep:]
    ind_list.append(ind + len_h * (n_splits - 1))

    ind_subset = np.hstack(ind_list)
    a_subset = a_r[ind_subset]
    ind = np.argpartition(a_subset, -n_edges_keep)[-n_edges_keep:]

    row_idx = []
    col_idx = []
    for idx in ind:
        idx = ind_subset[idx]
        row_idx.append(idx // N)
        col_idx.append(idx % N)
        assert (col_idx < row_idx)
    data_idx = np.ones(n_edges_keep, dtype=np.int32)

    mat = sp.csr_matrix((data_idx, (row_idx, col_idx)), shape=(N, N))
    return mat + mat.T


def get_noise(noise_type, size, eps=10, delta=1e-5, sensitivity=2):

    if noise_type == 'laplace':
        noise = rng.laplace(0, sensitivity/eps, size)
    elif noise_type == 'gaussian':
        c = np.sqrt(2*np.log(1.25/delta))
        stddev = c * sensitivity / eps
        noise = rng.normal(0, stddev, size)
    else:
        raise NotImplementedError('noise {} not implemented!'.format(noise_type))

    return noise
