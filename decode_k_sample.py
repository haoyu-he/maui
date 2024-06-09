import os
import csv
import scipy.sparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, average_precision_score, \
    roc_auc_score, accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.preprocessing import normalize, minmax_scale

import torch
import torch.nn.functional as F
import torch_geometric.utils

from torch_geometric.nn.models import GCN, GraphSAGE, GAT
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph, to_scipy_sparse_matrix, sort_edge_index
import torch_geometric.transforms as T

from configs import get_arguments
from load_datasets import get_dataset
from attacks.link_attack import Attacker, auc_node
from attacks.link_teller import LinkTeller
from models import GAT_ad


args = get_arguments()
dataset_name = args.dataset.lower()
epsilon = args.epsilon
if args.eval_true or args.defense != 1:
    epsilon = None
dataset = get_dataset('./datasets', dataset_name, epsilon=epsilon)
data = dataset.data
normalize_feat = T.NormalizeFeatures()
data = normalize_feat(data)
print(torch_geometric.utils.is_undirected(data.edge_index))
print(data.num_edges/2)

print(data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum())
print(data.train_mask.sum() / data.num_nodes, data.val_mask.sum() / data.num_nodes, data.test_mask.sum() / data.num_nodes)

if args.model.lower() == 'gcn':
    gnn = GCN(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              jk='last')
elif args.model.lower() == 'sage':
    gnn = GraphSAGE(in_channels=dataset.num_node_features,
                    hidden_channels=args.hidden_channels,
                    num_layers=args.num_layers,
                    out_channels=dataset.num_classes,
                    dropout=args.dropout,
                    jk='last',
                    aggr='max')
elif args.model.lower() == 'gat':
    gnn = GAT(in_channels=dataset.num_node_features,
              hidden_channels=args.hidden_channels,
              num_layers=args.num_layers,
              out_channels=dataset.num_classes,
              dropout=args.dropout,
              jk='last',
              heads=8)
else:
    raise NotImplementedError('GNN not implemented!')
model_dir = './src'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)
model_name = dataset_name + '_' + args.model.lower() + '_l' + str(args.num_layers)
if args.defense == 1:
    model_name += '_ep' + '%.1f' % args.epsilon
elif args.defense == 2:
    gnn = GAT_ad(in_channels=dataset.num_node_features,
                 hidden_channels=args.hidden_channels,
                 num_layers=args.num_layers,
                 out_channels=dataset.num_classes,
                 dropout=args.dropout,
                 jk='last')
    model_name += '_ad' + '%.1f' % args.beta
gnn.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
gnn.eval()
out = gnn(data.x, data.edge_index)
pred = out.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = torch.div(correct / data.test_mask.sum(), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc:.4f}')
print('#Edges:', data.edge_index.shape[1] / 2)
print()

result_dir = os.path.join('influence_values', dataset_name)
rng = np.random.default_rng(0)

topk = [0.25, 0.5, 0.75, 1, 1.25, 1.5]

precision_topk = np.zeros((args.sample_num, len(topk)))
precision_norm = np.zeros((args.sample_num, len(topk)))
recall_topk = np.zeros((args.sample_num, len(topk)))
recall_norm = np.zeros((args.sample_num, len(topk)))

del gnn.lin
for s in range(args.sample_num):

    sample_dir = os.path.join(result_dir, str(s))
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    sample_nodes = rng.choice(dataset.data.num_nodes, args.sample_size, replace=False, shuffle=False)
    sample_mask = torch.zeros(dataset.data.num_nodes, dtype=torch.bool)
    sample_mask[sample_nodes] = 1

    data = dataset.data.clone()
    data = data.subgraph(sample_mask)
    data.edge_index = sort_edge_index(data.edge_index)

    attacker = Attacker(gnn, data.edge_index, data.x, sample=args.sample)
    attacker.update_influence_matrix(sample_dir, defense_ep=epsilon)
    if args.attacker == 0:
        if args.combo:
            res_name = os.path.join(sample_dir, 'attacker' + str(args.attacker) + '_' +
                                    model_name + '_c.npz')
        elif args.efficient:
            res_name = os.path.join(sample_dir, 'attacker' + str(args.attacker) + '_' +
                                    model_name + '_e.npz')
        else:
            res_name = os.path.join(sample_dir, 'attacker' + str(args.attacker) + '_' +
                                    model_name + '.npz')
    elif args.attacker == 1:
        res_name = os.path.join(sample_dir, 'attacker' + str(args.attacker) + '_' +
                                model_name + '.npz')
    elif args.attacker == 2:
        res_name = os.path.join(sample_dir, 'attacker' + str(args.attacker) + 'pos' + str(args.pos_type) + '_' +
                                model_name + '.npz')
    else:
        raise ValueError(f'attacker={args.attacker} error')

    targets = range(data.num_nodes)

    adj_true = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).detach().numpy().astype(bool)
    # adj_true = to_scipy_sparse_matrix(data.edge_index).tocsr()
    emb = gnn(data.x, data.edge_index)
    adj = torch.matmul(emb, emb.t()) * torch.tensor(attacker.influence_matrix)
    adj_ori = adj.detach().numpy()
    adj_true = adj_true[targets, :]
    adj_true = adj_true[:, targets]
    adj_ori = adj_ori[targets, :]
    adj_ori = adj_ori[:, targets]

    adj_topk = adj_ori + adj_ori.transpose()
    adj_value = normalize(adj_ori, norm='max', axis=1)
    adj_norm = adj_value + adj_value.transpose()

    for i, k in enumerate(topk):

        print('Sample:', s, 'k:', k)
        print('---')
        '''top-k'''
        print('Top-k')

        y_idx = np.argpartition(adj_topk, -int(data.num_edges * k),
                                axis=None)[-int(data.num_edges * k):]
        y_pred = np.zeros_like(adj_topk).reshape(-1)
        y_pred[y_idx] = 1
        y = adj_true.reshape(-1)
        y_value = adj_topk.reshape(-1)

        result = precision_recall_fscore_support(y, y_pred)
        precision_topk[s, i] = result[0][1]
        recall_topk[s, i] = result[1][1]
        print('Precision, recall, F1:', result)

        '''normalization'''
        print('\nNormalization')
        y_idx = np.argpartition(adj_norm, -int(data.num_edges * k),
                                axis=None)[-int(data.num_edges * k):]
        y_pred = np.zeros_like(adj_norm).reshape(-1)
        y_pred[y_idx] = 1
        y = adj_true.reshape(-1)

        result = precision_recall_fscore_support(y, y_pred)
        precision_norm[s, i] = result[0][1]
        recall_norm[s, i] = result[1][1]
        print('Precision, recall, F1:', result)
        print()

float_formatter = "{:.1f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

precision_topk_mean = [100] + (precision_topk.mean(axis=0)*100).tolist()
precision_topk_sd = [0] + (precision_topk.std(axis=0)*100).tolist()
recall_topk_mean = [0] + (recall_topk.mean(axis=0)*100).tolist()
recall_topk_sd = [0] + (recall_topk.std(axis=0)*100).tolist()

precision_norm_mean = [100] + (precision_norm.mean(axis=0)*100).tolist()
precision_norm_sd = [0] + (precision_norm.std(axis=0)*100).tolist()
recall_norm_mean = [0] + (recall_norm.mean(axis=0)*100).tolist()
recall_norm_sd = [0] + (recall_norm.std(axis=0)*100).tolist()

print()
print('topk')
print('precision_topk_mean:', precision_topk_mean)
print('precision_topk_sd:', precision_topk_sd)
print('recall_topk_mean:', recall_topk_mean)
print('recall_topk_sd:', recall_topk_sd)

print()
print('norm')
print('precision_norm_mean:', precision_norm_mean)
print('precision_norm_sd:', precision_norm_sd)
print('recall_norm_mean:', recall_norm_mean)
print('recall_norm_sd:', recall_norm_sd)

