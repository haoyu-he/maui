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
# import torch_geometric.utils

from torch_geometric.nn.models import GCN, GraphSAGE, GAT
from torch_geometric.utils import k_hop_subgraph, to_dense_adj, subgraph, to_scipy_sparse_matrix, is_undirected
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
print(is_undirected(data.edge_index))
print(data.num_edges/2)

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

result_dir = os.path.join('influence_values', dataset_name)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

if args.attacker == 0:
    if args.combo:
        res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + '_' +
                                model_name + '_c.npz')
    elif args.efficient:
        res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + '_' +
                                model_name + '_e.npz')
    else:
        feat = '_feat_' if args.feat else '_'
        res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + feat +
                                model_name + '.npz')
elif args.attacker == 1:
    norm = '_norm_' if args.norm else '_'
    res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + norm +
                            model_name + '.npz')
elif args.attacker == 2:
    res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + 'pos' + str(args.pos_type) + '_' +
                            model_name + '.npz')
else:
    raise ValueError(f'attacker={args.attacker} error')

is_whole_graph = False
if args.targets is None:
    targets = range(data.x.shape[0])
    is_whole_graph = True
elif type(args.targets) is int:
    targets = [args.targets]
elif len(args.targets) == 2:
    targets = range(args.targets[0], args.targets[1])
else:
    raise ValueError()

adj_true = to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).squeeze(0).detach().numpy().astype(bool)
# adj_true = to_scipy_sparse_matrix(data.edge_index).tocsr()
adj_ori = scipy.sparse.load_npz(res_name).toarray()
adj_true = adj_true[targets, :]
adj_true = adj_true[:, targets]
adj_ori = adj_ori[targets, :]
adj_ori = adj_ori[:, targets]

precision = []
AUC = []
for target_idx in tqdm(range(len(targets))):
    if np.count_nonzero(adj_true[target_idx]) == 0:
        continue
    pre = average_precision_score(adj_true[target_idx], adj_ori[target_idx])
    precision.append(pre)
    auc = roc_auc_score(adj_true[target_idx], adj_ori[target_idx])
    AUC.append(auc)

print('{:.1f}'.format(sum(precision)/len(precision)*100//0.01*0.01), min(precision), max(precision))
print(sum(AUC)/len(AUC), min(AUC), max(AUC))

topk = args.topk

'''top-k'''
print('\nTop-k')
adj_value = adj_ori + adj_ori.transpose()

y_idx = np.argpartition(adj_value, -int(data.num_edges * topk),
                        axis=None)[-int(data.num_edges * topk):]
y_pred = np.zeros_like(adj_value).reshape(-1)
y_pred[y_idx] = 1
y = adj_true.reshape(-1)
y_value = adj_value.reshape(-1)

result = precision_recall_fscore_support(y, y_pred)
p = result[0][1]
r = result[1][1]

print('Precision, recall, F1:', result)
print('Precision, recall: {:.1f}, {:.1f}'.format(p*100//0.01*0.01, r*100//0.01*0.01))
print('AUC: {:.1f}'.format(roc_auc_score(y, y_value)*100//0.01*0.01))
print('AP: {:.1f}'.format(average_precision_score(y, y_value)*100//0.01*0.01))

'''normalization'''
print('\nNormalization')
adj_value = normalize(adj_ori, norm='max', axis=1)
adj_topk = adj_value + adj_value.transpose()
adj_value_ = adj_value.transpose()
adj_value_[adj_value == 1] = 1
adj_value[adj_value_ == 1] = 1
adj_value = adj_value + adj_value_

y_idx = np.argpartition(adj_topk, -int(data.num_edges * topk),
                        axis=None)[-int(data.num_edges * topk):]
y_pred = np.zeros_like(adj_topk).reshape(-1)
y_pred[y_idx] = 1
y = adj_true.reshape(-1)
y_value = adj_value.reshape(-1)

result = precision_recall_fscore_support(y, y_pred)
p = result[0][1]
r = result[1][1]

print('Precision, recall, F1:', result)
print('Precision, recall: {:.1f}, {:.1f}'.format(p*100//0.01*0.01, r*100//0.01*0.01))
print('AUC: {:.1f}'.format(roc_auc_score(y, y_value)*100//0.01*0.01))
print('AP: {:.1f}'.format(average_precision_score(y, y_value)*100//0.01*0.01))
print()

'''when knowing extra knowledge'''
adj_wise = np.zeros_like(adj_true, dtype=bool)
adj_value = normalize(adj_ori, norm='max', axis=1)
adj_norm = adj_value + adj_value.transpose()
for i in range(data.num_nodes):
    num = np.count_nonzero(adj_true[i])
    if num == 0:
        continue
    idx = np.argpartition(adj_norm[i], -num, axis=None)[-num:]
    adj_wise[i, idx] = 1
adj_wise |= adj_wise.transpose()
adj_norm *= adj_wise
print(np.count_nonzero(adj_norm)/2)
for i in range(data.num_nodes):
    num_true = np.count_nonzero(adj_true[i])
    num_actual = np.count_nonzero(adj_norm[i])
    if num_true < num_actual:
        diff = num_actual - num_true
        non_zero_idx = np.flatnonzero(adj_norm[i])
        idx = np.argsort(adj_norm[i, non_zero_idx], axis=None)
        real_idx = non_zero_idx[idx]
        for j in real_idx:
            num_true = np.count_nonzero(adj_true[j])
            num_actual = np.count_nonzero(adj_norm[j])
            if num_true == num_actual:
                continue
            adj_norm[i, j] = 0
            adj_norm[j, i] = 0
            diff -= 1
            if not diff:
                break
print(np.count_nonzero(adj_norm)/2)
idx = np.argpartition(adj_norm, -data.num_edges, axis=None)[-data.num_edges:]
y_pred = np.zeros_like(adj_wise).reshape(-1)
y_pred[idx] = 1
y = adj_true.reshape(-1)
print('Precision wise: {:.1f}'.format(precision_score(y, y_pred)*100//0.01*0.01))