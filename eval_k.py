import os
import scipy.sparse

import numpy as np
from sklearn.metrics import average_precision_score, \
    precision_recall_fscore_support
from sklearn.preprocessing import normalize

import torch
# import torch_geometric.utils

from torch_geometric.nn.models import GCN, GraphSAGE, GAT
from torch_geometric.utils import to_dense_adj, is_undirected
import torch_geometric.transforms as T

from configs import get_arguments
from load_datasets import get_dataset
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
print(f'Accuracy: {acc:.4f}\n')

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

adj_topk = adj_ori + adj_ori.transpose()
adj_value = normalize(adj_ori, norm='max', axis=1)
adj_norm = adj_value + adj_value.transpose()

topk = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
precision_topk = [100]
precision_norm = [100]
recall_topk = [0]
recall_norm = [0]
for i, k in enumerate(topk):
    print('k:', k)
    print('---')
    '''top-k'''
    print('Top-k')

    y_idx = np.argpartition(adj_topk, -int(data.num_edges * k),
                            axis=None)[-int(data.num_edges * k):]
    y_pred = np.zeros_like(adj_topk).reshape(-1)
    y_pred[y_idx] = 1
    y = adj_true.reshape(-1)

    result = precision_recall_fscore_support(y, y_pred)
    p = result[0][1]
    r = result[1][1]
    precision_topk.append(p * 100)
    recall_topk.append(r * 100)

    print('Precision, recall, F1:', result)
    print('Precision, recall: {:.2f}, {:.2f}'.format(p*100//0.01*0.01, r*100//0.01*0.01))

    '''normalization'''
    print('\nNormalization')

    y_idx = np.argpartition(adj_norm, -int(data.num_edges * k),
                            axis=None)[-int(data.num_edges * k):]
    y_pred = np.zeros_like(adj_norm).reshape(-1)
    y_pred[y_idx] = 1
    y = adj_true.reshape(-1)

    result = precision_recall_fscore_support(y, y_pred)
    p = result[0][1]
    r = result[1][1]
    precision_norm.append(p * 100)
    recall_norm.append(r * 100)

    print('Precision, recall, F1:', result)
    print('Precision, recall: {:.2f}, {:.2f}'.format(p*100//0.01*0.01, r*100//0.01*0.01))
    print()

float_formatter = "{:.1f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})

print('Precision topk:', precision_topk)
print('Recall topk:', recall_topk)
print('Precision norm:', precision_norm)
print('Recall norm:', recall_norm)