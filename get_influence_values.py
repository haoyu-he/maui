import os
import time
from tqdm import tqdm
from scipy.sparse import dok_matrix, save_npz, load_npz
from sklearn.preprocessing import normalize

import torch
import torch_geometric.utils

from torch_geometric.nn.models import GCN, GraphSAGE, GAT
import torch_geometric.transforms as T

from configs import get_arguments
from load_datasets import get_dataset
from attacks.link_attack import Attacker
from attacks.link_teller import LinkTeller
from attacks.link_stealing import LinkStealing
from models import GAT_ad

args = get_arguments()
dataset_name = args.dataset.lower()
epsilon = args.epsilon
if args.defense != 1:
    epsilon = None
dataset = get_dataset('./datasets', dataset_name, epsilon=epsilon)
data = dataset[0]
normalize_feat = T.NormalizeFeatures()
data = normalize_feat(data)
print(torch_geometric.utils.is_undirected(data.edge_index))
print(data.num_edges / 2)
self_loop = [range(data.num_nodes), range(data.num_nodes)]
self_loop = torch.tensor(self_loop, dtype=torch.long)

print(data.train_mask.sum(), data.val_mask.sum(), data.test_mask.sum())
print(data.train_mask.sum() / data.num_nodes, data.val_mask.sum() / data.num_nodes,
      data.test_mask.sum() / data.num_nodes)

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

x = data.x.clone()
# when node features are not accessible, we randomly set it or pick one that is known
if args.efficient or args.combo or not args.feat:
    x[:] = x[2]

if args.attacker == 0:
    attacker = Attacker(gnn, data.edge_index, x, sample=args.sample)
    if args.efficient or args.combo:
        res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + '_' +
                                model_name + '_e.npz')
    else:
        feat = '_feat_' if args.feat else '_'
        res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + feat +
                                model_name + '.npz')
elif args.attacker == 1:
    attacker = LinkTeller(gnn, data.edge_index, data.x)
    norm = '_norm_' if args.norm else '_'
    res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + norm +
                            model_name + '.npz')
elif args.attacker == 2:
    attacker = LinkStealing(gnn, data.edge_index, data.x, args.pos_type)
    res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + 'pos' + str(args.pos_type) + '_' +
                            model_name + '.npz')
else:
    raise ValueError(f'attacker={args.attacker} error')
# get influence matrix
time_in = attacker.update_influence_matrix(result_dir, defense_ep=epsilon)

if args.targets is None:
    targets_all = range(data.num_nodes)
elif type(args.targets) is int:
    targets_all = [args.targets]
elif len(args.targets) == 2:
    targets_all = range(args.targets[0], args.targets[1])
else:
    raise ValueError()

# continue previous attack
if os.path.exists(res_name) and not args.erase:
    adj_values = load_npz(res_name).todok()
    attacker.update_influence_value(adj_values)

# run the attack
run_time = 0
qbar = tqdm(targets_all)
for i, target_idx in enumerate(qbar):
    qbar.set_description('Node %d' % target_idx)
    start_time = time.time()
    if args.attacker == 0:
        if args.efficient or args.combo:
            attacker.get_influence_value_efficient(target_idx)
        else:
            attacker.get_influence_value(target_idx)
    elif args.attacker == 1:
        attacker.get_influence_value(target_idx, efficient=args.efficient, norm=args.norm)
    elif args.attacker == 2:
        attacker.get_influence_value(target_idx)
    run_time += time.time() - start_time

    # save in the middle in case it crashes
    if target_idx % 100 == 0:
        adj_values = attacker.influence_values.tocoo()
        adj_values.eliminate_zeros()
        save_npz(res_name, adj_values)
print('Runtime: {:.1f}'.format(run_time // 0.01 * 0.01))

'''Maui_comb'''
run_time = 0
if args.combo and args.attacker == 0:
    # save the result from Maui_efficient
    adj_values = attacker.influence_values.tocoo()
    adj_values.eliminate_zeros()
    save_npz(res_name, adj_values)

    adj_values_e = load_npz(res_name)
    adj_values = normalize(adj_values_e, norm='max', axis=1).tocoo()
    adj_values_ = dok_matrix(adj_values.shape, dtype=float)
    for i in range(adj_values.data.shape[0]):
        if adj_values.data[i] < args.combo_bar:
            adj_values_[adj_values.row[i], adj_values.col[i]] = -1e-8
    attacker.update_influence_value(dok_matrix(adj_values_))
    res_name = os.path.join(result_dir, 'attacker' + str(args.attacker) + '_' +
                            model_name + '_c.npz')

    qbar = tqdm(targets_all)
    for i, target_idx in enumerate(qbar):
        qbar.set_description('Combo-Node %d' % target_idx)
        start_time = time.time()
        attacker.get_influence_value(target_idx)
        run_time += time.time() - start_time
        if i % 100 == 0:
            adj_values = attacker.influence_values
            save_npz(res_name, adj_values.tocoo())
    attacker.influence_values[attacker.influence_values < 0] = 0
    attacker.influence_values = normalize(attacker.influence_values, norm='max', axis=1) + normalize(
        adj_values_e, norm='max', axis=1)
    print('Runtime: {:.1f}'.format(run_time // 0.01 * 0.01))

'''save the result'''
adj_values = attacker.influence_values.tocoo()
adj_values.eliminate_zeros()
save_npz(res_name, adj_values)
