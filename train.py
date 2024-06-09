import os

import torch
import torch.nn.functional as F

from torch_geometric.nn import GCN, GraphSAGE, GAT
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_edge

from configs import get_arguments
from load_datasets import get_dataset
from models import GAT_ad

args = get_arguments()
dataset_name = args.dataset.lower()
epsilon = args.epsilon
if args.defense != 1:
    epsilon = None
dataset = get_dataset('./datasets', dataset_name, epsilon=epsilon)
data = dataset.data
normalize_feat = T.NormalizeFeatures()
data = normalize_feat(data)

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
beta = args.beta
if args.defense == 1:
    model_name += '_ep' + '%.1f' % args.epsilon
elif args.defense == 2:
    gnn = GAT_ad(in_channels=dataset.num_node_features,
                 hidden_channels=args.hidden_channels,
                 num_layers=args.num_layers,
                 out_channels=dataset.num_classes,
                 dropout=args.dropout,
                 jk='last',
                 heads=1)
    model_name += '_ad' + '%.1f' % beta

optimizer = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=5e-4)

early_stop = 100
early_stop_count = 0
best_acc = 0
best_loss = 100
EPS = 1e-15
for epoch in range(args.epochs):
    # doing this can achieve better accuracy
    if args.model.lower() == 'sage' and (dataset_name == 'cs' or dataset_name == 'github'):
        data.edge_index, _ = dropout_edge(dataset[0].edge_index)

    gnn.train()
    optimizer.zero_grad()
    out = gnn(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    if args.defense == 2:
        pred = F.softmax(out, dim=-1)
        # ent_pred = pred * torch.log(pred + EPS) + (1 - pred) * torch.log(1 - pred + EPS)
        # loss += ent_pred.mean() * beta
        for layer in gnn.convs:
            alpha = layer.alpha
            ent = -alpha * torch.log(alpha + EPS) - (1 - alpha) * torch.log(1 - alpha + EPS)
            loss += ent.mean() * beta
    loss.backward()
    optimizer.step()

    gnn.eval()
    out = gnn(data.x, data.edge_index)
    pred = out.argmax(dim=-1)
    eval_acc = (pred[data.val_mask] == data.y[data.val_mask]).sum()
    eval_loss = F.cross_entropy(out[data.val_mask], data.y[data.val_mask])
    print(epoch, eval_acc / data.val_mask.sum(), eval_loss)

    is_best = (eval_acc > best_acc) or (eval_loss < best_loss and eval_acc == best_acc)
    if is_best:
        early_stop_count = 0
        best_acc = eval_acc
        best_loss = eval_loss
        torch.save(gnn.state_dict(), os.path.join(model_dir, model_name + '.pt'))
    else:
        early_stop_count += 1
    if early_stop_count > early_stop:
        break

gnn.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
gnn.eval()
out = gnn(data.x, data.edge_index)
pred = out.argmax(dim=-1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = torch.div(correct / data.test_mask.sum(), 1e-4, rounding_mode='floor') * 1e-4
print(f'Accuracy: {acc*100:.2f}')
