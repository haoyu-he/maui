import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.utils import k_hop_subgraph

import time
import os
import numpy as np
from scipy.sparse import dok_matrix, coo_matrix, load_npz, save_npz
from tqdm import tqdm

EPS = 1e-15


class LinkTeller:

    def __init__(self, model: torch.nn.Module, edge_index: Adj, x: Tensor, influence=0.0001):
        self.model = model
        self.edge_index = edge_index
        self.x = x
        self.influence = influence
        self.influence_matrix = np.zeros((self.x.shape[0], self.x.shape[0]), dtype=bool)
        self.influence_values = dok_matrix((self.x.shape[0], self.x.shape[0]), dtype=float)

        out = self.model(self.x, self.edge_index)
        self.pred = F.softmax(out, dim=-1)

    def update_influence_matrix(self, src_dir, defense_ep: float = None, erase: bool = False):
        run_time = 0
        if defense_ep is not None:
            influence_nodes_path = os.path.join(src_dir, 'influence_matrix_' + str(self.model.num_layers)
                                                + '_ep%.1f.npz' % defense_ep)
        else:
            influence_nodes_path = os.path.join(src_dir, 'influence_matrix_' + str(self.model.num_layers) + '.npz')
        if os.path.exists(influence_nodes_path) and not erase:
            self.influence_matrix = load_npz(influence_nodes_path).toarray().astype(bool)
        else:
            for i in tqdm(range(self.x.shape[0]), desc='Processing influence nodes'):
                start_time = time.time()
                self.find_influence_nodes(i)
                run_time += time.time() - start_time
            save_npz(influence_nodes_path, coo_matrix(self.influence_matrix, dtype=bool))
            print('Runtime: {:.1f}'.format(run_time // 0.01 * 0.01))
        return run_time

    def find_influence_nodes(self, target_idx: int, pert: float = 1e-4):
        x_pert = self.x.clone()
        x_pert[target_idx] = x_pert[target_idx] + pert * torch.rand(self.x.shape[1])
        x_pert[target_idx] /= x_pert[target_idx].sum()
        out_pert = self.model(x_pert, self.edge_index)
        pred_pert = F.softmax(out_pert, dim=-1)
        pred_diff = self.pred - pred_pert
        diff = torch.sum(torch.abs(pred_diff), dim=-1)
        diff[target_idx] = 0
        diff = torch.nonzero(diff).view(-1)
        self.influence_matrix[target_idx, diff] = True

    def update_influence_value(self, adj_values):
        self.influence_values = adj_values

    def get_gradient(self, target_idx: int, node_idx: int):
        res = torch.zeros(self.x.shape[-1])
        for i in range(self.x.shape[-1]):
            pert = torch.zeros_like(self.x)
            pert[node_idx, i] = self.influence
            with torch.no_grad():
                grad = (F.softmax(self.model(self.x + pert, self.edge_index), dim=-1) -
                        F.softmax(self.model(self.x - pert, self.edge_index), dim=-1)) / (2 * self.influence)
                res[i] = grad[target_idx].sum()
        return res

    def get_gradient_eps(self, target_idx: int, node_idx: int, norm: bool = False):
        union_index, edge_index, target_inv, edge_mask = k_hop_subgraph(int(target_idx), self.model.num_layers * 2 + 1,
                                                                        self.edge_index, relabel_nodes=True)
        x = self.x[union_index]
        node_dict = dict()
        for i, u in enumerate(union_index.tolist()):
            node_dict[u] = i
        node_inv = node_dict[node_idx]

        pert = x.clone()
        pert[node_inv] += self.x[node_inv] * self.influence
        if norm:
            pert /= pert.sum(dim=-1, keepdim=True) + EPS
        grad = (F.softmax(self.model(pert, edge_index), dim=-1) - self.pred[union_index]) / self.influence
        return grad[target_inv]

    def get_gradient_eps_mat(self, target_idx: int, norm: bool = False):
        influence_nodes = np.flatnonzero(self.influence_matrix[target_idx])
        if np.count_nonzero(influence_nodes) == 0:
            return

        union_index, edge_index, target_inv, edge_mask = k_hop_subgraph(int(target_idx), self.model.num_layers * 2 + 1,
                                                                        self.edge_index, relabel_nodes=True)
        x = self.x[union_index]
        node_dict = dict()
        for i, u in enumerate(union_index.tolist()):
            node_dict[u] = i
        # influence_nodes_new: new_index of influence nodes
        influence_nodes_new = np.array([node_dict[k] for k in influence_nodes])

        pert = x.clone()
        pert[target_inv] += pert[target_inv] * self.influence
        if norm:
            pert /= pert.sum(dim=-1, keepdim=True) + EPS
        grad = (F.softmax(self.model(pert, edge_index), dim=-1) - self.pred[union_index]) / self.influence
        grad[target_inv] = 0
        influence_values = torch.linalg.norm(grad, dim=-1)
        self.influence_values[influence_nodes, target_idx] = influence_values[influence_nodes_new].detach().numpy()

    def get_influence_value(self, target_idx: int, efficient: bool = False, norm: bool = False):
        if efficient:
            self.get_gradient_eps_mat(target_idx, norm)
        else:
            influence_nodes = np.flatnonzero(self.influence_matrix[target_idx])
            influence_values = torch.zeros(self.x.shape[0])
            for node_idx in influence_nodes:
                grad = self.get_gradient_eps(target_idx, node_idx, norm)
                influence_values[node_idx] = torch.linalg.norm(grad)
            influence_values[target_idx] = 0
            self.influence_values[target_idx] = influence_values.detach().numpy()
            return influence_values
