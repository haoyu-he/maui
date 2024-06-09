import scipy.sparse
import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj
from torch_geometric.utils import k_hop_subgraph, subgraph

import os
import random
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.sparse import dok_matrix
from tqdm import tqdm

rng = np.random.default_rng(0)


class Attacker:

    def __init__(self, model: torch.nn.Module, edge_index: Adj, x: Tensor, sample: int = None):
        self.model = model
        self.edge_index = edge_index
        self.x = x

        # for efficient strategy
        self.sample = sample

        out = self.model(self.x, self.edge_index)
        self.pred = F.softmax(out, dim=-1)

        self.influence_matrix = np.zeros((self.x.shape[0], self.x.shape[0]), dtype=bool)
        self.influence_values = dok_matrix((self.x.shape[0], self.x.shape[0]), dtype=float)

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

    def update_influence_matrix(self, src_dir: str, defense_ep: float = None, erase: bool = False) -> float:
        r"""Load the saved influence matrix. Returns the running time.

        Args:
            src_dir (str): file to load and save
            defense_ep (float): for LapGraph
            erase (bool): whether to recalculate the influence matrix
        """
        run_time = 0
        if defense_ep is not None:
            influence_nodes_path = os.path.join(src_dir, 'influence_matrix_' + str(self.model.num_layers)
                                                + '_ep%.1f.npz' % defense_ep)
        else:
            influence_nodes_path = os.path.join(src_dir, 'influence_matrix_' + str(self.model.num_layers) + '.npz')

        if os.path.exists(influence_nodes_path) and not erase:
            self.influence_matrix = scipy.sparse.load_npz(influence_nodes_path).toarray().astype(bool)
        else:
            for i in tqdm(range(self.x.shape[0]), desc='Processing influence nodes'):
                start_time = time.time()
                self.find_influence_nodes(i)
                run_time += time.time() - start_time
            scipy.sparse.save_npz(influence_nodes_path, scipy.sparse.coo_matrix(self.influence_matrix, dtype=bool))
            print('Runtime: {:.1f}'.format(run_time // 0.01 * 0.01))

        return run_time

    def update_influence_value(self, adj_values=None):
        self.influence_values = adj_values

    def get_overlap_marginal_value_pair(self, target_idx: int, subj_idx: int):
        overlap_mask = self.influence_matrix[target_idx] & self.influence_matrix[subj_idx]

        # we use a computational subgraph rather than the whole graph
        union_index, edge_index, _, edge_mask = k_hop_subgraph(
            [int(target_idx), int(subj_idx)], self.model.num_layers + 1, self.edge_index, relabel_nodes=True)
        union_mask = np.zeros(self.x.shape[0], dtype=bool)
        union_mask[union_index] = 1
        node_dict = dict(zip(np.flatnonzero(union_mask), range(union_index.shape[0])))

        x = self.x.clone()
        if np.count_nonzero(overlap_mask) > 0:
            x[overlap_mask] = 0
        x_plus = x[union_mask]
        x_minus_subj = x_plus.clone()
        x_minus_target = x_plus.clone()
        x_minus_subj[node_dict[subj_idx]] = 0
        x_minus_target[node_dict[target_idx]] = 0

        # we calculate pair-wise influence values
        out_plus = self.model(x_plus, edge_index)
        pred_plus = F.softmax(out_plus, dim=-1)
        out_minus_subj = self.model(x_minus_subj, edge_index)
        pred_minus_subj = F.softmax(out_minus_subj, dim=-1)
        out_minus_target = self.model(x_minus_target, edge_index)
        pred_minus_target = F.softmax(out_minus_target, dim=-1)

        phi_subj_to_target = pred_plus[node_dict[target_idx]] - pred_minus_subj[node_dict[target_idx]]
        phi_target_to_subj = pred_plus[node_dict[subj_idx]] - pred_minus_target[node_dict[subj_idx]]

        # update influence values
        self.influence_values[subj_idx, target_idx] = torch.linalg.vector_norm(phi_target_to_subj, ord=2, dim=-1).item()
        self.influence_values[target_idx, subj_idx] = torch.linalg.vector_norm(phi_subj_to_target, ord=2, dim=-1).item()

    def get_overlap_marginal_value_from_target(self, target_idx: int):
        influence_mask_target = self.influence_matrix[target_idx]
        influence_nodes = np.flatnonzero(influence_mask_target)
        num_influence_nodes = np.count_nonzero(influence_mask_target)
        if num_influence_nodes == 0:
            return
        elif num_influence_nodes <= 2:
            for subj_idx in influence_nodes:
                if self.influence_values[subj_idx, target_idx] > 0:
                    continue
                self.get_overlap_marginal_value_pair(target_idx, subj_idx)
            return

        '''sampling'''
        # influence_mask_subj = self.influence_matrix[influence_mask_target]
        # overlap_mask = influence_mask_target & influence_mask_subj
        # num_overlaps = np.count_nonzero(overlap_mask, axis=-1)
        # avg_overlap = np.ceil(num_overlaps.mean()).astype(int)
        avg_overlap = np.ceil(np.sqrt(num_influence_nodes)).astype(int)
        avg_overlaps = [avg_overlap - 1, avg_overlap]  # prevent fixed groups

        # the smallest number of samplings
        sample = self.sample
        if self.sample is None:
            sample = np.max([avg_overlap, 5])
        # adjust the number of sampled nodes
        if sample * (num_influence_nodes - avg_overlap) <= num_influence_nodes:
            avg_overlap = np.floor(num_influence_nodes - num_influence_nodes / sample).astype(int)
            avg_overlaps = [avg_overlap - 1, avg_overlap]

        # union_index: new_index -> old_index
        union_index, edge_index, target_inv, edge_mask = k_hop_subgraph(target_idx.item(), self.model.num_layers * 2 + 1,
                                                                        self.edge_index, relabel_nodes=True)
        union_mask = np.zeros(self.x.shape[0], dtype=bool)
        union_mask[union_index] = 1
        x = self.x[union_mask]
        node_dict = dict(zip(np.flatnonzero(union_mask), range(union_index.shape[0])))
        # influence_nodes_new: new_index of influence nodes
        influence_nodes_new = np.array([node_dict[k] for k in influence_nodes])

        phi_sum = np.zeros(num_influence_nodes)
        # sample_times: stores how many times a node can be sampled -> for fair sampling
        sample_times = np.full(num_influence_nodes, sample)
        next_idx = rng.choice(num_influence_nodes, avg_overlap, replace=False, shuffle=False)
        for i in range(sample):
            sample_idx = next_idx  # indices within influence nodes
            sample_times[sample_idx] -= 1
            sample_nodes = influence_nodes_new[sample_idx]  # new indices of influence nodes

            x_plus = x.clone()
            x_plus[sample_nodes] = 0
            x_minus = x_plus.clone()
            x_minus[target_inv] = 0

            '''calculate outgoing influence values'''
            out_plus = self.model(x_plus, edge_index)
            pred_plus = F.softmax(out_plus, dim=-1)
            out_minus = self.model(x_minus, edge_index)
            pred_minus = F.softmax(out_minus, dim=-1)
            phi = pred_plus - pred_minus
            phi = torch.linalg.vector_norm(phi, ord=2, dim=-1).detach().numpy()
            phi[sample_nodes] = 0
            phi_sum += phi[influence_nodes_new]

            '''for fair sampling'''
            avg_overlap = avg_overlaps[i % 2]
            avoid_idx = np.where(sample_times == sample_times.min(axis=None))[0]
            # try to sample nodes that are not sampled much
            if len(avoid_idx) > num_influence_nodes - avg_overlap:
                avoid_idx = rng.choice(avoid_idx, num_influence_nodes - avg_overlap, replace=False, shuffle=False)
            p = sample_times.copy()
            p[avoid_idx] = 0
            p = p / p.sum()
            next_idx = rng.choice(num_influence_nodes, avg_overlap, replace=False, p=p, shuffle=False)

        phi = phi_sum / sample_times
        self.influence_values[influence_nodes, target_idx] = phi

    def get_influence_value_efficient(self, target_idx: int):
        if np.count_nonzero(self.influence_matrix[target_idx]) == 0:
            raise Exception("There are nodes without edges.")
        if self.influence_values[:, target_idx].count_nonzero() == 0:
            self.get_overlap_marginal_value_from_target(target_idx)
        return self.influence_values[:, target_idx]

    def get_influence_value(self, target_idx: int):
        influence_nodes = np.flatnonzero(self.influence_matrix[target_idx])
        for node_idx in influence_nodes:
            if self.influence_values[target_idx, node_idx] == 0:
                self.get_overlap_marginal_value_pair(target_idx, node_idx)
        return self.influence_values[target_idx]


def auc_node(edge_index: Adj, target_idx: int, influence_nodes: list, influence_values: Tensor):
    neighbor, _, _, _ = k_hop_subgraph(node_idx=target_idx, num_hops=1, edge_index=edge_index)
    neighbor = [n for n in neighbor.tolist() if n != target_idx]
    if len(neighbor) == len(influence_nodes):
        return 1
    y_pred = influence_values.detach().numpy()
    y = np.zeros_like(y_pred, dtype=np.int)
    j = 0
    for i in range(len(influence_nodes)):
        if influence_nodes[i] == neighbor[j]:
            y[i] = 1
            j += 1
            if j == len(neighbor):
                break
    return roc_auc_score(y, y_pred)
