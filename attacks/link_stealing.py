import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.typing import Adj

from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import time
import os


class LinkStealing:

    def __init__(self, model: torch.nn.Module, edge_index: Adj, x: Tensor, pos_type: int = 0):
        self.model = model
        self.edge_index = edge_index
        self.x = x
        self.influence_matrix = np.zeros((self.x.shape[0], self.x.shape[0]), dtype=bool)
        self.influence_values = sp.dok_matrix((self.x.shape[0], self.x.shape[0]), dtype=float)

        out = self.model(self.x, self.edge_index)
        self.pred = F.softmax(out, dim=-1)

        if pos_type == 0:
            self.pos = self.pred
            self.pos_mean = torch.mean(self.pos, dim=0)
        elif pos_type == 1:
            self.pos = self.x
            self.pos_mean = torch.mean(self.pos, dim=0)
        else:
            raise ValueError(f'pos_type={pos_type} error')

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

    def update_influence_matrix(self, src_dir, defense_ep: float = None, erase: bool = False):
        run_time = 0
        if defense_ep is not None:
            influence_nodes_path = os.path.join(src_dir, 'influence_matrix_' + str(self.model.num_layers)
                                                + '_ep%.1f.npz' % defense_ep)
        else:
            influence_nodes_path = os.path.join(src_dir, 'influence_matrix_' + str(self.model.num_layers) + '.npz')
        if os.path.exists(influence_nodes_path) and not erase:
            self.influence_matrix = sp.load_npz(influence_nodes_path).toarray().astype(bool)
        else:
            for i in tqdm(range(self.x.shape[0]), desc='Processing influence nodes'):
                start_time = time.time()
                self.find_influence_nodes(i)
                run_time += time.time() - start_time
            sp.save_npz(influence_nodes_path, sp.coo_matrix(self.influence_matrix, dtype=bool))
            print('Runtime: {:.1f}'.format(run_time // 0.01 * 0.01))
        return run_time

    def update_influence_value(self, adj_values):
        self.influence_values = adj_values

    def get_influence_value(self, target_idx: int):
        for node_idx in range(self.x.shape[0]):
            if self.influence_matrix[target_idx, node_idx] == 0 or self.influence_values[target_idx, node_idx] != 0:
                continue
            d_0 = self.pos[target_idx] - self.pos_mean
            d_1 = self.pos[node_idx] - self.pos_mean
            dist = torch.dot(d_0, d_1) / torch.linalg.norm(d_0) / torch.linalg.norm(d_1)
            self.influence_values[target_idx, node_idx] = dist.item()
            self.influence_values[node_idx, target_idx] = dist.item()
            # dist = correlation(self.pos[target_idx].detach().numpy(),
            #                    self.pos[node_idx].detach().numpy())
            # self.influence_values[target_idx, node_idx] = dist
            # self.influence_values[node_idx, target_idx] = dist
        return self.influence_values[target_idx]