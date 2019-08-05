#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
- The idea behind this approach is to represent each molecule as a graph with atoms as nodes and bonds as edges.
- Each node will have an initial feature vector modeling its elemental and neighborhood properties (e.g. ACSF).
- The edges will be represented as an adjacency matrix (which can optionally be weighted by the bond order)
- Different flavors of GCN models will be applied to the graph. The scalar coupling constant can be predicted
using the final node representations by using DistMult (a bilinear scoring function).
- If training on all of the data at once, it would make sense to have separate DistMult matrices for each coupling type.
"""
import os
import numpy as np
import pandas as pd
from dtsckit.utils import read_pickle

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, GCNConv, APPNP, GATConv


########################################################################################################################
#                                               Feature generation
########################################################################################################################


def create_local_molecule_descriptors(molecule_map,
                                      base_atomic_features=None,
                                      acsf=None,
                                      soap=None,
                                      lmbtr=None):
    molecule_names = []
    molecule_sizes = []
    for name, molecule in molecule_map.items():
        if 'scalar' not in name:
            molecule_names.append(name)
            molecule_sizes.append(len(molecule['symbols']))

    local_descriptors = []
    if base_atomic_features is not None:
        print('Calculating the base atomic features...')
        local_descriptors.append(base_atomic_features(molecule_names))
    if acsf is not None:
        print('Calculating the ACSF features...')
        local_descriptors.append(acsf(molecule_names))
    if soap is not None:
        print('Calculating the SOAP features...')
        local_descriptors.append(soap(molecule_names))
    if lmbtr is not None:
        print('Calculating the LMBTR features...')
        local_descriptors.append(lmbtr(molecule_names))

    local_descriptors = np.concatenate(local_descriptors, axis=1)
    local_descriptors = torch.Tensor(local_descriptors)

    i = 0
    local_descriptor_map = dict()
    for name, size in zip(molecule_names, molecule_sizes):
        local_descriptor_map[name] = local_descriptors[i:i+size]
        i += size

    return local_descriptor_map


########################################################################################################################
#                                                  Define the models
########################################################################################################################

class GCNSequential(nn.Sequential):
    """A subclass of Sequential that extends the functionality for pytorch geometric layers"""

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(module, MessagePassing):
                x = module(*inputs)
            else:
                x = module(inputs[0])
            inputs = (x, *inputs[1:])
        return inputs[0]


class NodePairScorer(nn.Module):
    """An abstract class defining the infrastructure for the GCN models to predict node pair scores

    Models that extends this class should generally be graph convolution models. Once the GCN models are done updating
    the node representations, node pair scores will be calculated using the bilinear DistMult approach.

    Parameters
    ----------
    input_dim: int
        The initial dimension of the node representations.
    hidden_dim: int
        The dimension of the node representation in the hidden layers.
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.map_input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.distmult = nn.Parameter(torch.rand(1, hidden_dim))
        self.gcn = None  # this attribute should be extended with the appropriate graph convolutions

    def score_node_pairs(self, x_i, x_j):
        """Calculate the score between pairs of nodes

        Parameters
        ----------
        x_i: torch.Tensor
            The representation of the first element in the pairs of nodes with shape [num_pairs, hidden_dim]
        x_j: torch.Tensor
            The representation of the second element in the pairs of nodes with shape [num_pairs, hidden_dim]

        Returns
        -------
        torch.Tensor
            The scores of each node pair as a tensor with shape [num_pairs]
        """
        num_pairs = x_i.shape[0]
        d = torch.repeat_interleave(self.distmult, num_pairs, 0)
        return torch.sum(d * x_i * x_j, dim=1)

    def forward(self, x, edge_index, node_i, node_j):
        """Run the model (should not be directly called, only extended)

        Parameters
        ----------
        x: torch.Tensor
            The initial node representations with shape [num_nodes, input_dim]
        edge_index: torch.LongTensor
            The edge index with shape [2, num_edges] where the first row defines the source nodes
            and the second row defines the destination nodes.
        node_i: torch.LongTensor
            A tensor with shape [num_pairs] defining the index of the first node in each of the node pairs.
        node_j: torch.LongTensor
            A tensor with shape [num_pairs] defining the index of the second node in each of the node pairs.

        Returns
        -------
        torch.Tensor
            A tensor of shape [num_pairs] containing the scores of each node pair.
        """
        x = self.map_input(x)
        x = self.gcn(x, edge_index)
        scores = self.score_node_pairs(x[node_i], x[node_j])
        return scores


class GCNConvNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__(input_dim, hidden_dim)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            gcn_layers.append(nn.ReLU())
        gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn = GCNSequential(*gcn_layers)


class APPNPNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, k=10, alpha=0.1):
        super().__init__(input_dim, hidden_dim)
        self.k = k
        self.alpha = alpha
        self.gcn = APPNP(K=k, alpha=alpha)


class GATNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, num_layers=1, heads=1, concat_heads=True, leaky_slope=0.2, dropout=0,
                 update_activation=True):
        super().__init__(input_dim, hidden_dim)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(GATConv(hidden_dim, hidden_dim, heads, concat_heads, leaky_slope, dropout))
            if update_activation:
                gcn_layers.append(nn.ReLU())
        gcn_layers.append(GATConv(hidden_dim, hidden_dim, heads, concat_heads, leaky_slope, dropout))
        self.gcn = GCNSequential(*gcn_layers)


########################################################################################################################
#                                              Define the training pipeline
########################################################################################################################


def update_batch_atom_pairs(batch):
    """Update atom pairs index in the Data batch

    Parameters
    ----------
    batch: Data
        A pytorch geometric Data instance containing a batch of graphs stacked as a large, unconnected graph.
    """
    atom_pairs = batch['atom_pairs']
    num_couplings = batch['num_couplings']
    num_atoms = batch['num_atoms']

    coupling_lens = [0] + torch.cumsum(num_couplings, 0).tolist()
    offsets = [0] + torch.cumsum(num_atoms, 0)[:-1].tolist()

    for offset, (i, j) in zip(offsets, zip(coupling_lens[:-1], coupling_lens[1:])):
        atom_pairs[i:j] += offset

    batch['atom_pairs'] = atom_pairs


if __name__ == '__main__':
    coupling_type = '1JHN'
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules'
    data = pd.read_csv(os.path.join(ROOT_DIR, f'train/data_{coupling_type}.csv'))
    molecule_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))
