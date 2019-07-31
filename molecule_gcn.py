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
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, APPNP


# TODO: update this so that it works with batches of graphs
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
        self.graph_convs = None  # this attribute should be extended with the appropriate list of graph convolutions

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
        if self.graph_convs is None:
            raise ValueError("Subclass of NodePairScorer must define a 'graph_convs'"
                             " attribute as a ModuleList of graph convolutions")
        x = self.map_input(x)
        for gcn in self.graph_convs:
            x = gcn(x, edge_index)
        scores = self.score_node_pairs(x[node_i], x[node_j])
        return scores


class GCNConvNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, num_layers=1):
        super().__init__(input_dim, hidden_dim)
        self.graph_convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])


class APPNPNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, k=10, alpha=0.1, num_layers=1):
        super().__init__(input_dim, hidden_dim)
        self.k = k
        self.alpha = alpha
        self.graph_convs = nn.ModuleList([APPNP(k, alpha) for _ in range(num_layers)])


if __name__ == '__main__':
    X = torch.rand(4, 5)
    e = [[0, 0, 0, 1, 1, 2, 2, 3], [1, 2, 3, 0, 2, 1, 3, 2]]
    a = torch.LongTensor(e)
    model = GCNConvNodePairScorer(5, 3, 2)
    print(model(X, a, torch.LongTensor([0, 0, 2]), torch.LongTensor([1, 2, 3])))
