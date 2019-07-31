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
import torch
import torch.nn as nn


class NodePairScorer(nn.Module):
    """An abstract class defining the infrastructure for the GCN models to predict node pair scores

    Models that extends this class should generally be graph convolution models.

    Parameters
    ----------
    hidden_dim: int
        The dimension of the node representation in the hidden layers.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dist_mult = nn.Parameter(torch.rand(1, hidden_dim))

    def score_node_pairs(self, x_i, x_j):
        """Calculate the score between pairs of nodes

        Parameters
        ----------
        x_i: torch.Tensor
            The representation of the first element in the pairs of nodes with shape [batch_size, hidden_dim]
        x_j: torch.Tensor
            The representation of the second element in the pairs of nodes with shape [batch_size, hidden_dim]

        Returns
        -------
        torch.Tensor
            The scores of each node pair as a tensor with shape [batch_size]
        """
        batch_size = x_i.shape[0]
        d = torch.repeat_interleave(self.dist_mult, batch_size, 0)
        return torch.sum(d * x_i * x_j, dim=1)

    def forward(self, x, edge_index, node_i, node_j):
        raise NotImplementedError


if __name__ == '__main__':
    print()
