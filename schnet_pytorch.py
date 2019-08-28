#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A pytorch version of the Chainer Schnet code"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphLinear(nn.Linear):
    """3D input, apply Linear to the third axis of x with shape [batch_size, molecule_idx, atom_idx]"""
    def __call__(self, x):
        s0, s1, s2 = x.size()
        x = x.view(s0*s1, s2)
        x = super().__call__(x)
        x = x.view(s0, s1, self.out_features)
        return x


class CFConv(nn.Module):
    """Continuous filter convolution"""
    def __init__(self, num_rbf=300, radius_resolution=0.1, gamma=10.0, hidden_dim=64):
        super().__init__()
        self.dense1 = nn.Linear(num_rbf, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_dim = hidden_dim
        self.num_rbf = num_rbf
        self.radius_resolution = radius_resolution
        self.gamma = gamma

    def __call__(self, h, dist):
        """
        Parameters
        ----------
        h: torch.Tensor
            A tensor with shape [batch_idx, atom_idx, n_features]
        dist: torch.Tensor
            A tensor with shape [batch_idx, atom_idx, atom_idx] representing interatomic distances
        """
        mb, atom, ch = h.size()
        if ch != self.hidden_dim:
            raise ValueError(f'The last dimension of the input tensor "h" ({ch}) '
                             f'does not match the hidden dim ({self.hidden_dim})')

        embedlist = torch.arange(self.num_rbf, dtype=torch.float32) * self.radius_resolution
        dist = dist.view(mb, atom, atom, 1)
        dist = torch.repeat_interleave(dist, self.num_rbf, dim=-1)
        dist = torch.exp(-self.gamma * (dist - embedlist) ** 2)
        dist = dist.view(-1, self.num_rbf)
        dist = F.softplus(self.dense1(dist))
        dist = F.softplus(self.dense2(dist))
        dist = dist.view(mb, atom, atom, self.hidden_dim)
        h = h.view(mb, atom, 1, self.hidden_dim)
        h = torch.repeat_interleave(h, atom, dim=2)
        h = torch.sum(h * dist, dim=1)
        return h
