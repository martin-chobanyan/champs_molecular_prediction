#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A pytorch version of the Chainer Schnet code"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler
from features import encode_element


########################################################################################################################
#                                               Define the Modules
########################################################################################################################


class GraphLinear(nn.Linear):
    """3D input, apply Linear to the third axis of x with shape [batch_size, molecule_idx, atom_idx]"""

    def forward(self, x):
        s0, s1, s2 = x.size()
        x = x.view(s0 * s1, s2)
        x = super().forward(x)
        x = x.view(s0, s1, self.out_features)
        return x


class GraphBatchNorm(nn.BatchNorm1d):
    """x is a 3D tensor"""

    def forward(self, x):
        s0, s1, s2 = x.size()
        x = x.view((s0 * s1), s2)
        x = super().forward(x)
        x = x.view(s0, s1, s2)
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

    def forward(self, h, dist):
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


class SchNetUpdate(nn.Module):
    def __init__(self, input_dim=None, hidden_dim=64, bnorm=False):
        super().__init__()
        input_dim = hidden_dim if input_dim is None else input_dim
        self.linear1 = GraphLinear(hidden_dim, hidden_dim)
        self.linear2 = GraphLinear(hidden_dim, hidden_dim)
        self.linear3 = GraphLinear(hidden_dim, hidden_dim)
        self.cfconv = CFConv(hidden_dim=hidden_dim)
        self.bnorm = GraphBatchNorm(hidden_dim) if bnorm else None
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

    def forward(self, x, dist):
        v = self.linear1(x)
        v = self.cfconv(v, dist)
        v = F.softplus(self.linear2(v))
        v = self.linear3(v)
        v = self.bnorm(v) if self.bnorm else v
        return x + v


class SchNet(nn.Module):
    """Custom SchNet defined for predicting the decomposed coupling constant"""

    def __init__(self, input_dim=10, hidden_dim=512, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.gn = GraphLinear(input_dim, hidden_dim)

        self.schnet_list = nn.ModuleList()
        for l in range(self.num_layers):
            self.schnet_list.append(SchNetUpdate(hidden_dim=hidden_dim, bnorm=True))

        self.interaction1 = nn.Linear(2 * (input_dim + hidden_dim) + 1, 128)
        self.interaction2 = nn.Linear(128, 128)
        self.interaction3 = nn.Linear(128, 4)

    def forward(self, input_array, dists, pairs_index):
        """
        Parameters
        ----------
        input_array: [batch size, number of atoms, feature dim]
        dists: [batch size, number of atoms, number of atoms]
        pairs_index: [num couplings, 3]
        """
        h = self.gn(input_array)
        for update in self.schnet_list:
            h = update(h, dists)
        h = torch.cat([h, input_array], dim=2)

        x = torch.cat([h[pairs_index[:, 0], pairs_index[:, 1], :],
                       h[pairs_index[:, 0], pairs_index[:, 2], :],
                       dists[pairs_index[:, 0], pairs_index[:, 1], pairs_index[:, 2]].unsqueeze(1)],
                      dim=1)

        x = F.leaky_relu(self.interaction1(x))
        x = F.leaky_relu(self.interaction2(x))
        x = self.interaction3(x)
        return x


########################################################################################################################
#                                               Define the DataLoader
########################################################################################################################


class MoleculeGraph(Dataset):
    def __init__(self, molecule_map, pair_map, coupling_map):
        super().__init__()
        self.molecule_names = list(molecule_map.keys())
        self.molecule_map = molecule_map
        self.pair_map = pair_map
        self.coupling_map = coupling_map

    def __create_features(self, molecule):
        edge_index = molecule['bonds']
        atoms = molecule['symbols']

        # onehot encode the atomic symbols in the molecule
        one_hot = torch.Tensor([encode_element(a) for a in atoms])

        # onehot encode the node degrees
        degrees = torch.LongTensor([torch.sum(edge_index[0] == atom_i) for atom_i in range(len(atoms))])
        bonds = torch.eye(5)[degrees - 1]

        # concatenate the two onehot encodings into a feature matrix
        return torch.cat([one_hot, bonds], dim=1)

    def __getitem__(self, idx):
        molecule_name = self.molecule_names[idx]
        molecule = self.molecule_map[molecule_name]

        # get the feature matrix
        x = self.__create_features(molecule)

        # get the distance matrix
        distances = torch.Tensor(molecule['distance_matrix'])

        # get the atom pairs and molecule index
        pairs = torch.LongTensor(self.pair_map[molecule_name])
        mol_idx = torch.LongTensor([idx] * len(pairs))
        pairs = torch.cat([mol_idx, pairs], dim=1)

        # get the target couplings
        couplings = torch.Tensor(self.coupling_map[molecule_name])
        couplings = couplings.view(-1, 1)

        return x, distances, pairs, couplings

    def __len__(self):
        return len(self.molecule_map)


class SameSizeSampler(Sampler):
    def __init__(self, dataset):
        super().__init__(dataset)






if __name__ == '__main__':
    # model = SchNet()
    # x = torch.rand(3, 4, 10)
    # d = torch.rand(3, 4, 4)
    # p = torch.LongTensor([[0, 1, 2], [0, 0, 2], [1, 1, 3], [1, 0, 1], [1, 1, 2], [2, 2, 3]])
    # y = model(x, d, p)
    # print(y.shape)

    # from dtsckit.utils import read_pickle
    # molecule_map = read_pickle('/home/mchobanyan/data/kaggle/molecules/molecular_structure_map.pkl')
    # count = 0
    # dataset = MoleculeGraph(molecule_map)
    # for name in molecule_map.keys():
    #     x, d = dataset[name]
    #     print(x)
    #     print(d)
    #     print('------------------------------')
    #     count += 1
    #     if count == 5:
    #         break
    print()

