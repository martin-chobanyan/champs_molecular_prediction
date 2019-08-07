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
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from dtsckit.model import AverageKeeper
from dtsckit.utils import read_pickle, write_pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import MessagePassing, GCNConv, APPNP, GATConv
from torch_geometric.data import Data, DataLoader

from features import ElementalFeatures, BaseAtomicFeatures, AtomCenteredSymmetryFeatures
from pairwise_predictions import process_filename


########################################################################################################################
#                                               Feature generation
########################################################################################################################


class MoleculeGraph(object):
    def __init__(self,
                 molecule_map,
                 base_atomic_features=None,
                 acsf=None,
                 soap=None,
                 lmbtr=None,
                 elemental_features=False):
        self.molecule_map = molecule_map
        self.element_features = ElementalFeatures() if elemental_features else None
        self.atomic_features = base_atomic_features
        self.acsf = acsf
        self.soap = soap
        self.lmbtr = lmbtr

    def __call__(self, name):
        features = []
        if self.element_features is not None:
            symbols = self.molecule_map[name]['symbols']
            elemental_features = [self.element_features(s) for s in symbols]
            features.append(torch.Tensor(elemental_features))

        if self.atomic_features is not None:
            features.append(torch.Tensor(self.atomic_features(name)))

        if self.acsf is not None:
            features.append(torch.Tensor(self.acsf(name)))

        if self.soap is not None:
            features.append(torch.Tensor(self.soap(name)))

        if self.lmbtr is not None:
            features.append(torch.Tensor(self.lmbtr(name)))

        node_features = torch.cat(features, dim=1)
        edge_index = self.molecule_map[name]['bonds']
        return node_features, edge_index


def create_molecule_graphs(molecule_map, generate_mol_graph):
    molecule_graphs = dict()
    for name in tqdm(molecule_map.keys()):
        if name != 'scalar_descriptor_keys':
            molecule_graphs[name] = generate_mol_graph(name)
    return molecule_graphs


class MoleculeData(object):
    def __init__(self, mol_graph, df):
        self.mol_graph = mol_graph
        self.df = df

    def __call__(self, name):
        x, edge_index = self.mol_graph[name]
        num_atoms = torch.LongTensor([x.shape[0]])

        molecule_rows = self.df.loc[self.df['molecule_name'] == name]
        atom_pairs = torch.LongTensor(molecule_rows[['atom_index_0', 'atom_index_1']].values)
        couplings = torch.FloatTensor(molecule_rows['scalar_coupling_constant'].values)
        num_couplings = torch.LongTensor([len(couplings)])

        data = Data(x=x, edge_index=edge_index,
                    atom_pairs=atom_pairs, couplings=couplings,
                    num_atoms=num_atoms, num_couplings=num_couplings)
        return data


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


def train_molecule_gcn_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    avg_keeper = AverageKeeper()
    for batch in dataloader:
        update_batch_atom_pairs(batch)
        node_features = batch.x.to(device)
        edge_index = batch.edge_index.to(device)
        atom_pairs = batch.atom_pairs.to(device)
        couplings = batch.couplings.to(device)

        optimizer.zero_grad()
        pred = model(node_features, edge_index, atom_pairs[:, 0], atom_pairs[:, 1])
        loss = criterion(pred, couplings)
        loss.backward()
        optimizer.step()
        avg_keeper.add(loss)

    avg_loss = avg_keeper.calculate()
    return avg_loss


def validate_molecule_gcn_epoch(model, dataloader, criterion, device):
    model.eval()
    avg_keeper = AverageKeeper()
    with torch.no_grad():
        for batch in dataloader:
            update_batch_atom_pairs(batch)
            node_features = batch.x.to(device)
            edge_index = batch.edge_index.to(device)
            atom_pairs = batch.atom_pairs.to(device)
            couplings = batch.couplings.to(device)

            pred = model(node_features, edge_index, atom_pairs[:, 0], atom_pairs[:, 1])
            loss = criterion(pred, couplings)
            avg_keeper.add(loss)

    avg_loss = avg_keeper.calculate()
    return avg_loss


if __name__ == '__main__':
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
    TEST_DIR = os.path.join(ROOT_DIR, 'test')
    MAX_MOL_SIZE = 29

    print('Reading the molecular structures...')
    molecule_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))
    num_molecules = len(molecule_map)

    submission_filepath = os.path.join(ROOT_DIR, 'submissions/gat_submission.csv')
    submission_df = pd.read_csv(os.path.join(ROOT_DIR, 'submissions/submission.csv'))
    submission_df['scalar_coupling_constant'] = 0
    submission_df.index = submission_df['id'].values

    ###################################### Define the local molecular descriptors ######################################
    r_c = 0.0
    for name in molecule_map:
        if 'scalar' not in name:
            distances = molecule_map[name]['distance_matrix']
            r_c = max(r_c, distances.max())

    # Configure the "Atom-centered symmetry functions"

    # G2 - eta/Rs couples:
    g2_params = []
    for eta in [0.01, 0.1, 0.5, 1]:
        for rs in [2, 6]:
            g2_params.append((eta, rs))

    # G4 - eta/ksi/lambda triplets:
    g4_params = []
    for eta in [0.01, 0.1, 0.5, 1]:
        for zeta in [4]:
            for lambda_exp in [-1, 1]:
                g4_params.append((eta, zeta, lambda_exp))

    base_features = BaseAtomicFeatures(molecule_map)
    acsf = AtomCenteredSymmetryFeatures(molecule_map, r_c, g2_params=g2_params, g4_params=g4_params)
    mol_graph = MoleculeGraph(molecule_map, base_features, acsf)

    print('Creating the molecule features:')
    mol_graphs = create_molecule_graphs(molecule_map, mol_graph)

    num_features = mol_graphs['dsgdb9nsd_000011'][0].shape[1]
    print(f'number of features: {num_features}')
    ####################################################################################################################

    models = dict()
    scores = dict()
    feature_importances = dict()
    total_time = 0
    for filename in os.listdir(TRAIN_DIR):
        ######################################## Load the training data ################################################
        start_time = time.time()
        coupling_type, num_hops, h2h = process_filename(filename)
        print(f'\nTraining model for {coupling_type}')
        train_df = pd.read_csv(os.path.join(TRAIN_DIR, filename))
        train_df = train_df[['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]

        standardize = StandardScaler()
        train_df['scalar_coupling_constant'] = standardize.fit_transform(
            train_df['scalar_coupling_constant'].values.reshape(-1, 1)
        ).squeeze()

        print(f'Coupling average:\t{standardize.mean_[0]}')
        print(f'Coupling stdev:\t\t{np.sqrt(standardize.var_[0])}')

        ######################################## Create the dataloader #################################################
        graphs = MoleculeData(mol_graphs, train_df)
        molecule_names = train_df['molecule_name'].unique()

        data_list = []
        for name in tqdm(molecule_names):
            data_list.append(graphs(name))

        dataloader = DataLoader(data_list, batch_size=128, shuffle=True)

        ##################################### Define and train the model ###############################################
        num_layers = 3
        hidden_dim = 300
        device = torch.device('cuda')
        model = GATNodePairScorer(num_features, hidden_dim, num_layers, heads=8, concat_heads=False)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.0005)

        num_epochs = 80
        train_losses = []
        val_losses = []
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            train_loss = train_molecule_gcn_epoch(model, dataloader, criterion, optimizer, device)
            print(f'Training loss:\t\t{train_loss}')
            train_losses.append(train_loss)

        models[coupling_type] = model
        ######################################## Make the predictions ##################################################
        test_df = pd.read_csv(os.path.join(TEST_DIR, filename))

        row_ids = []
        preds = []
        model.eval()
        molecule_groups = test_df.groupby('molecule_name')
        for name, molecule_rows in tqdm(molecule_groups):
            num_atoms = molecule_map[name]['rdkit'].GetNumAtoms()

            x, edge_index = mol_graph(name)
            idx, i, j = molecule_rows[['id', 'atom_index_0', 'atom_index_1']].values.T

            x = x.to(device)
            edge_index = edge_index.to(device)
            i = torch.LongTensor(i).to(device)
            j = torch.LongTensor(j).to(device)

            with torch.no_grad():
                pred = model(x, edge_index, i, j)

            row_ids.append(idx)
            preds.append(pred.cpu().numpy())

        row_ids = np.concatenate(row_ids, 0)
        preds = standardize.inverse_transform(np.concatenate(preds, 0).reshape(-1, 1)).squeeze()
        submission_df.loc[row_ids, 'scalar_coupling_constant'] = preds

        elapsed_time = (time.time() - start_time) / 3600
        print(f'Time elapsed: {elapsed_time} hours')
        total_time += elapsed_time
        ################################################################################################################

    print(f'\nTotal time elapsed: {total_time} hours')
    print('\nSaving the submissions...')
    write_pickle(models, os.path.join(ROOT_DIR, 'models/pairwise/gat_models.pkl'))
    submission_df.to_csv(submission_filepath, index=False)
    print('Done!')
