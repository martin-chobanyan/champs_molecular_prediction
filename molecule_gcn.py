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
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from dtsckit.model import AverageKeeper
from dtsckit.utils import read_pickle, write_pickle

import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.nn import MessagePassing, GCNConv, APPNP, GATConv, NNConv, EdgeConv
from torch_geometric.data import Data, DataLoader

from features import BaseAtomicFeatures, AtomCenteredSymmetryFeatures
from pairwise_predictions import process_filename


########################################################################################################################
#                                               Feature generation
########################################################################################################################


class MoleculeGraph:
    def __init__(self,
                 molecule_map,
                 base_atomic_features=None,
                 acsf=None,
                 soap=None,
                 lmbtr=None):
        self.molecule_map = molecule_map
        self.atomic_features = base_atomic_features
        self.acsf = acsf
        self.soap = soap
        self.lmbtr = lmbtr

    def __call__(self, name):
        features = []
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
    def __init__(self, mol_graph, df, edge_map=None):
        self.mol_graph = mol_graph
        self.edge_map = edge_map
        self.df = df

    def __call__(self, name):
        x, edge_index = self.mol_graph[name]
        edge_attr = torch.FloatTensor(self.edge_map(name)) if self.edge_map is not None else torch.FloatTensor([])
        num_atoms = torch.LongTensor([x.shape[0]])

        molecule_rows = self.df.loc[self.df['molecule_name'] == name]
        atom_pairs = torch.LongTensor(molecule_rows[['atom_index_0', 'atom_index_1']].values)
        couplings = torch.FloatTensor(molecule_rows['scalar_coupling_constant'].values)
        num_couplings = torch.LongTensor([len(couplings)])

        return Data(x=x,
                    edge_index=edge_index,
                    edge_attr=edge_attr,
                    atom_pairs=atom_pairs,
                    couplings=couplings,
                    num_atoms=num_atoms,
                    num_couplings=num_couplings)


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
    dropout: float
        The dropout probability to apply to the input fully connected layer (default=0; no dropout)
    """

    def __init__(self, input_dim, hidden_dim, distmult=False, dropout=0.0):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.map_input = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(), nn.Dropout(dropout))
        self.gcn = None  # this attribute should be extended with the appropriate graph convolutions
        self.distmult = distmult
        if distmult:
            self.scorer = nn.Parameter(torch.rand(1, hidden_dim))
        else:
            self.scorer = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.LeakyReLU(), nn.Linear(hidden_dim, 1))


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
        if self.distmult:
            num_pairs = x_i.shape[0]
            d = torch.repeat_interleave(self.scorer, num_pairs, 0)
            return torch.sum(d * x_i * x_j, dim=1)
        else:
            x = torch.cat([x_i, x_j], dim=1)
            return self.scorer(x).view(-1)

    def forward(self, x, edge_index, node_i, node_j, edge_attr=None):
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
        edge_attr: torch.Tensor
            A tensor with shape [num_edges, num_edge_features] containing the features of each edge.
            This should only be specified if the 'gcn' attribute uses edge labels (e.g. NNConv).

        Returns
        -------
        torch.Tensor
            A tensor of shape [num_pairs] containing the scores of each node pair.
        """
        x = self.map_input(x)
        if edge_attr is None:
            x = self.gcn(x, edge_index)
        else:
            x = self.gcn(x, edge_index, edge_attr)
        scores = self.score_node_pairs(x[node_i], x[node_j])
        return scores


class GCNConvNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, distmult=False, num_layers=1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            gcn_layers.append(nn.LeakyReLU())
        gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn = GCNSequential(*gcn_layers)


class APPNPNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, distmult=False, k=10, alpha=0.1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        self.gcn = APPNP(K=k, alpha=alpha)


class GATNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, distmult=False,
                 num_layers=1, heads=1, concat_heads=True, leaky_slope=0.2, gat_dropout=0,
                 update_activation=True, input_dropout=0):

        super().__init__(input_dim, hidden_dim, distmult, input_dropout)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(GATConv(hidden_dim, hidden_dim, heads, concat_heads, leaky_slope, gat_dropout))
            if update_activation:
                gcn_layers.append(nn.LeakyReLU())
        gcn_layers.append(GATConv(hidden_dim, hidden_dim, heads, concat_heads, leaky_slope, gat_dropout))
        self.gcn = GCNSequential(*gcn_layers)


class NNConvNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, edge_net, distmult=False, num_layers=1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(NNConv(hidden_dim, hidden_dim, nn=edge_net))
            gcn_layers.append(nn.LeakyReLU())
        self.gcn = GCNSequential(*gcn_layers)


class EdgeConvNodePairScorer(NodePairScorer):
    def __init__(self, input_dim, hidden_dim, node_pair_net, distmult=False, num_layers=1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers):
            gcn_layers.append(EdgeConv(nn=node_pair_net))
            gcn_layers.append(nn.BatchNorm1d(hidden_dim))
            gcn_layers.append(nn.LeakyReLU())
        self.gcn = GCNSequential(*gcn_layers)


class NodePairNet(nn.Module):
    def __init__(self, dim, dropout=0):
        super().__init__()
        self.fc1 = nn.Linear(2 * dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


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
    use_edge_attr = isinstance(model, NNConvNodePairScorer)
    for batch in dataloader:
        update_batch_atom_pairs(batch)
        batch = batch.to(device)
        optimizer.zero_grad()
        if use_edge_attr:
            pred = model(batch.x, batch.edge_index, batch.atom_pairs[:, 0], batch.atom_pairs[:, 1], batch.edge_attr)
        else:
            pred = model(batch.x, batch.edge_index, batch.atom_pairs[:, 0], batch.atom_pairs[:, 1])
        loss = criterion(pred, batch.couplings)
        loss.backward()
        optimizer.step()
        avg_keeper.add(loss)
    avg_loss = avg_keeper.calculate()
    return avg_loss


def validate_molecule_gcn_epoch(model, dataloader, criterion, device):
    model.eval()
    avg_keeper = AverageKeeper()
    use_edge_attr = isinstance(model, NNConvNodePairScorer)
    with torch.no_grad():
        for batch in dataloader:
            update_batch_atom_pairs(batch)
            batch = batch.to(device)
            if use_edge_attr:
                pred = model(batch.x, batch.edge_index, batch.atom_pairs[:, 0], batch.atom_pairs[:, 1], batch.edge_attr)
            else:
                pred = model(batch.x, batch.edge_index, batch.atom_pairs[:, 0], batch.atom_pairs[:, 1])
            loss = criterion(pred, batch.couplings)
            avg_keeper.add(loss)
    avg_loss = avg_keeper.calculate()
    return avg_loss


def early_stop(train_loader, eval_loader, model, optimizer, criterion, device, check=1, patience=16, max_epochs=160):
    epoch = 0
    p = 0
    best_validation_loss = float('inf')
    stop_epoch = 0

    training_losses = []
    validation_losses = []

    # while the validation loss has not consistently increased
    while p < patience:
        # train the model for check steps
        for i in range(check):
            if epoch == max_epochs:
                return stop_epoch, training_losses, validation_losses
            training_loss = train_molecule_gcn_epoch(model, train_loader, criterion, optimizer, device)
            print(f'Early stopping epoch {epoch}')
            print(f'Training loss:\t\t{training_loss}')
            training_losses.append(training_loss)
            epoch += 1

        # get the validation loss
        validation_loss = validate_molecule_gcn_epoch(model, eval_loader, criterion, device)
        validation_losses.append(validation_loss)
        print(f'Validation loss:\t{validation_loss}')

        if validation_loss < best_validation_loss:
            p = 0
            best_validation_loss = validation_loss
            # save the model and update the stopping epoch
            stop_epoch = epoch
        else:
            p += 1
    return stop_epoch, training_losses, validation_losses


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    # parser.add_argument('--epochs', type=int)
    args = parser.parse_args()
    model_name = args.model_name
    # num_epochs = args.epochs

    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
    TEST_DIR = os.path.join(ROOT_DIR, 'test')
    MAX_MOL_SIZE = 29

    submission_filepath = os.path.join(ROOT_DIR, f'submissions/{model_name}_submission.csv')
    submission_df = pd.read_csv(os.path.join(ROOT_DIR, 'submissions/submission.csv'))
    submission_df['scalar_coupling_constant'] = 0
    submission_df.index = submission_df['id'].values

    ###################################### Define the local molecular descriptors ######################################
    print('Creating the molecule features:')
    # mol_graphs = create_molecule_graphs(molecule_map, mol_graph)
    # mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/complete_graphs.pkl'))
    # mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/graphs_lmbtr_features.pkl'))

    if 'acsf' in model_name:
        mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/graphs_acsf_835_features.pkl'))
    elif 'lmbtr' in model_name:
        mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/graphs_lmbtr_1140_features.pkl'))
    else:
        raise ValueError('Wrong model name')

    num_features = mol_graphs['dsgdb9nsd_000011'][0].shape[1]
    print(f'Number of features: {num_features}')
    ####################################################################################################################
    models = dict()
    scores = dict()
    total_time = 0
    for filename in os.listdir(TRAIN_DIR):
        ######################################## Load the training data ################################################
        start_time = time.time()
        coupling_type, *_ = process_filename(filename)
        print(f'\nTraining model for {coupling_type}')
        col_subset = ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'scalar_coupling_constant']
        train_df = pd.read_csv(os.path.join(TRAIN_DIR, filename))[col_subset]
        print(train_df.shape)

        standardize = StandardScaler()
        train_df['scalar_coupling_constant'] = standardize.fit_transform(
            train_df['scalar_coupling_constant'].values.reshape(-1, 1)
        ).squeeze()

        print(f'Coupling average:\t{standardize.mean_[0]}')
        print(f'Coupling stdev:\t\t{np.sqrt(standardize.var_[0])}')

        ################################ Early stop to find the number of epochs #######################################
        print(f'Early stopping for {coupling_type}')
        graphs = MoleculeData(mol_graphs, train_df)

        molecule_names = train_df['molecule_name'].unique()
        names_train, names_val = train_test_split(molecule_names, train_size=0.8, shuffle=True, random_state=0)

        data_train = []
        for name in tqdm(names_train):
            data_train.append(graphs(name))

        data_val = []
        for name in tqdm(names_val):
            data_val.append(graphs(name))

        bsize = 512
        train_loader = DataLoader(data_train, batch_size=bsize, shuffle=True)
        val_loader = DataLoader(data_val, batch_size=bsize, shuffle=True)

        num_layers = 3
        hidden_dim = 1000  # previously 600

        # device = torch.device('cuda')
        # model = EdgeConvNodePairScorer(num_features, hidden_dim, NodePairNet(hidden_dim), False, num_layers, 0)
        # model = model.to(device)
        # criterion = nn.MSELoss()
        # optimizer = Adam(model.parameters(), lr=0.0005)

        # stop_epoch, training_losses, validation_losses = early_stop(train_loader, val_loader,
        #                                                             model, optimizer, criterion, device,
        #                                                             patience=18, max_epochs=200)

        # estop_scores = {'stop': stop_epoch, 'training_losses': training_losses, 'validation_losses': validation_losses}
        # scores[coupling_type] = estop_scores
        ######################################## Create the dataloader #################################################
        data_list = data_train + data_val
        dataloader = DataLoader(data_list, batch_size=bsize, shuffle=True)
        ##################################### Define and train the model ###############################################
        print(f'Training for {coupling_type}')

        device = torch.device('cuda')
        model = EdgeConvNodePairScorer(num_features, hidden_dim, NodePairNet(hidden_dim), False, num_layers, 0)
        model = model.to(device)
        criterion = nn.MSELoss()
        optimizer = Adam(model.parameters(), lr=0.0005)

        num_epochs = 200  # 100 before
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            if epoch % 10 == 0:
                write_pickle(model, f'/home/mchobanyan/data/kaggle/molecules/models/fully_trained_gcn/{coupling_type}_{model_name}_epoch_{epoch}.pkl')
            train_loss = train_molecule_gcn_epoch(model, dataloader, criterion, optimizer, device)
            print(f'Training loss:\t\t{train_loss}')

        models[coupling_type] = model
        ######################################## Make the predictions ##################################################
        test_df = pd.read_csv(os.path.join(TEST_DIR, filename))

        row_ids = []
        preds = []
        model.eval()
        molecule_groups = test_df.groupby('molecule_name')
        for name, molecule_rows in tqdm(molecule_groups):
            x, edges = mol_graphs[name]
            idx, i, j = molecule_rows[['id', 'atom_index_0', 'atom_index_1']].values.T

            x = x.to(device)
            edges = edges.to(device)
            i = torch.LongTensor(i).to(device)
            j = torch.LongTensor(j).to(device)

            with torch.no_grad():
                pred = model(x, edges, i, j)

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
    write_pickle(models, os.path.join(ROOT_DIR, f'models/gcn/{model_name}_models.pkl'))
    write_pickle(scores, os.path.join(ROOT_DIR, f'models/gcn/{model_name}_scores.pkl'))
    submission_df.to_csv(submission_filepath, index=False)
    print('Done!')
