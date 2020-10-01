#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Description:
- The idea behind this approach is to represent each molecule as a graph with atoms as nodes and bonds as edges.
- Each node will have an initial feature vector modeling its local atomic environment
- These feature vectors can be constructed using chemical descriptors such as:
    - ACSF (Atom-Centered Symmetry Functions)
    - LMBTR (Local Many-Body Tensor Representation)
    - SOAP (Smooth Overlap of Atomic Positions)
- Different flavors of GCN models are provided to apply on the graph. The scalar coupling constant can be predicted by
passing the final node representations to a bilinear scoring functions (e.g. DistMult)
"""
import os
import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from misc import AverageKeeper, read_pickle, write_pickle

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


# TODO: update this so that it uses the base features + an optional descriptor
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
    """Generates the molecular graphs and stores the results in a dictionary

    Parameters
    ----------
    molecule_map: dict
        A dictionary mapping each molecule string id to a dictionary containing its properties
    generate_mol_graph: callable
        A callable object that returns the graph representation of the molecule (nodes and edges)

    Returns
    -------
    molecule_graphs: dict
        A dictionary mapping the molecule string id to its graph representation
    """
    molecule_graphs = dict()
    for name in tqdm(molecule_map.keys()):
        molecule_graphs[name] = generate_mol_graph(name)
    return molecule_graphs


# TODO: modify and document
class MoleculeData:
    """This class collects the input features for the GCN model along with the target values"""
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

        return Data(x=x,
                    edge_index=edge_index,
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
    distmult: bool, optional
        If True then DistMult is used as the scoring function.
        If False, a two layer feedforward net is used with leaky ReLU (default=False).
    dropout: float, optional
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
            # initialize a learnable DistMult diagonal
            self.scorer = nn.Parameter(torch.rand(1, hidden_dim))
        else:
            # initialize a bilinear network that takes two concatenated atomic representations and maps them to a score
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

    def forward(self, x, edge_index, node_i, node_j):
        """
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
    """A node pair scorer using the vanilla graph convolution model"""
    def __init__(self, input_dim, hidden_dim, distmult=False, num_layers=1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
            gcn_layers.append(nn.LeakyReLU())
        gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        self.gcn = GCNSequential(*gcn_layers)


class APPNPNodePairScorer(NodePairScorer):
    """A node pair scorer using the "Approximate Personalized Propagation of Neural Predictions" as the gcn module"""
    def __init__(self, input_dim, hidden_dim, distmult=False, k=10, alpha=0.1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        self.gcn = APPNP(K=k, alpha=alpha)


class GATNodePairScorer(NodePairScorer):
    """A node pair scorer using Graph Attention Networks as the gcn module

    Parameters
    ----------
    input_dim: int
    hidden_dim: int
    distmult: bool
    num_layers: int, optional
    heads: int, optional
        The number of heads to use in the multi-head attention mechanism in GAT (default=1)
    concat_heads: bool, optional
        If True, then the heads are concatenated. If False, then the heads are averaged (default=True).
    dropout: float, optional
    """
    def __init__(self, input_dim, hidden_dim, distmult=False, num_layers=1, heads=1, concat_heads=True, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(GATConv(hidden_dim, hidden_dim, heads, concat_heads))
            gcn_layers.append(nn.LeakyReLU())
        gcn_layers.append(GATConv(hidden_dim, hidden_dim, heads, concat_heads))
        self.gcn = GCNSequential(*gcn_layers)


class NNConvNodePairScorer(NodePairScorer):
    """Uses the NNConv as the gcn module

    Note: this flavor of GCN takes an input of edge features and passes them through the 'edge_net' module to create
    the convolutional filters. The goal was to use the bond properties (e.g. distance, bond order), but the results
    did not beat EdgeConvNodePairScorer. NodePairScorer no longer supports NNConv in order to make the code simpler.
    """
    def __init__(self, input_dim, hidden_dim, edge_net, distmult=False, num_layers=1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers - 1):
            gcn_layers.append(NNConv(hidden_dim, hidden_dim, nn=edge_net))
            gcn_layers.append(nn.LeakyReLU())
        self.gcn = GCNSequential(*gcn_layers)


class EdgeConvNodePairScorer(NodePairScorer):
    """A node pair scorer that uses EdgeConv as its gcn module

    Parameters
    ----------
    input_dim: int
    hidden_dim: int
    distmult: bool
    num_layers: int, optional
    node_pair_net: nn.Module
        A pytorch model mapping [*, 2*hidden_dim] to [*, hidden_dim]
    dropout: float, optional
    """
    def __init__(self, input_dim, hidden_dim, node_pair_net, distmult=False, num_layers=1, dropout=0):
        super().__init__(input_dim, hidden_dim, distmult, dropout)
        gcn_layers = []
        for _ in range(num_layers):
            gcn_layers.append(EdgeConv(nn=node_pair_net))
            gcn_layers.append(nn.BatchNorm1d(hidden_dim))
            gcn_layers.append(nn.LeakyReLU())
        self.gcn = GCNSequential(*gcn_layers)


class NodePairNet(nn.Module):
    """Base node pair module for EdgeConvNodePairScorer"""
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
    """Train a NodePairScorer for a single epoch

    Parameters
    ----------
    model: NodePairScorer
    dataloader: DataLoader
    criterion: loss function
    optimizer: pytorch optimizer
    device: torch.device

    Returns
    -------
    avg_loss: float
    """
    model.train()
    avg_keeper = AverageKeeper()
    for batch in dataloader:
        update_batch_atom_pairs(batch)
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch.x, batch.edge_index, batch.atom_pairs[:, 0], batch.atom_pairs[:, 1])
        loss = criterion(pred, batch.couplings)
        loss.backward()
        optimizer.step()
        avg_keeper.add(loss)
    avg_loss = avg_keeper.calculate()
    return avg_loss


def validate_molecule_gcn_epoch(model, dataloader, criterion, device):
    """Run the model on a validation set

    Parameters
    ----------
    model: NodePairScorer
    dataloader: DataLoader
    criterion: loss function
    device: torch.device

    Returns
    -------
    avg_loss: float
    """
    model.eval()
    avg_keeper = AverageKeeper()
    with torch.no_grad():
        for batch in dataloader:
            update_batch_atom_pairs(batch)
            batch = batch.to(device)
            pred = model(batch.x, batch.edge_index, batch.atom_pairs[:, 0], batch.atom_pairs[:, 1])
            loss = criterion(pred, batch.couplings)
            avg_keeper.add(loss)
    avg_loss = avg_keeper.calculate()
    return avg_loss


# TODO: add documentation
def early_stop(train_loader, eval_loader, model, optimizer, criterion, device, check=1, patience=16, max_epochs=160):
    """Performs early stopping for NodePairScorer models"""
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


def train_and_make_predictions(root_dir, model_name):
    train_dir = os.path.join(root_dir, 'train')
    test_dir = os.path.join(root_dir, 'test')
    submission_filepath = os.path.join(ROOT_DIR, f'submissions/{model_name}_submission.csv')
    submission_df = pd.read_csv(os.path.join(ROOT_DIR, 'submissions/submission.csv'))
    submission_df['scalar_coupling_constant'] = 0
    submission_df.index = submission_df['id'].values

    ###################################### Define the local molecular descriptors ######################################
    print('Creating the molecule features:')
    # TODO: update these conditions
    if 'acsf' in model_name:
        mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/graphs_acsf_835_features.pkl'))
        acsf_averages, acsf_std_devs = read_pickle('/home/mchobanyan/data/kaggle/molecules/graphs/acsf_standardize.pkl')
        eps = 0.0000001
        for molecule_name in tqdm(mol_graphs):
            x, edges = mol_graphs[molecule_name]
            x = (x - acsf_averages) / (acsf_std_devs + eps)
            mol_graphs[molecule_name] = x, edges
    elif 'lmbtr' in model_name:
        mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/graphs_lmbtr_1140_features.pkl'))
    elif 'soap' in model_name:
        mol_graphs = read_pickle(os.path.join(ROOT_DIR, 'graphs/graphs_standardized_soap_1050_noF.pkl'))
    else:
        raise ValueError('Wrong model name')

    num_features = mol_graphs['dsgdb9nsd_000011'][0].shape[1]
    print(f'Number of features: {num_features}')
    ####################################################################################################################
    models = dict()
    scores = dict()
    total_time = 0
    for filename in os.listdir(train_dir):
        ######################################## Load the training data ################################################
        start_time = time.time()
        coupling_type, *_ = process_filename(filename)
        print(f'\nTraining model for {coupling_type}')
        col_subset = ['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'scalar_coupling_constant']
        train_df = pd.read_csv(os.path.join(train_dir, filename))[col_subset]
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
        hidden_dim = 1000  # previously 1000

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

        num_epochs = 200  # 200 before
        for epoch in range(num_epochs):
            print(f'Epoch: {epoch}')
            if epoch % 10 == 0:
                write_pickle(model,
                             f'/home/mchobanyan/data/kaggle/molecules/models/fully_trained_gcn/{coupling_type}_{model_name}_epoch_{epoch}.pkl')
            train_loss = train_molecule_gcn_epoch(model, dataloader, criterion, optimizer, device)
            print(f'Training loss:\t\t{train_loss}')

        models[coupling_type] = model
        ######################################## Make the predictions ##################################################
        test_df = pd.read_csv(os.path.join(test_dir, filename))

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
    print(f'\nSaving the submissions to {submission_filepath}')
    write_pickle(models, os.path.join(ROOT_DIR, f'models/gcn/{model_name}_models.pkl'))
    write_pickle(scores, os.path.join(ROOT_DIR, f'models/gcn/{model_name}_scores.pkl'))
    submission_df.to_csv(submission_filepath, index=False)
    print('Done!')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    args = parser.parse_args()
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    train_and_make_predictions(ROOT_DIR, args.model_name)
