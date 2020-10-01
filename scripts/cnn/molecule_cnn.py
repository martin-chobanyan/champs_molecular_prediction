#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file defines a standard convolutional approach to the problem:
- Represent each molecule as matrices where each matrix captures an atomic interaction feature
    e.g. atomic distance matrix and the Coulomb matrix
- Pad these matrices so that they are the same shape for each molecule
- Stack the matrices as channels of a "molecule volume"
- Apply a CNN on the molecule volumes and then linearly map the feature maps to the prediction matrix
- Extract the original non-padded shape from the prediction matrix and collect the appropriate coupling constants
    i.e. predictions for target atoms i and j are at positions (i, j) or (j, i)
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from misc import AverageKeeper, read_pickle
from features import calculate_coulomb_matrix, calculate_connectivity_matrix, calculate_cep_matrix

MAX_MOL_SIZE = 29


########################################################################################################################
#                                               Feature generation
########################################################################################################################

class MoleculeRaster(object):
    """Calculate each feature matrix for a molecule and stack them as a raster

    Parameters
    ----------
    molecule_map: dict[str, dict]
        The molecule structure map containing the structure and features of the molecule
    transform: callable, optional
        A callable object that can be applied on the stacked raster before returning
    """
    def __init__(self, molecule_map, transform=None):
        self.molecule_map = molecule_map
        self.transform = transform

    def __call__(self, molecule_name):
        """Calculate and stack the feature matrices

        Currently there are four different feature matrices that are calculated:
            - atomic distance matrix
            - coulomb repulsion matrix
            - connectivity matrix weighted by bond order
            - cep matrix

        Parameters
        ----------
        molecule_name: str

        Returns
        -------
        np.ndarray
            A numpy array of shape [4, n_atoms, n_atoms] where n_atoms is the number of atoms in the molecule
            Note: the returned raster may not be a numpy array depending on the 'transform' used.
        """
        molecule = self.molecule_map[molecule_name]

        distance_matrix = molecule['distance_matrix']
        # nonzero_idx = np.nonzero(distance_matrix)
        # distance_matrix[nonzero_idx] = distance_matrix[nonzero_idx] ** -3

        coulomb_matrix = calculate_coulomb_matrix(molecule['rdkit'], distance_matrix)
        connectivity_matrix = calculate_connectivity_matrix(molecule['rdkit'])
        cep_matrix = calculate_cep_matrix(molecule['rdkit'])

        raster = np.stack([distance_matrix, coulomb_matrix, connectivity_matrix, cep_matrix])
        if self.transform is not None:
            raster = self.transform(raster)
        return raster


def create_molecule_rasters(generate_raster, molecule_names):
    """Create the rasters for all molecules

    Note: the purpose of this function is to create the rasters once instead of creating them on the fly

    Parameters
    ----------
    generate_raster: callable
        A callable object that takes a molecule name and returns a multi-channel raster (e.g. MoleculeRaster)
    molecule_names: list[str]
        A list of the molecule names as strings

    Returns
    -------
    dict
        A dictionary mapping each molecule name to its raster
    """
    molecule_rasters = dict()
    for name in tqdm(molecule_names):
        if name != 'scalar_descriptor_keys':
            molecule_rasters[name] = generate_raster(name)
    return molecule_rasters


def create_target_coupling_rasters(data, molecule_map):
    """Create the target coupling matrices for each molecule

    Parameters
    ----------
    data: pd.DataFrame
        The pandas DataFrame where each row defines a pair of atoms and their coupling constant.
        Includes columns for 'molecule_name', 'atom_index_0', 'atom_index_1', and 'scalar_coupling_constant'
    molecule_map: dict[str, dict]
        A dictionary mapping each molecule name to a dictionary containing its structure and features.

    Returns
    -------
    dict
        A dictionary mapping each molecule name to its target coupling matrix
    """
    coupling_targets = dict()
    molecule_groups = data.groupby('molecule_name')
    for name, atom_pairs in tqdm(molecule_groups):
        num_atoms = molecule_map[name]['rdkit'].GetNumAtoms()
        coupling_constants = np.zeros((num_atoms, num_atoms))

        for row_idx, row in atom_pairs.iterrows():
            i, j, coupling = row[['atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]
            coupling_constants[i, j] = coupling
            coupling_constants[j, i] = coupling

        coupling_targets[name] = coupling_constants
    return coupling_targets


########################################################################################################################
#                                                  Data pipeline
########################################################################################################################


def pad_raster(raster, dim):
    """Pad a each channel of the raster to a dim x dim channel and return the inverse index

    Parameters
    ----------
    raster: torch.Tensor
        A pytorch tensor representing the multi-channel raster
    dim: int
        The width/height dimensionality of each channel after padding

    Returns
    -------
    padded_raster: torch.Tensor, inverse_idx: tuple[int, int]
        The padded raster as pytorch tensor with shape [*, dim, dim] and a tuple of integer indices for retrieving the
        feature from the original raster by subsetting each channel.
        e.g. for raster x and inverse index (i, j), the original raster would be x[:, i:j, i:j]
    """
    num_atoms = raster.shape[-1]
    padding = dim - num_atoms

    left_pad = padding // 2
    right_pad = padding - left_pad
    inverse_idx = (left_pad, left_pad + num_atoms)

    padded_channels = []
    for channel in raster:
        channel = F.pad(channel, (left_pad, right_pad, left_pad, right_pad), mode='constant', value=0)
        padded_channels.append(channel)

    padded_raster = torch.stack(padded_channels)
    return padded_raster, inverse_idx


class MoleculeRasterDataset(Dataset):
    """A pytorch Dataset class that returns the molecule rasters and their respective target coupling matrix

    Parameters
    ----------
    molecule_names: list[str]
        The list of molecule names to index
    x_map: dict
        A dictionary mapping a molecule name to its multi-channel raster representation
    y_map: dict
        A dictionary mapping a molecule name to its target coupling matrix
    max_mol_size: int, optional
        The maximum number of atoms across all molecules.
        All channels will be padded to this dimensionality (default=MAX_MOL_SIZE)
    x_transform: callable, optional
        A callable object that operates on the feature rasters
    y_transform: callable, optional
        A callable object that operates on the coupling matrix
    """
    def __init__(self, molecule_names, x_map, y_map, max_mol_size=MAX_MOL_SIZE,
                 x_transform=None, y_transform=None):
        super().__init__()
        self.names = molecule_names
        self.x_map = x_map
        self.y_map = y_map
        self.dim = max_mol_size
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __getitem__(self, idx):
        name = self.names[idx]
        feature_raster = torch.Tensor(self.x_map[name])
        target_raster = torch.Tensor(self.y_map[name]).unsqueeze(0)

        if self.x_transform is not None:
            feature_raster = self.x_transform(feature_raster)
        if self.y_transform is not None:
            target_raster = self.y_transform(target_raster)

        feature_raster, inverse_idx = pad_raster(feature_raster, self.dim)
        target_raster, _ = pad_raster(target_raster, self.dim)

        return feature_raster, target_raster, inverse_idx

    def __len__(self):
        return len(self.names)


########################################################################################################################
#                                                       Model
########################################################################################################################


class MoleculeCNN(nn.Module):
    """The pytorch convolutional model that creates the predicted coupling matrix"""
    def __init__(self, raster_dim=MAX_MOL_SIZE):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.LeakyReLU()
        self.fc = nn.Linear(raster_dim * raster_dim, raster_dim * raster_dim)
        self.raster_dim = raster_dim

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(-1, self.raster_dim * self.raster_dim)
        x = self.fc(x)
        x = x.view(-1, self.raster_dim, self.raster_dim)
        return x


########################################################################################################################
#                                                   Training Pipeline
########################################################################################################################

def get_filled_cells(pred_raster, true_raster):
    """Get the predictions for all cell values in the target matrix that are filled in (i.e. not zero)"""
    zero_mask = (true_raster != 0)
    return pred_raster[zero_mask], true_raster[zero_mask]


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    avg_keeper = AverageKeeper()
    batch_size = dataloader.batch_size
    for batch_x, batch_y, (batch_i, batch_j) in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_i = batch_i.to(device)
        batch_j = batch_j.to(device)

        loss = 0
        optimizer.zero_grad()
        out = model(batch_x)
        for channel_x, channel_y, i, j in zip(out, batch_y.squeeze(), batch_i, batch_j):
            pred, targ = get_filled_cells(channel_x[i:j, i:j], channel_y[i:j, i:j])
            loss += criterion(pred, targ)

        loss /= batch_size
        loss.backward()
        optimizer.step()
        avg_keeper.add(loss)

    avg_loss = avg_keeper.calculate()
    return avg_loss.item()


def val_epoch(model, dataloader, criterion, device):
    model.eval()
    avg_keeper = AverageKeeper()
    batch_size = dataloader.batch_size
    with torch.no_grad():
        for batch_x, batch_y, (batch_i, batch_j) in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            batch_i = batch_i.to(device)
            batch_j = batch_j.to(device)

            loss = 0
            out = model(batch_x)
            for channel_x, channel_y, i, j in zip(out, batch_y.squeeze(), batch_i, batch_j):
                pred, targ = get_filled_cells(channel_x[i:j, i:j], channel_y[i:j, i:j])
                loss += criterion(pred, targ)

            loss /= batch_size
            avg_keeper.add(loss)

        avg_loss = avg_keeper.calculate()
        return avg_loss.item()


########################################################################################################################
#                                                  Run the script
########################################################################################################################


def run(root_dir, coupling, molecule_map):
    data = pd.read_csv(os.path.join(root_dir, f'train/data_{coupling}.csv'))
    data = data[['id', 'molecule_name', 'atom_index_0', 'atom_index_1', 'scalar_coupling_constant']]

    mol_mat = MoleculeRaster(molecule_map)
    molecule_rasters = create_molecule_rasters(mol_mat, list(molecule_map.keys()))
    coupling_targets = create_target_coupling_rasters(data, molecule_map)

    molecule_names = list(coupling_targets.keys())
    names_train, names_val = train_test_split(molecule_names, train_size=0.8, shuffle=True, random_state=0)

    train_dataset = MoleculeRasterDataset(names_train, molecule_rasters, coupling_targets)
    val_dataset = MoleculeRasterDataset(names_val, molecule_rasters, coupling_targets)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=True, num_workers=8)

    device = torch.device('cuda')
    model = MoleculeCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # TODO: refactor this so that it makes and saves the predictions for this coupling type
    num_epochs = 50
    train_losses = []
    val_losses = []
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch}')

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f'Training loss:\t\t{train_loss}')
        train_losses.append(train_loss)

        val_loss = val_epoch(model, val_loader, criterion, device)
        print(f'Validation loss:\t{val_loss}\n')
        val_losses.append(val_loss)


if __name__ == '__main__':
    coupling_type = '1JHN'
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    molecule_structure_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))
    run(ROOT_DIR, coupling_type, molecule_structure_map)
