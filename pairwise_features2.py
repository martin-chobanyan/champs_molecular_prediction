#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from argparse import ArgumentParser
import numpy as np
import pandas as pd
from rdkit.Chem.rdchem import Atom
from dtsckit.utils import read_pickle
from chem_math import find_atomic_path, vectorized_dihedral_angle


########################################################################################################################
#                               Define the helper functions to get/calculate the features
########################################################################################################################

def get_distance(molecule_map, molecule_name, a0, a1):
    distance_matrix = molecule_map[molecule_name]['distance_matrix']
    return distance_matrix[a0, a1]


def calculate_neighborhood(molecule_map, molecule_name, atom_idx):
    H, C, N, O = 4 * [0]
    molecule = molecule_map[molecule_name]['rdkit']
    atom = molecule.GetAtomWithIdx(atom_idx)
    hydrogen_root = (atom.GetSymbol() == 'H')

    for neighbor in atom.GetNeighbors():
        symbol = neighbor.GetSymbol()
        if symbol == 'H':
            H += 1
        elif symbol == 'C':
            C += 1
        elif symbol == 'N':
            N += 1
        elif symbol == 'O':
            O += 1

    neighborhood = [C, N, O] if hydrogen_root else [H, C, N, O]
    return neighborhood


def find_hybridization(molecule_map, molec_name, atom_idx):
    molecule = molecule_map[molec_name]['rdkit']
    atom = molecule.GetAtomWithIdx(atom_idx)
    hybridization = str(atom.GetHybridization())
    if hybridization == 'S':
        return [0, 0, 0]
    elif hybridization == 'SP':
        return [1, 0, 0]
    elif hybridization == 'SP2':
        return [0, 1, 0]
    elif hybridization == 'SP3':
        return [0, 0, 1]
    else:
        raise ValueError(f'Encountered a new hybridization: {hybridization}')


def encode_inner_element(molecule_map, molec_name, atom_idx):
    """Determine and one-hot encode the element of an atom that is not on the peripheries of the molecule"""
    molecule = molecule_map[molec_name]['rdkit']
    symbol = molecule.GetAtomWithIdx(atom_idx).GetSymbol()
    if symbol == 'C':
        return [0, 0]
    elif symbol == 'N':
        return [1, 0]
    elif symbol == 'O':
        return [0, 1]
    else:
        raise ValueError("'Inner atoms' must be carbons, nitrogens, or oxygens")


def get_middle_atom(molecule_map, molecule_name, atom_0_idx, atom_1_idx):
    try:
        molecule = molecule_map[molecule_name]['rdkit']
        _, middle_atom_idx, _ = find_atomic_path(molecule.GetAtomWithIdx(atom_0_idx),
                                                 molecule.GetAtomWithIdx(atom_1_idx), k=2)
        return middle_atom_idx
    except:
        return -1


########################################################################################################################
#                  Define the objects to extract the appropriate features from each coupling type
########################################################################################################################

class FeatureEngineer(object):
    """A base class for the feature engineering objects across coupling types

    Parameters
    ----------
    molecule_map: dict
    """

    def __init__(self, molecule_map):
        self.molecule_map = molecule_map

    def get_neighbors(self, df, atom_idx_col, col_names):
        neighbors = df.apply(lambda x: calculate_neighborhood(self.molecule_map, x['molecule_name'], x[atom_idx_col]),
                             axis=1)
        neighbors = pd.DataFrame(neighbors.values.tolist(), columns=col_names)
        df = pd.concat([df, neighbors], axis=1)
        return df

    def get_hybridizations(self, df, atom_idx_col, col_names):
        hybridizations = df.apply(lambda x: find_hybridization(self.molecule_map, x['molecule_name'], x[atom_idx_col]),
                                  axis=1)
        hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=col_names)
        df = pd.concat([df, hybridizations], axis=1)
        return df

    def get_element_encodings(self, df, atom_idx_col, col_names):
        elements = df.apply(lambda x: encode_inner_element(self.molecule_map, x['molecule_name'], x[atom_idx_col]),
                            axis=1)
        elements = pd.DataFrame(elements.values.tolist(), columns=col_names)
        df = pd.concat([df, elements], axis=1)
        return df

    def __call__(self, df):
        """Get the distance between the atom pairs

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df['distance'] = df.apply(
            lambda x: get_distance(self.molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']), axis=1)
        return df


class Prepare1JH_(FeatureEngineer):
    """Prepare features for 1JHC and 1JHN

    Features:
    - distance of H-C (or H-N)
    - C neighbors (or N neighbors)
    - C hybridization (or N hybridization)
    """

    @staticmethod
    def feature_cols():
        return ['distance', 'H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'sp', 'sp2', 'sp3']

    def __call__(self, df):
        df = super().__call__(df)

        # get the neighbor distribution
        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        # get the hybridization
        hybrid_cols = ['sp', 'sp2', 'sp3']
        df = self.get_hybridizations(df, 'atom_index_1', hybrid_cols)

        return df


class Prepare2JHH(FeatureEngineer):
    """Prepares features for 2JHH coupling, where there is an atomic path H-X-H

    Features:
    - d of H-H
    - d of H-X
    - element of X
    - neighbors of X
    - hybridization of X
    """

    @staticmethod
    def feature_cols():
        return ['distance', 'distance_hx', 'x_nitrogen', 'x_oxygen',
                'x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors',
                'x_sp', 'x_sp2', 'x_sp3']

    def __call__(self, df):
        df = super().__call__(df)

        # get the index of the atom connecting the two hydrogen atoms
        df['atom_index_x'] = df.apply(
            lambda x: get_middle_atom(self.molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']),
            axis=1)

        # get the distance between the first hydrogen and the X atom
        df['distance_hx'] = df.apply(
            lambda x: get_distance(self.molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_x']), axis=1)

        # get the one hot encoded element of atom X
        x_element_cols = ['x_nitrogen', 'x_oxygen']
        df = self.get_element_encodings(df, 'atom_index_x', x_element_cols)

        # get the neighbor distribution of atom X
        neighbor_cols = ['x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors']
        df = self.get_neighbors(df, 'atom_index_x', neighbor_cols)

        # get the hybridization of atom X
        hybrid_cols = ['x_sp', 'x_sp2', 'x_sp3']
        df = self.get_hybridizations(df, 'atom_index_x', hybrid_cols)

        return df


if __name__ == '__main__':
    print('Pairwise feature extraction:\n')
    mode = 'train'
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TARGET_DIR = os.path.join(ROOT_DIR, mode)

    print('Reading the molecular structures...\n')
    molec_struct_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))

    for filename in os.listdir(TARGET_DIR):
        data = pd.read_csv(os.path.join(TARGET_DIR, filename)).head(100)
        coupling_type, = re.findall(r'data_(.*)\.csv', filename)

        if coupling_type != '2JHH':
            continue
        print(f'Coupling: {coupling_type}')

        data = Prepare1JH_(molec_struct_map)(data)
        for col in data.columns:
            print(f'Column: {col}')
            print(data[col].iloc[:5].values)
            print()
