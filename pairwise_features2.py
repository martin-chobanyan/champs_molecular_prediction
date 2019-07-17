#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from argparse import ArgumentParser
from tqdm import tqdm
import numpy as np
import pandas as pd
from dtsckit.utils import read_pickle
from chem_math import find_atomic_path, vectorized_dihedral_angle


########################################################################################################################
#                               Define the helper functions to get/calculate the features
########################################################################################################################

def get_distance(molecule_map, name, a0, a1):
    distance_matrix = molecule_map[name]['distance_matrix']
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


########################################################################################################################
#                  Define the objects to extract the appropriate features from each coupling type
########################################################################################################################

class FeatureEngineer(object):
    """An base class for the feature engineering objects across coupling types

    Parameters
    ----------
    molecule_map: dict
    """

    def __init__(self, molecule_map):
        self.molecule_map = molecule_map

    def __call__(self, df):
        """Get the distance between the atom pairs

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df['distance'] = df.progress_apply(
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
        c_neighbors = df.progress_apply(
            lambda x: calculate_neighborhood(self.molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
        c_neighbors = pd.DataFrame(c_neighbors.values.tolist(), columns=neighbor_cols)
        df = pd.concat([df, c_neighbors], axis=1)

        # get the hybridization
        hybrid_cols = ['sp', 'sp2', 'sp3']
        hybridizations = df.progress_apply(
            lambda x: find_hybridization(self.molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
        hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=hybrid_cols)
        df = pd.concat([df, hybridizations], axis=1)

        return df


def prepare_2JHH(df_2jhh, molecule_map):
    """
    Features:
    - d of H-H
    - d of H-X
    - element of X
    - neighbors of X
    - hybridization of X

    Parameters
    ----------
    df_2jhh: pd.DataFrame
    molecule_map: dict

    Returns
    -------
    pd.DataFrame
    """
    # df_2jhh['distance'] = df_2jhh.progress_apply(
    #     lambda x: get_distance(molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']), axis=1)
    #
    #
    # # get the distance
    # df_1jhn['distance'] = df_1jhn.progress_apply(
    #     lambda x: get_distance(molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']), axis=1)
    #
    # # get the neighbor distribution of N
    # neighbor_cols = ['H', 'C', 'N', 'O']
    # c_neighbors = df_1jhn.progress_apply(
    #     lambda x: calculate_neighborhood(molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
    # c_neighbors = pd.DataFrame(c_neighbors.values.tolist(), columns=neighbor_cols)
    # df_1jhn = pd.concat([df_1jhn, c_neighbors], axis=1)
    #
    # # get the hybridization of C
    # hybrid_cols = ['N_sp', 'N_sp2', 'N_sp3']
    # hybridizations = df_1jhn.progress_apply(
    #     lambda x: find_hybridization(molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
    # hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=hybrid_cols)
    # df_1jhn = pd.concat([df_1jhn, hybridizations], axis=1)
    #
    # return df_1jhn

    return df_2jhh


if __name__ == '__main__':
    print('Pairwise feature extraction:\n')
    mode = 'train'
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TARGET_DIR = os.path.join(ROOT_DIR, mode)

    print('Reading the molecular structures...\n')
    molec_struct_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))

    tqdm().pandas()
    for filename in os.listdir(TARGET_DIR):
        data = pd.read_csv(os.path.join(TARGET_DIR, filename)).head(100)
        coupling_type, = re.findall(r'data_(.*)\.csv', filename)

        print(f'Coupling: {coupling_type}')
        if coupling_type != '1JHC':
            continue

        data = Prepare1JH_(molec_struct_map)(data)
        for col in data.columns:
            print(f'Column: {col}')
            print(data[col].iloc[:5].values)
            print()
