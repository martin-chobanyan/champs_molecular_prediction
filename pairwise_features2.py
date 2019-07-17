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
#                 Define the functions to extract the appropriate features from each coupling type
########################################################################################################################

def prepare_1JHC(df_1jhc, molecule_map):
    """
    Features:
    - d of H-C
    - C neighbors
    - C hybridization

    Parameters
    ----------
    df_1jhc: pd.DataFrame
    molecule_map: dict

    Returns
    -------
    pd.DataFrame
    """
    # get the distance
    df_1jhc['distance'] = df_1jhc.progress_apply(
        lambda x: get_distance(molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']), axis=1)

    # get the neighbor distribution of C
    neighbor_cols = ['H', 'C', 'N', 'O']
    c_neighbors = df_1jhc.progress_apply(
        lambda x: calculate_neighborhood(molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
    c_neighbors = pd.DataFrame(c_neighbors.values.tolist(), columns=neighbor_cols)
    df_1jhc = pd.concat([df_1jhc, c_neighbors], axis=1)

    # get the hybridization of C
    hybrid_cols = ['C_sp', 'C_sp2', 'C_sp3']
    hybridizations = df_1jhc.progress_apply(
        lambda x: find_hybridization(molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
    hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=hybrid_cols)
    df_1jhc = pd.concat([df_1jhc, hybridizations], axis=1)

    return df_1jhc


def prepare_1JHN(df_1jhn, molecule_map):
    """
    Features:
    - d of H-N
    - N neighbors
    - N hybridization

    Parameters
    ----------
    df_1jhn: pd.DataFrame
    molecule_map: dict

    Returns
    -------
    pd.DataFrame
    """
    # get the distance
    df_1jhn['distance'] = df_1jhn.progress_apply(
        lambda x: get_distance(molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']), axis=1)

    # get the neighbor distribution of N
    neighbor_cols = ['H', 'C', 'N', 'O']
    c_neighbors = df_1jhn.progress_apply(
        lambda x: calculate_neighborhood(molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
    c_neighbors = pd.DataFrame(c_neighbors.values.tolist(), columns=neighbor_cols)
    df_1jhn = pd.concat([df_1jhn, c_neighbors], axis=1)

    # get the hybridization of C
    hybrid_cols = ['N_sp', 'N_sp2', 'N_sp3']
    hybridizations = df_1jhn.progress_apply(
        lambda x: find_hybridization(molecule_map, x['molecule_name'], x['atom_index_1']), axis=1)
    hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=hybrid_cols)
    df_1jhn = pd.concat([df_1jhn, hybridizations], axis=1)

    return df_1jhn


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
        if coupling_type != '1JHN':
            continue

        data = prepare_1JHC(data, molec_struct_map)
        for col in data.columns:
            print(f'Column: {col}')
            print(data[col].iloc[:5].values)
            print()
