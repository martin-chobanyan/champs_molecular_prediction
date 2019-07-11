#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This file will read in each coupling DataFrame and extract features at the pairwise level.
These features include:
- distance between the pair
- elemental neighborhood distribution around both atoms
- elements along the path between the two atoms (2 for 3J__, 1 for 2J__, none for 1J__)
- dihedral angle theta, cos(theta), cos(2*theta) (only for 3J__ coupling)
"""


import os
import re
from tqdm import tqdm
import pandas as pd
from dtsckit.utils import read_pickle


def get_distance(molec_struct_map, name, a0, a1):
    distance_matrix = molec_struct_map[name]['distance_matrix']
    return distance_matrix[a0, a1]


def neighborhood_distribution(neighbors, hydrogen_root):
    """Return a neighborhood distribution given the surrounding elements"""
    H, C, N, O, F = 5 * [0]
    for atom in neighbors:
        if atom == 'H':
            H += 1
        elif atom == 'C':
            C += 1
        elif atom == 'N':
            N += 1
        elif atom == 'O':
            O += 1
        elif atom == 'F':
            F += 1

    # Nones are placed so that the features can be stacked in a DataFrame
    neighborhood = [C, N, O] if hydrogen_root else [H, C, N, O, F]
    return neighborhood


def get_pair_neighborhoods(molec_struct_map, name, a0, a1):
    molecule = molec_struct_map[name]['rdkit']

    atom_0 = molecule.GetAtomWithIdx(a0)
    atom_1 = molecule.GetAtomWithIdx(a1)

    atom_0_neighbors = [neighbor.GetSymbol() for neighbor in atom_0.GetNeighbors()]
    atom_1_neighbors = [neighbor.GetSymbol() for neighbor in atom_1.GetNeighbors()]

    atom_0_distribution = neighborhood_distribution(atom_0_neighbors, atom_0.GetSymbol() == 'H')
    atom_1_distribution = neighborhood_distribution(atom_1_neighbors, atom_1.GetSymbol() == 'H')

    return atom_0_distribution + atom_1_distribution


if __name__ == '__main__':
    print('Extracting pairwise features...\n')

    mode = 'train'
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TARGET_DIR = os.path.join(ROOT_DIR, mode)

    print('Reading the molecular structures...\n')
    molec_struct_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))

    tqdm().pandas()
    for filename in os.listdir(TARGET_DIR):

        df = pd.read_csv(os.path.join(TARGET_DIR, filename))
        coupling_type, = re.findall(r'data_(.*)\.csv', filename)
        h2h = (coupling_type[-2:] == 'HH')

        print(f'Coupling: {coupling_type}')

        ############################################# Distance #########################################################
        print('Extracting the distance between pairs...')
        df['distance'] = df.progress_apply(lambda row: get_distance(molec_struct_map,
                                                                    row['molecule_name'],
                                                                    row['atom_index_0'],
                                                                    row['atom_index_1']), axis=1)
        ############################################ Neighborhood ######################################################
        print('Extracting the local neighborhoods...')
        neighbor_df = df.progress_apply(lambda row: get_pair_neighborhoods(molec_struct_map,
                                                                           row['molecule_name'],
                                                                           row['atom_index_0'],
                                                                           row['atom_index_1']), axis=1)

        if h2h:
            neighborhood_cols = ['a0_C', 'a0_N', 'a0_O', 'a1_C', 'a1_N', 'a1_O']
        else:
            neighborhood_cols = ['a0_C', 'a0_N', 'a0_O', 'a1_H', 'a1_C', 'a1_N', 'a1_O', 'a1_F']

        neighbor_df = pd.DataFrame(neighbor_df.values.tolist(), columns=neighborhood_cols)
        df = pd.concat([df, neighbor_df], axis=1)
        ################################################ Path ##########################################################
        num_hops = int(coupling_type[0])

        ############################################### Karplus ########################################################

        ################################################################################################################



        print(df.head())
        break