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
import numpy as np
import pandas as pd
from dtsckit.utils import read_pickle
from chem_math import find_atomic_path, vectorized_dihedral_angle


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


def get_molecular_path(molec_struct_map, name, a0, a1, k):
    try:
        molecule = molec_struct_map[name]['rdkit']
        return find_atomic_path(molecule.GetAtomWithIdx(a0), molecule.GetAtomWithIdx(a1), k)
    except:
        if k == 2:
            return [-1, -1, -1]
        elif k == 3:
            return [-1, -1, -1, -1]


def get_atoms_along_path(df, molec_struct_map, num_hops):
    df['path'] = df.progress_apply(lambda row: get_molecular_path(molec_struct_map,
                                                                  row['molecule_name'],
                                                                  row['atom_index_0'],
                                                                  row['atom_index_1'],
                                                                  num_hops), axis=1)

    p_cols = [f'p{i}' for i in range(num_hops + 1)]
    path_df = pd.DataFrame(df['path'].values.tolist(), columns=p_cols)
    df = pd.concat([df, path_df], axis=1)

    df[p_cols] = df[p_cols].astype(int)
    df = df.drop('path', axis=1)

    return df


def idx_to_elements(molec_struct_map, name, intermediate_atoms):
    """Transform a path of intermediate atom indexes to one-hot features for each atom's element

    Note that the path here consists of all atoms in between the first atom and second atom in the pair
    """
    if isinstance(intermediate_atoms, int):
        intermediate_atoms = [intermediate_atoms]

    onehot_features = []
    molecule = molec_struct_map[name]['rdkit']
    for i in intermediate_atoms:
        if i == -1:  # handle the case where the path does not exist
            onehot_features += [-1, -1, -1]
        else:
            symbol = molecule.GetAtomWithIdx(i).GetSymbol()
            if symbol == 'C':
                onehot_features += [1, 0, 0]
            elif symbol == 'N':
                onehot_features += [0, 1, 0]
            elif symbol == 'O':
                onehot_features += [0, 0, 1]
            else:
                raise ValueError(f'Intermediate atoms can only be carbon, nitrogen, or oxygen, not {symbol}.')
    return onehot_features


def transform_atomic_path(df, molec_struct_map, num_hops):
    if num_hops == 2:
        intermediate_cols = ['p1']
    elif num_hops == 3:
        intermediate_cols = ['p1', 'p2']
    else:
        raise ValueError(f"You shouldn't be dealing with atomic paths for num_hops = {num_hops}")

    onehot_features = df.progress_apply(lambda row: idx_to_elements(molec_struct_map,
                                                                    row['molecule_name'],
                                                                    row[intermediate_cols]), axis=1)

    cols = [f'{p}_{e}' for p in intermediate_cols for e in ['C', 'N', 'O']]
    onehot_features = pd.DataFrame(onehot_features.values.tolist(), columns=cols)

    df = pd.concat([df, onehot_features], axis=1)
    return df


def calculate_dihedral_angles(df, molec_struct_map, step):
    # find the coordinates of each atom in the path
    coords = {'p0_coord': [], 'p1_coord': [], 'p2_coord': [], 'p3_coord': []}
    for i, row in df.iterrows():
        molecule_coords = molec_struct_map[row['molecule_name']]['coords']
        coords['p0_coord'].append(molecule_coords[row['p0']])
        coords['p1_coord'].append(molecule_coords[row['p1']])
        coords['p2_coord'].append(molecule_coords[row['p2']])
        coords['p3_coord'].append(molecule_coords[row['p3']])

    # stack the vector positions into 2D arrays
    for position, coordinate in coords.items():
        coords[position] = np.stack(coordinate)

    # calculate the dihedral angle in chunks
    num_rows = df.shape[0]
    end = ((num_rows // step) + 1) * step
    df['dihedral'] = 0.0
    for i in range(0, end, step):
        print(f'Step {int(i / step)} / {int(end / step)}')
        df.loc[i:i + step - 1, 'dihedral'] = vectorized_dihedral_angle(coords['p0_coord'][i:i + step],
                                                                       coords['p1_coord'][i:i + step],
                                                                       coords['p2_coord'][i:i + step],
                                                                       coords['p3_coord'][i:i + step])
    return df


def create_karplus_features(df, molec_struct_map, step=10000):
    df = calculate_dihedral_angles(df, molec_struct_map, step)
    dihedral_angles = np.radians(df['dihedral'])
    df['cos_theta'] = np.cos(dihedral_angles)
    df['cos_2theta'] = np.cos(2 * dihedral_angles)
    return df


if __name__ == '__main__':
    print('Extracting pairwise features:\n')

    mode = 'train'
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TARGET_DIR = os.path.join(ROOT_DIR, mode)

    print('Reading the molecular structures...\n')
    molec_struct_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))

    tqdm().pandas()
    for filename in os.listdir(TARGET_DIR):

        data = pd.read_csv(os.path.join(TARGET_DIR, filename))
        coupling_type, = re.findall(r'data_(.*)\.csv', filename)
        num_hops = int(coupling_type[0])
        h2h = (coupling_type[-2:] == 'HH')

        print(f'Coupling: {coupling_type}')
        ############################################# Distance #########################################################
        print('Extracting the distance between pairs...')
        data['distance'] = data.progress_apply(lambda row: get_distance(molec_struct_map,
                                                                        row['molecule_name'],
                                                                        row['atom_index_0'],
                                                                        row['atom_index_1']), axis=1)
        ############################################ Neighborhood ######################################################
        print('Extracting the local neighborhoods...')
        neighbor_df = data.progress_apply(lambda row: get_pair_neighborhoods(molec_struct_map,
                                                                             row['molecule_name'],
                                                                             row['atom_index_0'],
                                                                             row['atom_index_1']), axis=1)

        if h2h:
            neighborhood_cols = ['a0_C', 'a0_N', 'a0_O', 'a1_C', 'a1_N', 'a1_O']
        else:
            neighborhood_cols = ['a0_C', 'a0_N', 'a0_O', 'a1_H', 'a1_C', 'a1_N', 'a1_O', 'a1_F']

        neighbor_df = pd.DataFrame(neighbor_df.values.tolist(), columns=neighborhood_cols)
        data = pd.concat([data, neighbor_df], axis=1)
        ################################################ Path ##########################################################
        if num_hops > 1:
            print('Finding the path connecting the two atoms...')
            data = get_atoms_along_path(data, molec_struct_map, num_hops)

            print('Encoding the atoms along the path...')
            data = transform_atomic_path(data, molec_struct_map, num_hops)
        ############################################### Karplus ########################################################
        if num_hops == 3:
            print('Calculating the dihedral angle and Karplus features...')
            data = create_karplus_features(data, molec_struct_map)
        ########################################## Clean up and save ###################################################
        if num_hops > 1:
            data = data.drop([f'p{i}' for i in range(num_hops+1)], axis=1)

        print('Saving the file...\n')
        data.to_csv(os.path.join(TARGET_DIR, filename), index=False)
        ################################################################################################################
    print('Done!')