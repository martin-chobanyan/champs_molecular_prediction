#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from argparse import ArgumentParser
import numpy as np
import pandas as pd
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


def get_middle_atom(molecule_map, molecule_name, start_atom_idx, end_atom_idx):
    try:
        molecule = molecule_map[molecule_name]['rdkit']
        _, middle_atom_idx, _ = find_atomic_path(molecule.GetAtomWithIdx(start_atom_idx),
                                                 molecule.GetAtomWithIdx(end_atom_idx), k=2)
        return middle_atom_idx
    except:
        return -1


def get_middle_atoms(molecule_map, molecule_name, start_atom_idx, end_atom_idx):
    try:
        molecule = molecule_map[molecule_name]['rdkit']
        _, middle_atom_idx_1, middle_atom_idx_2, _ = find_atomic_path(molecule.GetAtomWithIdx(start_atom_idx),
                                                                      molecule.GetAtomWithIdx(end_atom_idx), k=3)
        return [middle_atom_idx_1, middle_atom_idx_2]
    except:
        return [-1, -1]


def calculate_dihedral_angles(df, molecule_map, start_col, x_col, y_col, end_col, step=10000):
    # find the coordinates of each atom in the path
    coords = {start_col: [], x_col: [], y_col: [], end_col: []}
    for i, x in df.iterrows():
        molecule_coords = molecule_map[x['molecule_name']]['coords']
        coords[start_col].append(molecule_coords[x[start_col]])
        coords[x_col].append(molecule_coords[x[x_col]])
        coords[y_col].append(molecule_coords[x[y_col]])
        coords[end_col].append(molecule_coords[x[end_col]])

    # stack the vector positions into 2D arrays
    for position, coordinate in coords.items():
        coords[position] = np.stack(coordinate)

    # calculate the dihedral angle in chunks
    num_rows = df.shape[0]
    end = ((num_rows // step) + 1) * step
    df['dihedral'] = 0.0
    for i in range(0, end, step):
        print(f'Step {int(i / step)} / {int(end / step)}')
        df.iloc[i:i + step - 1, -1] = vectorized_dihedral_angle(coords[start_col][i:i + step],
                                                                coords[x_col][i:i + step],
                                                                coords[y_col][i:i + step],
                                                                coords[end_col][i:i + step])
    return df


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

    def get_distances(self, df, atom_i_col, atom_j_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_distance(self.molecule_map, x['molecule_name'], x[atom_i_col], x[atom_j_col]), axis=1)
        return df

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
        return self.get_distances(df, 'atom_index_0', 'atom_index_1', 'distance')


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
    - distance of H-H
    - distance of H-X
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
        df = self.get_distances(df, 'atom_index_0', 'atom_index_x', 'distance_hx')

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


class Prepare2JH_(Prepare2JHH):
    """Prepares features for 2JHC or 2JHN coupling types

    Features:
    - all features from the 2JHH feature engineer
    - distance from X to C/H
    - neighbors of C/H
    - hybridization of C/H
    """

    @staticmethod
    def feature_cols():
        base_cols = Prepare2JHH.feature_cols()
        return base_cols + ['distance_x_', 'H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors',
                            'sp', 'sp2', 'sp3']

    def __call__(self, df):
        df = super().__call__(df)

        df = self.get_distances(df, 'atom_index_x', 'atom_index_1', 'distance_x_')

        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        hybrid_cols = ['sp', 'sp2', 'sp3']
        df = self.get_hybridizations(df, 'atom_index_1', hybrid_cols)

        return df


class Prepare3JHH(FeatureEngineer):
    """Prepares features for 3JHH coupling, where there is an atomic path H-X-Y-H

    Features:
    - distance of H-H
    - distance of H-X
    - distance of X-Y
    - distance of Y-H
    - distance of H-Y
    - distance of X-H
    - element of X
    - element of Y
    - neighbors of X
    - neighbors of Y
    - hybridization of X
    - hybridization of Y
    - dihedral angle
    - cos(theta)
    - cos(2*theta)
    """

    @staticmethod
    def feature_cols():
        return ['distance', 'distance_0x', 'distance_xy', 'distance_y1', 'distance_0y', 'distance_x1',
                'x_nitrogen', 'x_oxygen', 'y_nitrogen', 'y_oxygen',
                'x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors',
                'y_H_neighbors', 'y_C_neighbors', 'y_N_neighbors', 'y_O_neighbors',
                'x_sp', 'x_sp2', 'x_sp3', 'y_sp', 'y_sp2', 'y_sp3',
                'dihedral', 'cos_theta', 'cos_2theta']

    def __call__(self, df):
        df = super().__call__(df)

        # get the indices of the atoms in between the hydrogen pair
        path_cols = ['atom_index_x', 'atom_index_y']
        middle_atoms = df.apply(
            lambda x: get_middle_atoms(self.molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']),
            axis=1)
        middle_atoms = pd.DataFrame(middle_atoms.values.tolist(), columns=path_cols)
        df = pd.concat([df, middle_atoms], axis=1)

        df = self.get_distances(df, 'atom_index_0', 'atom_index_x', 'distance_0x')
        df = self.get_distances(df, 'atom_index_x', 'atom_index_y', 'distance_xy')
        df = self.get_distances(df, 'atom_index_y', 'atom_index_1', 'distance_y1')
        df = self.get_distances(df, 'atom_index_0', 'atom_index_y', 'distance_0y')
        df = self.get_distances(df, 'atom_index_x', 'atom_index_1', 'distance_x1')

        x_element_cols = ['x_nitrogen', 'x_oxygen']
        y_element_cols = ['y_nitrogen', 'y_oxygen']
        df = self.get_element_encodings(df, 'atom_index_x', x_element_cols)
        df = self.get_element_encodings(df, 'atom_index_y', y_element_cols)

        x_neighbor_cols = ['x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors']
        y_neighbor_cols = ['y_H_neighbors', 'y_C_neighbors', 'y_N_neighbors', 'y_O_neighbors']
        df = self.get_neighbors(df, 'atom_index_x', x_neighbor_cols)
        df = self.get_neighbors(df, 'atom_index_y', y_neighbor_cols)

        x_hybrid_cols = ['x_sp', 'x_sp2', 'x_sp3']
        y_hybrid_cols = ['y_sp', 'y_sp2', 'y_sp3']
        df = self.get_hybridizations(df, 'atom_index_x', x_hybrid_cols)
        df = self.get_hybridizations(df, 'atom_index_y', y_hybrid_cols)

        df = calculate_dihedral_angles(df, self.molecule_map, 'atom_index_0',
                                       'atom_index_x', 'atom_index_y', 'atom_index_1')

        # calculate the karplus features
        dihedral_angles = np.radians(df['dihedral'])
        df['cos_theta'] = np.cos(dihedral_angles)
        df['cos_2theta'] = np.cos(2 * dihedral_angles)

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

        if coupling_type != '3JHH':
            continue
        print(f'Coupling: {coupling_type}')

        data = Prepare3JHH(molec_struct_map)(data)
        for col in data.columns:
            print(f'Column: {col}')
            print(data[col].iloc[:5].values)
            print()
