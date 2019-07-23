#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dtsckit.plot import CustomPlotSize
from dtsckit.utils import read_pickle
from chem_math import find_atomic_path, vectorized_dihedral_angle


########################################################################################################################
#                                             Feature Visualization
########################################################################################################################

def plot_karplus(df, coupling, plot_size=(10, 8), num_points=100000):
    if len(df) > num_points:
        df = df[:num_points]
    with CustomPlotSize(*plot_size):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(df['cos_theta'][:100000],
                   df['cos_2theta'][:100000],
                   df['scalar_coupling_constant'][:100000], alpha=0.8)
        ax.set_title(coupling)
        ax.set_xlabel('Cosine(dihedral)')
        ax.set_ylabel('Cosine(2*dihedral)')
        ax.set_zlabel('Scalar Coupling Constant')
        plt.show()


########################################################################################################################
#                               Define the helper functions to get/calculate the features
########################################################################################################################


def get_distance(molecule_map, molecule_name, a0, a1):
    if a0 == -1 or a1 == -1:
        return -1
    distance_matrix = molecule_map[molecule_name]['distance_matrix']
    return distance_matrix[a0, a1]


def get_gasteiger_charge(molecule_map, molecule_name, atom_idx):
    if atom_idx == -1:
        return -1000
    g_charge = molecule_map[molecule_name]['g_charges'][atom_idx]
    return g_charge


def get_ring_membership(molecule_map, molecule_name, atom_idx):
    if atom_idx == -1:
        return -1
    molecule = molecule_map[molecule_name]['rdkit']
    atom = molecule.GetAtomWithIdx(atom_idx)
    in_ring = int(atom.IsInRing())
    return in_ring


def calculate_neighborhood(molecule_map, molecule_name, atom_idx):
    if atom_idx == -1:
        return [-1, -1, -1, -1]

    neighbor_count = {'H': 0, 'C': 0, 'N': 0, 'O': 0, 'F': 0}
    molecule = molecule_map[molecule_name]['rdkit']
    atom = molecule.GetAtomWithIdx(atom_idx)

    for neighbor in atom.GetNeighbors():
        symbol = neighbor.GetSymbol()
        neighbor_count[symbol] += 1

    return [neighbor_count['H'], neighbor_count['C'], neighbor_count['N'], neighbor_count['O'], neighbor_count['F']]


def find_hybridization(molecule_map, molec_name, atom_idx):
    if atom_idx == -1:
        return [-1, -1, -1]

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
    if atom_idx == -1:
        return [-1, -1]

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
        df.loc[i:i + step - 1, 'dihedral'] = vectorized_dihedral_angle(coords[start_col][i:i + step],
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

    def get_gasteiger_charges(self, df, atom_idx_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_gasteiger_charge(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1)
        return df

    def is_in_ring(self, df, atom_idx_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_ring_membership(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1)
        return df

    def get_neighbors(self, df, atom_idx_col, col_names):
        neighbors = df.apply(
            lambda x: calculate_neighborhood(self.molecule_map, x['molecule_name'], x[atom_idx_col]),
            axis=1)
        neighbors = pd.DataFrame(neighbors.values.tolist(), columns=col_names)
        df = pd.concat([df, neighbors], axis=1)
        return df

    def get_hybridizations(self, df, atom_idx_col, col_names):
        hybridizations = df.apply(
            lambda x: find_hybridization(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1)
        hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=col_names)
        df = pd.concat([df, hybridizations], axis=1)
        return df

    def get_element_encodings(self, df, atom_idx_col, col_names):
        elements = df.apply(lambda x: encode_inner_element(self.molecule_map, x['molecule_name'], x[atom_idx_col]),
                            axis=1)
        elements = pd.DataFrame(elements.values.tolist(), columns=col_names)
        df = pd.concat([df, elements], axis=1)
        return df

    @staticmethod
    def feature_cols():
        return ['distance', 'gcharge_0', 'gcharge_1']

    def __call__(self, df):
        """Get the distance between the atom pairs

        Parameters
        ----------
        df: pd.DataFrame

        Returns
        -------
        pd.DataFrame
        """
        df = self.get_distances(df, 'atom_index_0', 'atom_index_1', 'distance')
        df = self.get_gasteiger_charges(df, 'atom_index_0', 'gcharge_0')
        df = self.get_gasteiger_charges(df, 'atom_index_1', 'gcharge_1')
        return df


class Prepare1JH_(FeatureEngineer):
    """Prepare features for 1JHC and 1JHN

    Features:
    - distance of H-C (or H-N)
    - C neighbors (or N neighbors)
    - C hybridization (or N hybridization)
    - ring membership of the C/N atom
    """

    @staticmethod
    def feature_cols():
        return FeatureEngineer.feature_cols() + ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors',
                                                 'sp', 'sp2', 'sp3', 'ring_1']

    def __call__(self, df):
        df = super().__call__(df)

        # get the neighbor distribution
        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'F_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        # get the hybridization
        hybrid_cols = ['sp', 'sp2', 'sp3']
        df = self.get_hybridizations(df, 'atom_index_1', hybrid_cols)

        # is the C/N atom in a ring?
        df = self.is_in_ring(df, 'atom_index_1', 'ring_1')

        return df


class Prepare2JHH(FeatureEngineer):
    """Prepares features for 2JHH coupling, where there is an atomic path H-X-H

    Features:
    - distance of H-H
    - distance of H-X
    - element of X
    - neighbors of X
    - hybridization of X
    - gasteiger charges of H0, X, and H1
    - ring membership of atom X
    """

    @staticmethod
    def feature_cols():
        return FeatureEngineer.feature_cols() + ['distance_hx', 'x_nitrogen', 'x_oxygen', 'x_H_neighbors',
                                                 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors',
                                                 'x_sp', 'x_sp2', 'x_sp3', 'gcharge_x', 'ring_x']

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
        neighbor_cols = ['x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors', 'x_F_neighbors']
        df = self.get_neighbors(df, 'atom_index_x', neighbor_cols)

        # get the hybridization of atom X
        hybrid_cols = ['x_sp', 'x_sp2', 'x_sp3']
        df = self.get_hybridizations(df, 'atom_index_x', hybrid_cols)

        # get the gasteiger charge of middle atom X
        df = self.get_gasteiger_charges(df, 'atom_index_x', 'gcharge_x')

        # ring membership of atom X
        df = self.is_in_ring(df, 'atom_index_x', 'ring_x')

        return df


class Prepare2JH_(Prepare2JHH):
    """Prepares features for 2JHC or 2JHN coupling types

    Features:
    - all features from the 2JHH feature engineer
    - distance from X to C/N
    - neighbors of C/N
    - hybridization of C/N
    - gasteiger charges of H, X, and C/N
    - ring membership of C/N
    """

    @staticmethod
    def feature_cols():
        base_cols = Prepare2JHH.feature_cols()
        return base_cols + ['distance_x_', 'H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors',
                            'sp', 'sp2', 'sp3', 'ring_1']

    def __call__(self, df):
        df = super().__call__(df)

        df = self.get_distances(df, 'atom_index_x', 'atom_index_1', 'distance_x_')

        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'F_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        hybrid_cols = ['sp', 'sp2', 'sp3']
        df = self.get_hybridizations(df, 'atom_index_1', hybrid_cols)

        df = self.is_in_ring(df, 'atom_index_1', 'ring_1')

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
    - ring membership of X, Y
    - gasteiger charges of H0, X, Y, H1
    - dihedral angle
    - cos(theta)
    - cos(2*theta)
    """

    @staticmethod
    def feature_cols():
        return FeatureEngineer.feature_cols() + ['distance_0x', 'distance_xy',
                                                 'distance_y1', 'distance_0y', 'distance_x1',
                                                 'x_nitrogen', 'x_oxygen', 'y_nitrogen', 'y_oxygen',
                                                 'x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors',
                                                 'y_H_neighbors', 'y_C_neighbors', 'y_N_neighbors', 'y_O_neighbors',
                                                 'x_sp', 'x_sp2', 'x_sp3', 'y_sp', 'y_sp2', 'y_sp3', 'ring_x', 'ring_y',
                                                 'gcharge_x', 'gcharge_y', 'dihedral', 'cos_theta', 'cos_2theta']

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

        x_neighbor_cols = ['x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors', 'x_F_neighbors']
        y_neighbor_cols = ['y_H_neighbors', 'y_C_neighbors', 'y_N_neighbors', 'y_O_neighbors', 'y_F_neighbors']
        df = self.get_neighbors(df, 'atom_index_x', x_neighbor_cols)
        df = self.get_neighbors(df, 'atom_index_y', y_neighbor_cols)

        x_hybrid_cols = ['x_sp', 'x_sp2', 'x_sp3']
        y_hybrid_cols = ['y_sp', 'y_sp2', 'y_sp3']
        df = self.get_hybridizations(df, 'atom_index_x', x_hybrid_cols)
        df = self.get_hybridizations(df, 'atom_index_y', y_hybrid_cols)

        df = self.is_in_ring(df, 'atom_index_x', 'ring_x')
        df = self.is_in_ring(df, 'atom_index_y', 'ring_y')

        df = self.get_gasteiger_charges(df, 'atom_index_x', 'gcharge_x')
        df = self.get_gasteiger_charges(df, 'atom_index_y', 'gcharge_y')

        df = calculate_dihedral_angles(df, self.molecule_map, 'atom_index_0',
                                       'atom_index_x', 'atom_index_y', 'atom_index_1')

        # calculate the karplus features
        dihedral_angles = np.radians(df['dihedral'])
        df['cos_theta'] = np.cos(dihedral_angles)
        df['cos_2theta'] = np.cos(2 * dihedral_angles)

        return df


class Prepare3JH_(Prepare3JHH):
    """Prepares features for 3JHC or 3JHN

    Most of the features are the same as for 3JHH. Additionally:
    - neighbors of C/N
    - hybridization of C/N
    - ring membership of C/N
    """

    @staticmethod
    def feature_cols():
        return Prepare3JHH.feature_cols() + ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors',
                                             'sp', 'sp2', 'sp3', 'ring_1']

    def __call__(self, df):
        df = super().__call__(df)

        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'F_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        hybrid_cols = ['sp', 'sp2', 'sp3']
        df = self.get_hybridizations(df, 'atom_index_1', hybrid_cols)

        df = self.is_in_ring(df, 'atom_index_1', 'ring_1')

        return df


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', required=True, type=str, help='Specify training or testing mode')
    args = parser.parse_args()

    mode = args.mode
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TARGET_DIR = os.path.join(ROOT_DIR, mode)
    print(f'Pairwise feature extraction on {mode}ing files\n')

    print('Reading the molecular structures...\n')
    molec_struct_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))

    for filename in os.listdir(TARGET_DIR):
        data = pd.read_csv(os.path.join(TARGET_DIR, filename))
        coupling_type, = re.findall(r'data_(.*)\.csv', filename)
        print(f'Coupling: {coupling_type}')

        if coupling_type == '1JHC' or coupling_type == '1JHN':
            data = Prepare1JH_(molec_struct_map)(data)
        elif coupling_type == '2JHH':
            data = Prepare2JHH(molec_struct_map)(data)
        elif coupling_type == '2JHC' or coupling_type == '2JHN':
            data = Prepare2JH_(molec_struct_map)(data)
        elif coupling_type == '3JHH':
            data = Prepare3JHH(molec_struct_map)(data)
        elif coupling_type == '3JHC' or coupling_type == '3JHN':
            data = Prepare3JH_(molec_struct_map)(data)
        else:
            raise ValueError(f"Unexpected coupling type: '{coupling_type}'")

        print('Saving the file...\n')
        data.to_csv(os.path.join(TARGET_DIR, filename), index=False)
    print('Done!')
