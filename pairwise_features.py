#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from dtsckit.utils import read_pickle

from chem_math import find_atomic_path, vectorized_dihedral_angle, bond_angle
from features import encode_hybridization


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
        return -1
    g_charge = molecule_map[molecule_name]['g_charges'][atom_idx]
    return g_charge


def get_eem_charge(molecule_map, molecule_name, atom_idx):
    if atom_idx == -1:
        return -1
    eem_charge = molecule_map[molecule_name]['eem_charges'][atom_idx]
    return eem_charge


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
    return encode_hybridization(str(atom.GetHybridization()))


def encode_inner_element(molecule_map, molec_name, atom_idx):
    """Determine and one-hot encode the element of an atom that is not on the peripheries of the molecule"""
    if atom_idx == -1:
        return [-1, -1, -1]

    molecule = molecule_map[molec_name]['rdkit']
    symbol = molecule.GetAtomWithIdx(atom_idx).GetSymbol()
    if symbol == 'C':
        return [1, 0, 0]
    elif symbol == 'N':
        return [0, 1, 0]
    elif symbol == 'O':
        return [0, 0, 1]
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


def calculate_bond_angle(molecule_map, molecule_name, atom_0, atom_1, atom_2):
    if atom_0 == -1 or atom_1 == -1 or atom_2 == -1:
        return None
    coords = molecule_map[molecule_name]['coords']
    return bond_angle(coords[atom_0], coords[atom_1], coords[atom_2])


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
        self.scalar_col_names = molecule_map['scalar_descriptor_keys']

    def get_distances(self, df, atom_i_col, atom_j_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_distance(self.molecule_map, x['molecule_name'], x[atom_i_col], x[atom_j_col]), axis=1
        )
        return df

    def get_gasteiger_charges(self, df, atom_idx_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_gasteiger_charge(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1
        )
        return df

    def get_eem_charges(self, df, atom_idx_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_eem_charge(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1
        )
        return df

    def is_in_ring(self, df, atom_idx_col, col_name):
        df[col_name] = df.apply(
            lambda x: get_ring_membership(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1
        )
        return df

    def get_neighbors(self, df, atom_idx_col, col_names):
        neighbors = df.apply(
            lambda x: calculate_neighborhood(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1
        )
        neighbors = pd.DataFrame(neighbors.values.tolist(), columns=col_names)
        df = pd.concat([df, neighbors], axis=1)
        return df

    def get_hybridizations(self, df, atom_idx_col, col_names):
        hybridizations = df.apply(
            lambda x: find_hybridization(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1
        )
        hybridizations = pd.DataFrame(hybridizations.values.tolist(), columns=col_names)
        df = pd.concat([df, hybridizations], axis=1)
        return df

    def get_element_encodings(self, df, atom_idx_col, col_names):
        elements = df.apply(
            lambda x: encode_inner_element(self.molecule_map, x['molecule_name'], x[atom_idx_col]), axis=1
        )
        elements = pd.DataFrame(elements.values.tolist(), columns=col_names)
        df = pd.concat([df, elements], axis=1)
        return df

    def get_scalar_descriptors(self, df):
        scalar_descriptors = df.apply(lambda x: self.molecule_map[x['molecule_name']]['scalar_descriptors'], axis=1)
        scalar_descriptors = pd.DataFrame(scalar_descriptors.values.tolist(), columns=self.scalar_col_names)
        df = pd.concat([df, scalar_descriptors], axis=1)
        return df

    @staticmethod
    def feature_cols():
        scalar_descriptor_cols = ['asphericity', 'crippen_log', 'crippen_mr',
                                  'weight', 'eccentricity', 'fraction_csp3',
                                  'surface_area', 'npr1', 'npr2', 'hall_kier_alpha',
                                  'num_H', 'num_C', 'num_N', 'num_O', 'num_F',
                                  'num_aliphatic_carbocycles', 'num_aliphatic_heterocycles',
                                  'num_aromatic_carbocycles', 'num_aromatic_heterocycles',
                                  'num_saturated_carbocycles', 'num_saturated_heterocycles',
                                  'num_spiro_atoms', 'num_bridgehead_atoms', 'num_amide_bonds',
                                  'num_H_acceptors', 'num_H_donors']

        return ['distance', 'gcharge_0', 'gcharge_1', 'eem_charge_0', 'eem_charge_1'] + scalar_descriptor_cols

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
        df = self.get_eem_charges(df, 'atom_index_0', 'eem_charge_0')
        df = self.get_eem_charges(df, 'atom_index_1', 'eem_charge_1')
        df = self.get_scalar_descriptors(df)
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
                                                 's', 'sp', 'sp2', 'sp3', 'ring_1']

    def __call__(self, df):
        df = super().__call__(df)

        # get the neighbor distribution
        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'F_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        # get the hybridization
        hybrid_cols = ['s', 'sp', 'sp2', 'sp3']
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
    - eem charges of H0, X, and H1
    - ring membership of atom X
    - bond angle between atom 0, atom X, and atom 1
    """

    @staticmethod
    def feature_cols():
        return FeatureEngineer.feature_cols() + ['distance_hx', 'x_carbon', 'x_nitrogen', 'x_oxygen', 'bond_angle',
                                                 'x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors',
                                                 'x_s', 'x_sp', 'x_sp2', 'x_sp3', 'gcharge_x', 'eem_charge_x', 'ring_x']

    def __call__(self, df):
        df = super().__call__(df)

        # get the index of the atom connecting the two hydrogen atoms
        df['atom_index_x'] = df.apply(
            lambda x: get_middle_atom(self.molecule_map, x['molecule_name'], x['atom_index_0'], x['atom_index_1']),
            axis=1)

        # get the distance between the first hydrogen and the X atom
        df = self.get_distances(df, 'atom_index_0', 'atom_index_x', 'distance_hx')

        # get the one hot encoded element of atom X
        x_element_cols = ['x_carbon', 'x_nitrogen', 'x_oxygen']
        df = self.get_element_encodings(df, 'atom_index_x', x_element_cols)

        # get the neighbor distribution of atom X
        neighbor_cols = ['x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors', 'x_F_neighbors']
        df = self.get_neighbors(df, 'atom_index_x', neighbor_cols)

        # get the hybridization of atom X
        hybrid_cols = ['x_s', 'x_sp', 'x_sp2', 'x_sp3']
        df = self.get_hybridizations(df, 'atom_index_x', hybrid_cols)

        # get the gasteiger charge of middle atom X
        df = self.get_gasteiger_charges(df, 'atom_index_x', 'gcharge_x')

        # get the eem charge of middle atom X
        df = self.get_eem_charges(df, 'atom_index_x', 'eem_charge_x')

        # ring membership of atom X
        df = self.is_in_ring(df, 'atom_index_x', 'ring_x')

        # calculate the bond angle between H-X-H
        df['bond_angle'] = df.apply(lambda x: calculate_bond_angle(self.molecule_map,
                                                                   x['molecule_name'],
                                                                   x['atom_index_0'],
                                                                   x['atom_index_x'],
                                                                   x['atom_index_1']), axis=1)
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
                            's', 'sp', 'sp2', 'sp3', 'ring_1']

    def __call__(self, df):
        df = super().__call__(df)

        df = self.get_distances(df, 'atom_index_x', 'atom_index_1', 'distance_x_')

        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'F_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        hybrid_cols = ['s', 'sp', 'sp2', 'sp3']
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
    - eem charges of H0, X, Y, H1
    - dihedral angle
    - cos(theta)
    - cos(2*theta)
    """

    @staticmethod
    def feature_cols():
        return FeatureEngineer.feature_cols() + ['distance_0x', 'distance_xy',
                                                 'distance_y1', 'distance_0y', 'distance_x1',
                                                 'x_carbon', 'x_nitrogen', 'x_oxygen',
                                                 'y_carbon', 'y_nitrogen', 'y_oxygen',
                                                 'x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors',
                                                 'y_H_neighbors', 'y_C_neighbors', 'y_N_neighbors', 'y_O_neighbors',
                                                 'x_s', 'x_sp', 'x_sp2', 'x_sp3',
                                                 'y_s', 'y_sp', 'y_sp2', 'y_sp3',
                                                 'ring_x', 'ring_y',
                                                 'gcharge_x', 'gcharge_y', 'eem_charge_x', 'eem_charge_y',
                                                 'bond_angle_0xy', 'bond_angle_xy1',
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

        x_element_cols = ['x_carbon', 'x_nitrogen', 'x_oxygen']
        y_element_cols = ['y_carbon', 'y_nitrogen', 'y_oxygen']
        df = self.get_element_encodings(df, 'atom_index_x', x_element_cols)
        df = self.get_element_encodings(df, 'atom_index_y', y_element_cols)

        x_neighbor_cols = ['x_H_neighbors', 'x_C_neighbors', 'x_N_neighbors', 'x_O_neighbors', 'x_F_neighbors']
        y_neighbor_cols = ['y_H_neighbors', 'y_C_neighbors', 'y_N_neighbors', 'y_O_neighbors', 'y_F_neighbors']
        df = self.get_neighbors(df, 'atom_index_x', x_neighbor_cols)
        df = self.get_neighbors(df, 'atom_index_y', y_neighbor_cols)

        x_hybrid_cols = ['x_s', 'x_sp', 'x_sp2', 'x_sp3']
        y_hybrid_cols = ['y_s', 'y_sp', 'y_sp2', 'y_sp3']
        df = self.get_hybridizations(df, 'atom_index_x', x_hybrid_cols)
        df = self.get_hybridizations(df, 'atom_index_y', y_hybrid_cols)

        df = self.is_in_ring(df, 'atom_index_x', 'ring_x')
        df = self.is_in_ring(df, 'atom_index_y', 'ring_y')

        df = self.get_gasteiger_charges(df, 'atom_index_x', 'gcharge_x')
        df = self.get_gasteiger_charges(df, 'atom_index_y', 'gcharge_y')

        df = self.get_eem_charges(df, 'atom_index_x', 'eem_charge_x')
        df = self.get_eem_charges(df, 'atom_index_y', 'eem_charge_y')

        # calculate the bond angle between H-X-Y
        df['bond_angle_0xy'] = df.apply(lambda x: calculate_bond_angle(self.molecule_map,
                                                                       x['molecule_name'],
                                                                       x['atom_index_0'],
                                                                       x['atom_index_x'],
                                                                       x['atom_index_y']), axis=1)
        # calculate the bond angle between X-Y-H
        df['bond_angle_xy1'] = df.apply(lambda x: calculate_bond_angle(self.molecule_map,
                                                                       x['molecule_name'],
                                                                       x['atom_index_x'],
                                                                       x['atom_index_y'],
                                                                       x['atom_index_1']), axis=1)

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
                                             's', 'sp', 'sp2', 'sp3', 'ring_1']

    def __call__(self, df):
        df = super().__call__(df)

        neighbor_cols = ['H_neighbors', 'C_neighbors', 'N_neighbors', 'O_neighbors', 'F_neighbors']
        df = self.get_neighbors(df, 'atom_index_1', neighbor_cols)

        hybrid_cols = ['s', 'sp', 'sp2', 'sp3']
        df = self.get_hybridizations(df, 'atom_index_1', hybrid_cols)

        df = self.is_in_ring(df, 'atom_index_1', 'ring_1')

        return df


########################################################################################################################
#                              Helper functions to reset the current pairwise feature files
########################################################################################################################


def split_by_types(df, target_dir):
    """Split a DataFrame by the 'type' feature and save each group as a separate CSV file

    Parameters
    ----------
    df: pd.DataFrame
        The pandas DataFrame containing the overall data, with a 'type' column to group by
    target_dir: str
        The string directory path where the file will be stored as 'data_{type}.csv'
    """
    coupling_groups = df.groupby('type')
    for coupling_type, data in coupling_groups:
        filepath = os.path.join(target_dir, f'data_{coupling_type}.csv')
        data.to_csv(filepath, index=False)


def reset_files(root_dir):
    """Replace the current pairwise feature files with the base features

    Parameters
    ----------
    root_dir: str
        The string directory path where the training and testing subdirectories can be found
    """
    train_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
    train_dir = os.path.join(root_dir, 'train')
    split_by_types(train_df, train_dir)

    test_df = pd.read_csv(os.path.join(root_dir, 'test.csv'))
    test_dir = os.path.join(root_dir, 'test')
    split_by_types(test_df, test_dir)


########################################################################################################################
#                                                     Main routine
########################################################################################################################


def calculate_pairwise_features(target_dir, molecule_map):
    for filename in os.listdir(target_dir):
        filepath = os.path.join(target_dir, filename)
        data = pd.read_csv(filepath)

        coupling_type, = re.findall(r'data_(.*)\.csv', filename)
        print(f'Coupling: {coupling_type}')

        if coupling_type in ['1JHC', '1JHN']:
            data = Prepare1JH_(molecule_map)(data)
        elif coupling_type == '2JHH':
            data = Prepare2JHH(molecule_map)(data)
        elif coupling_type in ['2JHC', '2JHN']:
            data = Prepare2JH_(molecule_map)(data)
        elif coupling_type == '3JHH':
            data = Prepare3JHH(molecule_map)(data)
        elif coupling_type in ['3JHC', '3JHN']:
            data = Prepare3JH_(molecule_map)(data)
        else:
            raise ValueError(f"Unexpected coupling type: '{coupling_type}'")

        print('Saving the file...\n')
        data.to_csv(filepath, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--reset', action='store_true', help='Resets the current pairwise feature files')
    parser.add_argument('--train', action='store_true', help='Builds the pairwise features for the training files')
    parser.add_argument('--test', action='store_true', help='Build the pairwise features for the testing files')

    args = parser.parse_args()
    reset = args.reset
    train_mode = args.train
    test_mode = args.test

    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    if reset:
        print('Resetting the training and testing pairwise features')
        reset_files(ROOT_DIR)
    if train_mode or test_mode:
        print('Reading the molecular structures...\n')
        molec_struct_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))
        if train_mode:
            print(f'Pairwise feature extraction on training files\n')
            calculate_pairwise_features(os.path.join(ROOT_DIR, 'train'), molec_struct_map)
        if test_mode:
            print(f'Pairwise feature extraction on testing files\n')
            calculate_pairwise_features(os.path.join(ROOT_DIR, 'test'), molec_struct_map)
    print('Done!')
