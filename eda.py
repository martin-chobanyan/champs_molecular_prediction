#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import matplotlib.pyplot as plt
import ase
import ase.visualize
from dtsckit.plot import CustomPlotSize

def find_unique_elements(molec_struct_map):
    """Return a set of all elements used in the molecules"""
    atoms = []
    for molec_name in molec_struct_map.keys():
        atoms += [subst['atom'] for subst in molec_struct_map[molec_name]]
    return set(atoms)


def split_by_types(df, dir):
    coupling_groups = df.groupby('type')
    for coupling_type, data in coupling_groups:
        filepath = os.path.join(dir, f'data_{coupling_type}.csv')
        data.to_csv(filepath, index=False)


def visualize_molecule(molecule):
    symbols = molecule['symbols']
    coords = molecule['coords']
    system = ase.Atoms(positions=coords, symbols=symbols)
    return ase.visualize.view(system, viewer="x3d")


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


if __name__ == '__main__':
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'

    train_df = pd.read_csv(os.path.join(ROOT_DIR, 'train.csv'))
    train_dir = os.path.join(ROOT_DIR, 'train')
    split_by_types(train_df, train_dir)

    test_df = pd.read_csv(os.path.join(ROOT_DIR, 'test.csv'))
    test_dir = os.path.join(ROOT_DIR, 'test')
    split_by_types(test_df, test_dir)
