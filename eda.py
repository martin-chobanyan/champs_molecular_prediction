#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import ase
import ase.visualize


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


def visualize_molecule(molecule_structures, name):
    molecule = molecule_structures[name]
    symbols = molecule['symbols']
    coords = molecule['coords']
    system = ase.Atoms(positions=coords, symbols=symbols)
    return ase.visualize.view(system, viewer="x3d")

if __name__ == '__main__':
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'

    train_df = pd.read_csv(os.path.join(ROOT_DIR, 'train.csv'))
    train_dir = os.path.join(ROOT_DIR, 'train')
    split_by_types(train_df, train_dir)

    test_df = pd.read_csv(os.path.join(ROOT_DIR, 'test.csv'))
    test_dir = os.path.join(ROOT_DIR, 'test')
    split_by_types(test_df, test_dir)
