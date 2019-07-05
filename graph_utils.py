#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def extract_list_from_string(s):
    tokens = s[1:-1].replace(' ', '').split(',')
    return [int(i) for i in tokens]


def create_molecule_structure_map(structures):
    """Create a dictionary mapping the molecule name to each of its atoms' symbols and coordinates

    The key will be the ID of the molecule as a string.
    The value will be a dictionary including:
    'symbols' --> a list of the atomic symbols
    'coords' --> a numpy array of the coordinates, aligned with the symbols with shape (n, 3)
    'bonds' --> a numpy array of the bonds as undirected, graph edges (following the pytorch geometric format)
    """
    structure_map = dict()
    molec_grouping = structures.groupby(['molecule_name'])
    for molecule_id, atoms in molec_grouping:
        symbols = []
        coords = []
        bonds = []
        atoms = atoms.reset_index(drop=True)
        for i, atom in atoms.iterrows():
            symbols.append(atom['atom'])
            coords.append(atom[['x', 'y', 'z']].values.astype(float))
            for j in extract_list_from_string(atom['bonds']):
                bonds.append((i, j))
        coords = np.stack(coords)
        bonds = np.stack(bonds).transpose((1, 0))
        structure_map[molecule_id] = {'symbols': symbols, 'coords': coords, 'bonds': bonds}
    return structure_map
