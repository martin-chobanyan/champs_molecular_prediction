#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from chem_math import angle_between, find_atomic_path


def extract_features(name, a0, a1, molec_struct_map, bond_separation=1):
    # H, C, N, O, F, distance
    num_features = 6 + (bond_separation - 1)
    features = np.zeros(num_features)

    molecule_struct = molec_struct_map[name]
    neighbors = set(molecule_struct[a0]['bonds'] + molecule_struct[a1]['bonds'])
    for neighbor_atom in neighbors:
        symbol = molecule_struct[neighbor_atom]['atom']
        if symbol == 'H':
            i = 0
        elif symbol == 'C':
            i = 1
        elif symbol == 'N':
            i = 2
        elif symbol == 'O':
            i = 3
        elif symbol == 'F':
            i = 4
        else:
            raise ValueError(f'New element encountered: {symbol}')
        features[i] += 1

    try:
        d = 0
        atomic_path = find_atomic_path(molecule_struct, a0, a1, k=bond_separation)
        for start, end in zip(atomic_path[:-1], atomic_path[1:]):
            d += np.linalg.norm(molecule_struct[start]['coord'] - molecule_struct[end]['coord'])
        features[5] = d
    except IndexError:
        atomic_path = None

    if atomic_path is not None:
        if bond_separation == 2:
            a1, a2, a3 = atomic_path
            p1 = molecule_struct[a1]['coord']
            p2 = molecule_struct[a2]['coord']
            p3 = molecule_struct[a3]['coord']
            features[6] = angle_between(p1 - p2, p3 - p2)

        elif bond_separation == 3:
            a1, a2, a3, a4 = atomic_path
            p1 = molecule_struct[a1]['coord']
            p2 = molecule_struct[a2]['coord']
            p3 = molecule_struct[a3]['coord']
            p4 = molecule_struct[a4]['coord']

            features[6] = angle_between(p1 - p2, p3 - p2)
            features[7] = angle_between(p2 - p3, p4 - p3)

    return features


def calculate_more_features(coupling_data, coupling_type, molec_struct_map):
    features = []
    bond_separation = int(coupling_type[0])
    for i, (name, a0, a1) in enumerate(zip(coupling_data['molecule_name'],
                                           coupling_data['atom_index_0'],
                                           coupling_data['atom_index_1'])):
        f = extract_features(name, a0, a1, molec_struct_map, bond_separation)
        features.append(f)

    features = np.stack(features)
    return features
