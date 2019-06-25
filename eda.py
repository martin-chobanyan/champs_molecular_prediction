#!/usr/bin/env python
# -*- coding: utf-8 -*-


def create_molecule_structure_map(structures):
    """Create a dictionary mapping the molecule name to each of its atoms' symbols and coordinates"""
    struct_map = dict()
    molec_grouping = structures.groupby(['molecule_name'])
    for molec_id, atoms in molec_grouping:
        molec_data = []
        for i, atom in atoms.iterrows():
            symbol = atom['atom']
            coord = atom[['x', 'y', 'z']].values
            bonds = atom['bonds']
            bond_lengths = atom['bond_lengths']
            molec_data.append({'atom': symbol, 'coord': coord, 'bonds': bonds, 'bond_lengths': bond_lengths})
        struct_map[molec_id] = molec_data
    return struct_map


def find_unique_elements(molec_struct_map):
    """Return a set of all elements used in the molecules"""
    atoms = []
    for molec_name in molec_struct_map.keys():
        atoms += [subst['atom'] for subst in molec_struct_map[molec_name]]
    return set(atoms)
