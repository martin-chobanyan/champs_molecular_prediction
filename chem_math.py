#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# find a path from an H atom to another atom three hops away
def find_fourway_atomic_path(molecule, start_atom, end_atom):
    if len(molecule[start_atom]['bonds']) > 1:
        raise ValueError('first atom connected to more than one atom')

    middle_atom_1 = molecule[start_atom]['bonds'][0]
    middle_atom_1_neighbors = molecule[middle_atom_1]['bonds']

    end_atom_neighbors = molecule[end_atom]['bonds']
    intersecting_atoms = [a for a in middle_atom_1_neighbors if a in end_atom_neighbors]
    middle_atom_2 = intersecting_atoms[0]

    return start_atom, middle_atom_1, middle_atom_2, end_atom


# Source: stackoverflow (Dihedral/Torsion Angle From Four Points in Cartesian Coordinates in Python)
def dihedral_angle(p0, p1, p2, p3):
    """Praxeolitic formula, 1 sqrt, 1 cross product"""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1)

    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1

    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return np.degrees(np.arctan2(y, x))


def vectorized_dihedral_angle(p0, p1, p2, p3):
    """Praxeolitic formula, 1 sqrt, 1 cross product"""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1, axis=1)[:, None]

    v = b0 - np.inner(b0, b1).diagonal()[:, None] * b1
    w = b2 - np.inner(b2, b1).diagonal()[:, None] * b1

    x = np.inner(v, w).diagonal()
    y = np.inner(np.cross(b1, v), w).diagonal()
    return np.degrees(np.arctan2(y, x))
