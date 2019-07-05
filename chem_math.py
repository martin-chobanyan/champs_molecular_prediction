#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def angle_between(p, q):
    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    return np.arccos(np.dot(p, q) / (p_norm * q_norm))


# find a path from an H atom to another atom k hops away
def find_atomic_path(molecule, atom_0, atom_k, k):
    if len(molecule[atom_0]['bonds']) > 1:
        raise ValueError('first atom connected to more than one atom')

    atom_1 = molecule[atom_0]['bonds'][0]

    if k == 1:
        return atom_0, atom_1
    elif k == 2:
        return atom_0, atom_1, atom_k
    elif k == 3:
        atom_1_neighbors = molecule[atom_1]['bonds']
        atom_k_neighbors = molecule[atom_k]['bonds']
        intersecting_atoms = [a for a in atom_1_neighbors if a in atom_k_neighbors]
        atom_2 = intersecting_atoms[0]
        return atom_0, atom_1, atom_2, atom_k
    else:
        raise ValueError(f'Atomic path not supported for k = {k} hops')


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
    """Each parameter is an array of shape (n, 3)"""
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1, axis=1)[:, None]

    v = b0 - np.inner(b0, b1).diagonal()[:, None] * b1
    w = b2 - np.inner(b2, b1).diagonal()[:, None] * b1

    x = np.inner(v, w).diagonal()
    y = np.inner(np.cross(b1, v), w).diagonal()
    return np.degrees(np.arctan2(y, x))



