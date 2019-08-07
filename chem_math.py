#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import rdkit
from rdkit.Chem.rdchem import Atom


def angle_between(p, q):
    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    return np.arccos(np.dot(p, q) / (p_norm * q_norm))


def find_atomic_path(atom_0, atom_k, k=3, return_indices=True):
    """Find a path from an H atom to another atom k hops away

    Parameters
    ----------
    atom_0: Atom
        The rdkit atom for the starting hydrogen in the molecule
    atom_k: Atom
        The rdkit atom for the ending atom in the molecule
    k: int
        The number of hops/edges along the path (default=3)
    return_indices: bool
        If True, then only returns the indices of the atoms along the path (default=True)

    Returns
    -------
    list[int] or list[Atom]
        A list of integer ids for each atom along the path.
        If return_indices is False, then a list of rdkit Atom objects will be returned.  The list has length k+1.
    """
    atom_0_neighbors = atom_0.GetNeighbors()
    if len(atom_0_neighbors) != 1:
        raise ValueError('First atom must be a hydrogen i.e. only have one bond.')

    # get the only neighbor of the starting hydrogen atom
    atom_1 = atom_0_neighbors[0]

    if k == 1:
        path = atom_0, atom_k
    elif k == 2:
        path = atom_0, atom_1, atom_k
    elif k == 3:
        atom_1_neighbors = atom_1.GetNeighbors()
        atom_k_neighbors = atom_k.GetNeighbors()

        atom_1_neighbors_idx = [a.GetIdx() for a in atom_1_neighbors]
        atom_k_neighbors_idx = [a.GetIdx() for a in atom_k_neighbors]

        intersecting_atoms = [a for (a, i) in zip(atom_1_neighbors, atom_1_neighbors_idx)
                              if i in atom_k_neighbors_idx]

        # grab the first atom in the neighborhood intersection
        # this means the path is arbitrary if k = 3
        atom_2 = intersecting_atoms[0]

        path = atom_0, atom_1, atom_2, atom_k
    else:
        raise ValueError(f'Atomic path not supported for k = {k} hops')
    return [atom.GetIdx() for atom in path] if return_indices else path


def bond_angle(p0, p1, p2):
    v0 = p0 - p1
    v1 = p2 - p1
    v1_norm = np.linalg.norm(v0)
    v2_norm = np.linalg.norm(v1)
    theta = np.arccos(np.dot(v0, v1) / (v1_norm * v2_norm))
    return theta


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



