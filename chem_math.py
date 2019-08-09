#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def angle_between(p, q):
    """Find the angle between two vectors

    Note: this only works for singleton vectors (the function is not vectorized)

    Parameters
    ----------
    p: np.ndarray
        An xyz coordinate
    q: np.ndarray
        An xyz coordinate

    Returns
    -------
    float
        The angle between the two vectors in degrees
    """
    p_norm = np.linalg.norm(p)
    q_norm = np.linalg.norm(q)
    return np.arccos(np.dot(p, q) / (p_norm * q_norm))


def bond_angle(p0, p1, p2):
    """Find the bond angle given three coordinate positions

    Note: this function only works for singleton coordinates (not vectorized)

    Parameters
    ----------
    p0: np.ndarray
        A numpy array of the xyz coordinate of the first atom.
    p1: np.ndarray
        A numpy array of the xyz coordinate of the second/middle atom.
    p2: np.ndarray
        A numpy array of the xyz coordinate of the last atom.

    Returns
    -------
    float
        The bond angle of the three atoms in degrees
    """
    v0 = p0 - p1
    v1 = p2 - p1
    theta = angle_between(v0, v1)
    return theta


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


def vectorized_dihedral_angle(p0, p1, p2, p3):
    """Calculate the dihedral angle given four different coordinates

    Parameters
    ----------
    p0: np.ndarray
    p1: np.ndarray
    p2: np.ndarray
    p3: np.ndarray
    Each parameter is an array of shape (n, 3) where n is the number of different coordinate systems

    Returns
    -------
    np.ndarray
        A numpy array of length n containing the dihedral angles in degrees.
    """
    b0 = -1.0 * (p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1 /= np.linalg.norm(b1, axis=1)[:, None]

    v = b0 - np.inner(b0, b1).diagonal()[:, None] * b1
    w = b2 - np.inner(b2, b1).diagonal()[:, None] * b1

    x = np.inner(v, w).diagonal()
    y = np.inner(np.cross(b1, v), w).diagonal()
    return np.degrees(np.arctan2(y, x))
