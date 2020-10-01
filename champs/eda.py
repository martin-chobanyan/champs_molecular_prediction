#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This file defines tools to explore and visualize the data."""

import matplotlib.pyplot as plt
import ase
import ase.visualize


def find_unique_elements(molecule_map):
    """Return a set of all elements used in the molecules"""
    atoms = []
    for molec_name in molecule_map.keys():
        atoms += [subst['atom'] for subst in molecule_map[molec_name]]
    return set(atoms)


def plot_karplus(df, coupling, plot_size=(10, 8), num_points=100000):
    """Plot the Karplus features vs the scalar coupling constants"""
    df_sample = df[:num_points] if len(df) > num_points else df
    fig = plt.figure(figsize=plot_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df_sample['cos_theta'],
               df_sample['cos_2theta'],
               df_sample['scalar_coupling_constant'], alpha=0.8)
    ax.set_title(coupling)
    ax.set_xlabel('Cosine(dihedral)')
    ax.set_ylabel('Cosine(2*dihedral)')
    ax.set_zlabel('Scalar Coupling Constant')
    plt.show()


class MoleculeVisualizer(object):
    """Use the 'ase' library to visualize different molecules

    Parameters
    ----------
    molecule_map: dict[str, dict]
        A dictionary mapping each molecule name to another dictionary defining its structure. The secondary dictionary
        must have the following mappings:
        'symbols' -> list of string elemental symbols for each atom in the molecule
        'coords' -> ndarray of shape [n_atoms, 3] containing the xyz coordinates of each atom, aligned with the symbols
    """
    def __init__(self, molecule_map):
        self.molecule_map = molecule_map

    def __call__(self, molecule_name):
        molecule = self.molecule_map[molecule_name]
        symbols = molecule['symbols']
        coords = molecule['coords']
        system = ase.Atoms(positions=coords, symbols=symbols)
        return ase.visualize.view(system, viewer="x3d")
