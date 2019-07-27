#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
from rdkit.Chem import Mol
import rdkit.Chem.rdMolDescriptors as rdMD


########################################################################################################################
#                                     Define matrix representations of molecules
########################################################################################################################


def get_bond_order(bond_type):
    """Map the bond_type string to its bond order"""
    if bond_type == 'single':
        return 1
    elif bond_type == 'aromatic':
        return 1.5
    elif bond_type == 'double':
        return 2
    elif bond_type == 'triple':
        return 3
    else:
        raise ValueError(f'Unexpected bond type: {bond_type.upper()}')


def calculate_connectivity_matrix(molecule, element_diagonal=False):
    """Calculates a 2D adjacency matrix weighted by bond orders

    Parameters
    ----------
    molecule: Mol
    element_diagonal: bool
        If true, then the atomic numbers of each node will be included in the diagonal (default=False).

    Returns
    -------
    np.ndarray
    """
    num_atoms = molecule.GetNumAtoms()
    adjacency = np.zeros((num_atoms, num_atoms))
    for bond in molecule.GetBonds():
        bond_type = str(bond.GetBondType()).lower()
        bond_order = get_bond_order(bond_type)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        adjacency[i, j] = bond_order
        adjacency[j, i] = bond_order
    if element_diagonal:
        for i, atom in enumerate(molecule.GetAtoms()):
            adjacency[i, i] = atom.GetAtomicNum()
    return adjacency


def calculate_cep_matrix(molecule):
    """Calculates the weighted electronic connectivity (CEP) matrix
    Source: http://revroum.lew.ro/wp-content/uploads/2006/RRC_11_2006/art10Berinde.pdf

    Parameters
    ----------
    molecule: Mol

    Returns
    -------
    np.ndarray
    """
    num_atoms = molecule.GetNumAtoms()
    charges = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    cep_matrix = np.zeros((num_atoms, num_atoms))
    for bond in molecule.GetBonds():
        bond_type = str(bond.GetBondType()).lower()
        bond_order = get_bond_order(bond_type)
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        degree_i = len(molecule.GetAtomWithIdx(i).GetNeighbors())
        degree_j = len(molecule.GetAtomWithIdx(j).GetNeighbors())
        z_i = degree_i * charges[i]
        z_j = degree_j * charges[j]
        weighted_electronic_distance = ((z_i + z_j) / (bond_order * degree_i * degree_j))
        cep_matrix[i, j] = weighted_electronic_distance
        cep_matrix[j, i] = weighted_electronic_distance
    return cep_matrix


def calculate_coulomb_matrix(molecule, distance_matrix):
    """Calculate the Coulomb matrix of the given molecule

    Parameters
    ----------
    molecule: Mol
    distance_matrix: np.ndarray

    Returns
    -------
    np.ndarray
        A 2D array with the atom-atom coulomb interactions
    """
    num_atoms = molecule.GetNumAtoms()
    charges = [atom.GetAtomicNum() for atom in molecule.GetAtoms()]
    coulomb = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                coulomb[i, i] = 0.5 * (charges[i] ** 2.4)
            else:
                coulomb[i, j] = (charges[i] * charges[j]) / distance_matrix[i, j]
    return coulomb


########################################################################################################################
#                                       Define molecular feature vectors
########################################################################################################################


def calculate_scalar_descriptors(molecule, symbols):
    features = []
    features.append(rdMD.CalcAsphericity(molecule))
    features += list(rdMD.CalcCrippenDescriptors(molecule))
    features.append(rdMD.CalcExactMolWt(molecule))
    features.append(rdMD.CalcEccentricity(molecule))
    features.append(rdMD.CalcFractionCSP3(molecule))
    features.append(rdMD.CalcLabuteASA(molecule))
    features.append(rdMD.CalcNPR1(molecule))
    features.append(rdMD.CalcNPR2(molecule))
    features.append(rdMD.CalcHallKierAlpha(molecule))

    # elemental distribution
    symbols = np.array(symbols)
    features.append(np.sum(symbols == 'H'))
    features.append(np.sum(symbols == 'C'))
    features.append(np.sum(symbols == 'N'))
    features.append(np.sum(symbols == 'O'))
    features.append(np.sum(symbols == 'F'))

    # ring features
    features.append(rdMD.CalcNumAliphaticCarbocycles(molecule))
    features.append(rdMD.CalcNumAliphaticHeterocycles(molecule))
    features.append(rdMD.CalcNumAromaticCarbocycles(molecule))
    features.append(rdMD.CalcNumAromaticHeterocycles(molecule))
    features.append(rdMD.CalcNumSaturatedCarbocycles(molecule))
    features.append(rdMD.CalcNumSaturatedHeterocycles(molecule))
    features.append(rdMD.CalcNumSpiroAtoms(molecule))  # atom shared between rings with one bond
    features.append(rdMD.CalcNumBridgeheadAtoms(molecule))  # atom shared between rings with at least two bonds

    # other counts
    features.append(rdMD.CalcNumAmideBonds(molecule))
    features.append(rdMD.CalcNumHBA(molecule))  # number of hydrogen acceptors
    features.append(rdMD.CalcNumHBD(molecule))  # number of hydrogen donors

    return np.array(features)


########################################################################################################################
#                                          Define the bond features
########################################################################################################################


def get_edge_index(molecule):
    bonds = []
    for bond in molecule.GetBonds():
        ai = bond.GetBeginAtomIdx()
        aj = bond.GetEndAtomIdx()
        bonds.append((ai, aj))
        bonds.append((aj, ai))
    edge_index = torch.LongTensor(bonds).transpose(1, 0)
    return edge_index


def get_edge_attributes(molecule, dist_mtrx):
    edge_attributes = []
    for bond in molecule.GetBonds():
        x = dict()
        x['start_atom'] = bond.GetBeginAtom().GetSymbol()
        x['end_atom'] = bond.GetEndAtom().GetSymbol()
        x['type'] = str(bond.GetBondType())
        x['conjugated'] = int(bond.GetIsConjugated())
        x['distance'] = dist_mtrx[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        edge_attributes.append(x)
    return edge_attributes


########################################################################################################################
#                                           Miscellaneous routines
########################################################################################################################


class ElementEncoder(object):
    """One-hot encode the elements used in molecules"""

    def __init__(self):
        base_symbols = ['H', 'C', 'N', 'O', 'F']
        self.label_enc = LabelEncoder()
        self.label_enc.fit_transform(base_symbols)
        element_labels = self.label_enc.transform(base_symbols)

        self.onehot_enc = OneHotEncoder(categories='auto', sparse=False)
        self.onehot_enc = self.onehot_enc.fit(np.expand_dims(element_labels, 1))

    def __call__(self, symbols):
        labels = self.label_enc.transform(symbols)
        labels = np.expand_dims(labels, 1)
        encoding = self.onehot_enc.transform(labels)
        return encoding


def get_unique_hybridizations(molec_struct_map):
    hybridization_set = set()
    for name in molec_struct_map:
        molecule_dict = molec_struct_map[name]
        molecule = molecule_dict['rdkit']
        h = [str(atom.GetHybridization()) for atom in molecule.GetAtoms()]
        hybridization_set.update(h)
    return hybridization_set
