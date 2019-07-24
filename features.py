#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch
import rdkit.Chem.rdMolDescriptors as rdMD
from chem_math import angle_between, find_atomic_path


def calculate_coulomb_matrix(molecule):
    """Calculate the Coulomb matrix of the given molecule

    Parameters
    ----------
    molecule: dict
        A dictionary with keys for 'distance_matrix', 'symbols', and 'rdkit' (maps to the rdkit object for the molecule)

    Returns
    -------
    np.ndarray
        A 2D array with the atom-atom coulomb interactions
    """
    m = molecule['rdkit']
    distances = molecule['distance_matrix']
    num_atoms = len(molecule['symbols'])
    charges = [atom.GetAtomicNum() for atom in m.GetAtoms()]

    coulomb = np.zeros((num_atoms, num_atoms))
    for i in range(num_atoms):
        for j in range(num_atoms):
            if i == j:
                coulomb[i, j] = 0.5 * (charges[i] ** 2.4)
            else:
                coulomb[i, j] = (charges[i] * charges[j]) / distances[i, j]

    return coulomb


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


def get_unique_hybridizations(molec_struct_map):
    hybridization_set = set()
    for name in molec_struct_map:
        molecule_dict = molec_struct_map[name]
        molecule = molecule_dict['rdkit']
        h = [str(atom.GetHybridization()) for atom in molecule.GetAtoms()]
        hybridization_set.update(h)
    return hybridization_set


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


# OLD CODE:
def extract_basic_features(name, a0, a1, molec_struct_map, bond_separation=1):
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
        f = extract_basic_features(name, a0, a1, molec_struct_map, bond_separation)
        features.append(f)

    features = np.stack(features)
    return features
