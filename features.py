#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import torch

from ase.atoms import Atoms as ase_atoms
from dscribe.descriptors import ACSF, LMBTR, SOAP
from mendeleev import element
from rdkit.Chem import Mol
import rdkit.Chem.rdMolDescriptors as rdMD


MAX_DISTANCE = 12.040386089286708


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
                pass
                # coulomb[i, i] = 0.5 * (charges[i] ** 2.4)
            else:
                coulomb[i, j] = (charges[i] * charges[j]) / distance_matrix[i, j]
    return coulomb


########################################################################################################################
#                                             Define edge features
########################################################################################################################


class EdgeFeatures(object):
    def __init__(self, molecule_map, max_distance=MAX_DISTANCE):
        self.molecule_map = molecule_map
        self.max_distance = max_distance

    def __call__(self, molecule_name):
        edge_features = []
        molecule = self.molecule_map[molecule_name]
        molecule_obj = molecule['rdkit']
        edge_index = molecule['bonds']
        distances = molecule['distance_matrix']

        for i, j in edge_index.transpose(1, 0).tolist():
            bond = molecule_obj.GetBondBetweenAtoms(i, j)
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()

            features = encode_bond_order(str(bond.GetBondType()))
            features += encode_element(begin_atom.GetSymbol())
            features += encode_element(end_atom.GetSymbol())

            d = distances[i, j]
            features.append(d / self.max_distance)

            features.append((begin_atom.GetAtomicNum() * end_atom.GetAtomicNum()) / (81 * d))
            edge_features.append(features)

        return np.array(edge_features)


########################################################################################################################
#                                            Define local molecule descriptors
########################################################################################################################


class BaseAtomicFeatures(object):
    def __init__(self, molecule_map):
        self.molecule_map = molecule_map
        self.elements = {'H': element('H'), 'C': element('C'), 'N': element('N'), 'O': element('O'), 'F': element('F')}

    def __call__(self, molecule_name):
        atomic_features = []
        molecule = self.molecule_map[molecule_name]
        symbols = molecule['symbols']
        gasteiger_charges = molecule['g_charges']
        eem_charges = molecule['eem_charges']
        mul_charges = molecule['m_charges'] if 'm_charges' in molecule else None
        shields = molecule['symmetric_tensor_shields'] if 'symmetric_tensor_shields' in molecule else None
        for i, atom in enumerate(molecule['rdkit'].GetAtoms()):
            features = [gasteiger_charges[i], eem_charges[i], mul_charges[i]]
            features += list(shields[i])
            features += encode_hybridization(str(atom.GetHybridization()))
            features += [int(atom.IsInRingSize(ring_size)) for ring_size in range(3, 10)]

            # add elemental features of the atom and normalize them with hardcoded maximums
            e = self.elements[symbols[i]]
            features.append(e.atomic_number / 9)
            features.append(e.en_pauling / 3.98)
            features.append(e.atomic_radius_rahm / 190)
            features.append(e.dipole_polarizability / 11.3)

            atomic_features.append(features)
        return np.array(atomic_features)


class DscribeAtomicFeatures(object):
    """Base class for all 'dscribe' based local/global molecule descriptors"""

    def __init__(self, molecule_map, n_jobs):
        self.molecule_map = molecule_map
        self.species = ['H', 'C', 'N', 'O', 'F']
        self.dscribe_func = None
        self.n_jobs = n_jobs

    def __call__(self, molecule_name):
        molecule_dict = self.molecule_map[molecule_name]
        molecule = ase_atoms(symbols=molecule_dict['symbols'], positions=molecule_dict['coords'])
        features = self.dscribe_func.create(molecule, n_jobs=self.n_jobs)
        return features


class AtomCenteredSymmetryFeatures(DscribeAtomicFeatures):
    def __init__(self, molecule_map, r_cut, g2_params=None, g3_params=None, g4_params=None, g5_params=None, n_jobs=1):
        super().__init__(molecule_map, n_jobs)
        self.r_cut = r_cut
        self.g2_params = g2_params
        self.g3_params = g3_params
        self.g4_params = g4_params
        self.g5_params = g5_params
        self.dscribe_func = ACSF(species=self.species,
                                 rcut=r_cut,
                                 g2_params=g2_params,
                                 g3_params=g3_params,
                                 g4_params=g4_params,
                                 g5_params=g5_params)


class SOAPFeatures(DscribeAtomicFeatures):
    def __init__(self, molecule_map, r_cut, n_max, l_max, n_jobs=1):
        super().__init__(molecule_map, n_jobs)
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.dscribe_func = SOAP(species=self.species, periodic=False, rcut=r_cut, nmax=n_max, lmax=l_max)

    def __call__(self, molecule_name):
        return super().__call__(molecule_name) / 1000


class LocalManyBodyTensorFeatures(DscribeAtomicFeatures):
    def __init__(self, molecule_map, k2_params=None, k3_params=None, n_jobs=1):
        super().__init__(molecule_map, n_jobs)
        self.k2_params = k2_params
        self.k3_params = k3_params
        self.dscribe_func = LMBTR(species=self.species,
                                  k2=k2_params,
                                  k3=k3_params,
                                  periodic=False,
                                  normalization='l2_each')


########################################################################################################################
#                                          Define global molecule descriptors
########################################################################################################################


def calculate_scalar_descriptors(molecule, symbols):
    features = list()
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


def encode_element(symbol):
    if symbol == 'H':
        return [1, 0, 0, 0, 0]
    elif symbol == 'C':
        return [0, 1, 0, 0, 0]
    elif symbol == 'N':
        return [0, 0, 1, 0, 0]
    elif symbol == 'O':
        return [0, 0, 0, 1, 0]
    elif symbol == 'F':
        return [0, 0, 0, 0, 1]
    else:
        raise ValueError(f'Encountered an unexpected element: {symbol}')


def encode_hybridization(hybridization):
    if hybridization == 'S':
        return [1, 0, 0, 0]
    elif hybridization == 'SP':
        return [0, 1, 0, 0]
    elif hybridization == 'SP2':
        return [0, 0, 1, 0]
    elif hybridization == 'SP3':
        return [0, 0, 0, 1]
    else:
        raise ValueError(f'Encountered an unexpected hybridization: {hybridization}')


def encode_bond_order(bond_order):
    if bond_order == 'SINGLE' or bond_order == 1:
        return [1, 0, 0, 0]
    elif bond_order == 'AROMATIC' or bond_order == 1.5:
        return [0, 1, 0, 0]
    elif bond_order == 'DOUBLE' or bond_order == 2:
        return [0, 0, 1, 0]
    elif bond_order == 'TRIPLE' or bond_order == 3:
        return [0, 0, 0, 1]
    else:
        raise ValueError(f'Encountered an unexpected bond order: {bond_order}')
