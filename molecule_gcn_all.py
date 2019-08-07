#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from multiprocessing import Pool, Process
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from dtsckit.model import AverageKeeper
from dtsckit.utils import read_pickle, write_pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader

from features import *
from molecule_gcn import GCNSequential


def standardize_couplings(data, coupling_types):
    scalers = dict()
    for coupling_type in coupling_types:
        standardizer = StandardScaler()
        data.loc[data['type'] == coupling_type, 'scalar_coupling_constant'] = standardizer.fit_transform(
            data.loc[data['type'] == coupling_type, 'scalar_coupling_constant'].values.reshape(-1, 1)
        ).squeeze()
        scalers[coupling_type] = standardizer
    return scalers


def unstandardize_couplings(data, scalers, col='scalar_coupling_constant'):
    for coupling_type, standardizer in scalers.items():
        data.loc[data['type'] == coupling_type, col] = standardizer.inverse_transform(
            data.loc[data['type'] == coupling_type, col].values.reshape(-1, 1)
        ).squeeze()


if __name__ == '__main__':
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    MAX_MOL_SIZE = 29

    print('Reading the training and testing data...')
    train_df = pd.read_csv(os.path.join(ROOT_DIR, 'train.csv'))
    test_df = pd.read_csv(os.path.join(ROOT_DIR, 'test.csv'))

    print('Reading the molecular structures...')
    molecule_map = read_pickle(os.path.join(ROOT_DIR, 'molecular_structure_map.pkl'))
    num_molecules = len(molecule_map)

    print('Reading ACSF features...')
    acsf_train = read_pickle(os.path.join(ROOT_DIR, 'acsf_train_graphs.pkl'))
    acsf_test = read_pickle(os.path.join(ROOT_DIR, 'acsf_val_graphs.pkl'))
    mol_graphs = acsf_train + acsf_test

    print('Reading submission file...')
    submission_filepath = os.path.join(ROOT_DIR, 'submissions/gat_submission.csv')
    submission_df = pd.read_csv(os.path.join(ROOT_DIR, 'submissions/submission.csv'))
    submission_df['scalar_coupling_constant'] = 0
    submission_df.index = submission_df['id'].values

    # calculate the max distance between atoms across all molecules
    r_c = 0.0
    for name in molecule_map:
        if 'scalar' not in name:
            distances = molecule_map[name]['distance_matrix']
            r_c = max(r_c, distances.max())

    # create maps from coupling types to integer labels
    coupling_types = train_df['type'].unique()
    coupling_idx = {c: i for i, c in enumerate(coupling_types)}
    idx_coupling = {i: c for i, c in enumerate(coupling_types)}

    # encode the coupling types with integers {0...7}
    train_df['encoded_type'] = train_df['type'].map(coupling_idx)
    test_df['encoded_type'] = test_df['type'].map(coupling_idx)

    scalers = standardize_couplings(train_df, coupling_types)

    num_features = 171

    data_loader = DataLoader()




