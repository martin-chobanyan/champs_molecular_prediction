#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from dtsckit.utils import write_pickle
from pairwise_features import Prepare1JH_, Prepare2JHH, Prepare2JH_, Prepare3JHH, Prepare3JH_


def process_filename(filename):
    coupling_type, = re.findall(r'data_(.*)\.csv', filename)
    num_hops = int(coupling_type[0])
    h2h = (coupling_type[-2:] == 'HH')
    return coupling_type, num_hops, h2h


def get_faulty_pairs(df):
    mask = (df['x_nitrogen'] == -1)
    ids = df.loc[mask]['id'].values
    return mask, ids


def get_feature_columns(coupling_type):
    if coupling_type == '1JHC' or coupling_type == '1JHN':
        return Prepare1JH_.feature_cols()
    elif coupling_type == '2JHH':
        return Prepare2JHH.feature_cols()
    elif coupling_type == '2JHC' or coupling_type == '2JHN':
        return Prepare2JH_.feature_cols()
    elif coupling_type == '3JHH':
        return Prepare3JHH.feature_cols()
    elif coupling_type == '3JHC' or coupling_type == '3JHN':
        return Prepare3JH_.feature_cols()


if __name__ == '__main__':
    ROOT_DIR = '/home/mchobanyan/data/kaggle/molecules/'
    TRAIN_DIR = os.path.join(ROOT_DIR, 'train')
    TEST_DIR = os.path.join(ROOT_DIR, 'test')

    submission_filepath = os.path.join(ROOT_DIR, 'submission.csv')
    submission_df = pd.read_csv(submission_filepath)
    submission_df['scalar_coupling_constant'] = 0
    submission_df.index = submission_df['id'].values

    models = dict()
    scores = dict()
    feature_importances = dict()
    total_time = 0
    for filename in os.listdir(TRAIN_DIR):
        ###################################### Prepare the training data ###############################################
        start_time = time.time()
        coupling_type, num_hops, h2h = process_filename(filename)
        print(f'\nTraining model for {coupling_type}')
        train_df = pd.read_csv(os.path.join(TRAIN_DIR, filename))

        if num_hops > 1:
            faulty_molecule_mask, _ = get_faulty_pairs(train_df)
            train_df = train_df.loc[~faulty_molecule_mask]
            train_df = train_df.reset_index(drop=True)

        feature_columns = get_feature_columns(coupling_type)
        x_train = train_df[feature_columns].values
        y_train = train_df['scalar_coupling_constant'].values
        ######################################## Hyperparameter tuning #################################################
        # params = [{'min_samples_leaf': [10, 20, 50],
        #            'max_depth': [3, 5, 8, None],
        #            'max_features': ['sqrt', 0.5]}]
        #
        # clf = GridSearchCV(RandomForestRegressor(300), params, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
        # clf.fit(x_train, y_train)
        #
        # score = np.log(-1 * clf.best_score_)
        # scores[coupling_type] = score
        # print(f'Log MAE Score: {score}')
        # print(f'Best parameters: {clf.best_params_}')
        ######################################## Train on all the data #################################################
        # model = clf.best_estimator_
        model = RandomForestRegressor(n_estimators=300, max_depth=None, max_features=0.5, min_samples_leaf=20, n_jobs=10)
        model.fit(x_train, y_train)
        models[coupling_type] = model

        print('Feature importances:')
        feature_importances[coupling_type] = dict()
        for feature, importance in zip(feature_columns, model.feature_importances_):
            print(f'{feature}: {importance}')
            feature_importances[coupling_type][feature] = importance
        ###################################### Prepare the testing data ################################################
        test_df = pd.read_csv(os.path.join(TEST_DIR, filename))

        if num_hops > 1:
            faulty_molecule_mask, _ = get_faulty_pairs(test_df)
            print(f'Number of faulty pairs: {faulty_molecule_mask.sum()}')
            test_df = test_df.loc[~faulty_molecule_mask]
            test_df = test_df.reset_index(drop=True)

        x_test = test_df[feature_columns].values
        ########################################## Make predictions ####################################################
        predictions = model.predict(x_test)
        submission_df.loc[test_df['id'].values, 'scalar_coupling_constant'] = predictions

        elapsed_time = (time.time() - start_time) / 3600
        print(f'Time elapsed: {elapsed_time} hours')
        total_time += elapsed_time
        ################################################################################################################

    print(f'\nTotal time elapsed: {total_time} hours')
    print('\nSaving the submissions...')
    write_pickle(models, os.path.join(ROOT_DIR, 'models/pairwise/rf_models.pkl'))
    write_pickle(scores, os.path.join(ROOT_DIR, 'models/pairwise/rf_scores.pkl'))
    write_pickle(feature_importances, os.path.join(ROOT_DIR, 'models/pairwise/rf_feature_importance.pkl'))
    submission_df.to_csv(submission_filepath, index=False)
    print('Done!')
