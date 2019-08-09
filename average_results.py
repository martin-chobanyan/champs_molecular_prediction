#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script performs a weighted average of different submission results into a final submission csv file"""
import os
import pandas as pd


SUBMISSION_DIR = '/home/mchobanyan/data/kaggle/molecules/submissions'
output_path = os.path.join(SUBMISSION_DIR, 'submission.csv')

rf_df = pd.read_csv(os.path.join(SUBMISSION_DIR, 'pairwise_submission.csv'))
gat_df = pd.read_csv(os.path.join(SUBMISSION_DIR, 'gat_submission.csv'))

df = rf_df.merge(gat_df, on=['id'], suffixes=['_rf', '_gat'])
df['scalar_coupling_constant'] = 0.4 * df['scalar_coupling_constant_rf'] + 0.6 * df['scalar_coupling_constant_gat']
df = df[['id', 'scalar_coupling_constant']]
df.to_csv(output_path, index=False)
