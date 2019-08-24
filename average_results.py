#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""This script performs a weighted average of different submission results into a final submission csv file"""
import os
import numpy as np
import pandas as pd


SUBMISSION_DIR = '/home/mchobanyan/data/kaggle/molecules/submissions'
output_path = os.path.join(SUBMISSION_DIR, 'submission.csv')

index = None
predictions = None
weights = {'pairwise': 0.1, 'edgeconv_acsf_835': 0.1, 'edgeconv_lmbtr_1140': 0.8}
for model_name, weight in weights.items():
    print(model_name)
    df = pd.read_csv(os.path.join(SUBMISSION_DIR, f'{model_name}_submission.csv'))
    if index is None:
        index = df['id'].values
        predictions = np.zeros(len(index))
    predictions += weight * df['scalar_coupling_constant'].values

df = pd.DataFrame({'id': index, 'scalar_coupling_constant': predictions})
df.to_csv(output_path, index=False)
