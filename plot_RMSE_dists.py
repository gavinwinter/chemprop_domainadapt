"""Trains a chemprop model on a dataset by doing domain classification followed by regression with re-weighting."""

import chemprop
import argparse
import subprocess
import itertools
import os
import pandas as pd
import numpy as np
import json

import matplotlib.pyplot as plt



parent_folder_name = 'remaining_smaller_runs/smaller_C1CCCCC1_C1CCNCC1_run'
parentdir = os.getcwd()
folder = '0'

df_target_holdout = pd.read_csv(os.path.join(parentdir, parent_folder_name, str(folder),'target_holdout.csv'))
target_holdout_logp = np.array(df_target_holdout.logp)

fig, ax = plt.subplots(1, figsize=(4.5, 3.75))


def calc_rmse(preds):
    preds_sanitized = [pred[0] for pred in preds]
    rmse = np.sqrt(((np.array(preds_sanitized) - target_holdout_logp)**2))
    print(rmse)
    return rmse


def predict(dataset_fname, task, save_folder):
    arguments = [
        '--test_path', os.path.join(parentdir, parent_folder_name, str(folder), dataset_fname),
        '--preds_path', os.path.join(parentdir, parent_folder_name, str(folder), save_folder, 'preds_hist_plot.csv'),
        '--checkpoint_dir', os.path.join(parentdir, parent_folder_name, str(folder), save_folder)
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    return preds

source_on_target_preds = predict('target_holdout.csv', 'regression', 'regression_source')
target_on_target_preds = predict('target_holdout.csv', 'regression', 'regression_target')
weighted_source_on_target_preds = predict('target_holdout.csv', 'regression', 'regression_target_weighted')

rmse_source_on_target = calc_rmse(source_on_target_preds)
rmse_target_on_target = calc_rmse(target_on_target_preds)
rmse_weighted_source_on_target = calc_rmse(weighted_source_on_target_preds)
print(len(rmse_source_on_target), len(rmse_target_on_target), len(rmse_weighted_source_on_target))

bins = np.histogram(rmse_source_on_target + rmse_target_on_target + rmse_weighted_source_on_target, bins=200)[1]
ax.hist(rmse_source_on_target, bins=bins, alpha=0.6, density=True, label = 'Source regressor')
ax.hist(rmse_target_on_target, bins = bins, alpha=0.6, density=True, label = 'Target regressor')
ax.hist(rmse_weighted_source_on_target, bins=bins, alpha=0.6, density=True, label = 'Weighted source regressor')
ax.legend(frameon=False)
ax.set_xlim([0,1])
ax.set_xlabel('RMSE on holdout target')
ax.set_ylabel('Frequency density')
fig.tight_layout()
plt.savefig('{}_{}.png'.format(parent_folder_name, folder), dpi=300)
plt.savefig('{}_{}.svg'.format(parent_folder_name, folder), dpi=300)
