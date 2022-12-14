"""Trains a chemprop model on a dataset by doing domain classification followed by regression with re-weighting."""

import chemprop
import argparse
import subprocess
import itertools
import os
import pandas as pd
import numpy as np
import json

# retrieve index of training folder
parser = argparse.ArgumentParser(description='driver code for domain adapt pipeline')
parser.add_argument('--folder_index', type=str, help='index of training data folder', required=True)
parser.add_argument('--parent_folder_name', type=str, help='name of parent data folder', required=True)
args = parser.parse_args()

parent_folder_name = args.parent_folder_name
parentdir = os.getcwd()
folder = args.folder_index

df_target_holdout = pd.read_csv(os.path.join(parentdir, parent_folder_name, str(folder),'target_holdout.csv'))
target_holdout_logp = np.array(df_target_holdout.logp)

def calc_rmse(preds):
    preds_sanitized = [pred[0] for pred in preds]
    rmse = np.sqrt(((np.array(preds_sanitized) - target_holdout_logp)**2).mean())
    print(rmse)
    return rmse

def extract_weights():
    df = pd.read_csv(os.path.join(parentdir, parent_folder_name, str(folder), 'classification', 'preds.csv') )
    p_target = df['label'].to_numpy()
    data_weights = p_target/(1-p_target)
    df = pd.DataFrame(data_weights, columns=['data_weights'])
    df.to_csv( os.path.join(parentdir, parent_folder_name, str(folder), 'classification', 'data_weights.csv'), index=False)
    return


def train(dataset_fname, task, save_folder, **kwargs): 
    if task == 'regression':
        target_col = 'logp'
    elif task == 'classification':
        target_col = 'label'
    arguments = [
        '--data_path', os.path.join(parentdir, parent_folder_name, str(folder), dataset_fname),
        '--dataset_type', task,
        '--save_dir', os.path.join(parentdir, parent_folder_name, str(folder), save_folder),
        '--target_columns', target_col,
        '--smiles_columns', 'smiles'
    ]
    if 'data_weights_path' in kwargs:
        data_weights_path = kwargs['data_weights_path']
        arguments.append('--data_weights_path')
        arguments.append(data_weights_path)

    args = chemprop.args.TrainArgs().parse_args(arguments)
    mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training) 
    return mean_score

def predict(dataset_fname, task, save_folder):
    arguments = [
        '--test_path', os.path.join(parentdir, parent_folder_name, str(folder), dataset_fname),
        '--preds_path', os.path.join(parentdir, parent_folder_name, str(folder), save_folder, 'preds.csv'),
        '--checkpoint_dir', os.path.join(parentdir, parent_folder_name, str(folder), save_folder)
    ]
    args = chemprop.args.PredictArgs().parse_args(arguments)
    preds = chemprop.train.make_predictions(args=args)
    return preds


# classification
rmse_test_classification = train('classification.csv', 'classification', 'classification')

# infer classification labels on source
predict('source.csv', 'classification', 'classification')

# train regression on source without reweighting
rmse_test_regression_source = train('source.csv', 'regression', 'regression_source')

# regression on target without re-weighting
rmse_test_regression_target = train('target_unlabeled.csv', 'regression', 'regression_target')

# source regression inference on heldout target
preds = predict('target_holdout.csv', 'regression', 'regression_source')
source_on_target = calc_rmse(preds)

# target regression inference on heldout target
preds = predict('target_holdout.csv', 'regression', 'regression_target')
target_on_target = calc_rmse(preds)

# regression on source with re-weighting
extract_weights()
data_weights_path = os.path.join(parentdir, parent_folder_name, str(folder), 'classification', 'data_weights.csv')
rmse_test_regression_weighted = train('source.csv', 'regression', 'regression_target_weighted', data_weights_path=data_weights_path)

# regression inference on heldout target
preds = predict('target_holdout.csv', 'regression', 'regression_target_weighted')
weighted_source_on_target = calc_rmse(preds)

results = {
    'rmse_test_classification': rmse_test_classification,
    'rmse_test_regression_source': rmse_test_regression_source,
    'rmse_test_regression_target': rmse_test_regression_target,
    'rmse_test_regression_weighted': rmse_test_regression_weighted, 
    'source_on_target': source_on_target,
    'target_on_target': target_on_target,
    'weighted_source_on_target': weighted_source_on_target
}

with open(os.path.join(parentdir, parent_folder_name, str(folder), 'results.json'), 'w') as out:
    json.dump(results, out)