import math
import pandas as pd
import numpy as np
import os
from itertools import permutations
from tqdm import tqdm


def generate_data(path_A_master, path_B_master, size_source, size_target_unlabeled, size_target_holdout, mix_fraction, folder_name):

    os.makedirs(folder_name)

    num_A_samples_in_source = math.ceil(size_source * mix_fraction)
    num_A_samples_in_target_unlabeled = math.ceil(size_target_unlabeled * (1 - mix_fraction))
    num_A_samples_in_target_holdout = math.ceil(size_target_holdout * (1 - mix_fraction))

    end_source_A_ind = num_A_samples_in_source
    end_target_unlabeled_A_ind = num_A_samples_in_source + num_A_samples_in_target_unlabeled


    num_B_samples_in_source = math.ceil(size_source * (1 - mix_fraction))
    num_B_samples_in_target_unlabeled = math.ceil(size_target_unlabeled * mix_fraction)
    num_B_samples_in_target_holdout = math.ceil(size_target_holdout * mix_fraction)

    end_source_B_ind = num_B_samples_in_source
    end_target_unlabeled_B_ind = num_B_samples_in_source + num_B_samples_in_target_unlabeled


    num_A_samples = num_A_samples_in_source + num_A_samples_in_target_unlabeled + num_A_samples_in_target_holdout
    num_B_samples = num_B_samples_in_source + num_B_samples_in_target_unlabeled + num_B_samples_in_target_holdout


    A_master_df = pd.read_csv(path_A_master)
    B_master_df = pd.read_csv(path_B_master)

    A_sample_df = A_master_df.sample(num_A_samples)
    B_sample_df = B_master_df.sample(num_B_samples)


    source_df = pd.concat([A_sample_df.iloc[:end_source_A_ind, :], B_sample_df.iloc[:end_source_B_ind]])
    target_unlabeled_df = pd.concat([A_sample_df.iloc[end_source_A_ind : end_target_unlabeled_A_ind, :], B_sample_df.iloc[end_source_B_ind : end_target_unlabeled_B_ind]])
    target_holdout_df = pd.concat([A_sample_df.iloc[end_target_unlabeled_A_ind:, :], B_sample_df.iloc[end_target_unlabeled_B_ind:]])

    source_df['label'] = 0
    target_unlabeled_df['label'] = 1
    classification_df = pd.concat([source_df, target_unlabeled_df])


    if(False):
        # Verification prints, calculated numbers of samples

        print(f'A in s: {num_A_samples_in_source}')
        print(f'A in t unlab: {num_A_samples_in_target_unlabeled}')
        print(f'A in t hold: {num_A_samples_in_target_holdout}')

        print(f'B in s: {num_B_samples_in_source}')
        print(f'B in t unlab: {num_B_samples_in_target_unlabeled}')
        print(f'B in t hold: {num_B_samples_in_target_holdout}')

        # Verification prints, actual number of samples
        
        print(f"A in s: {len(source_df[source_df['scaffold'] == 'c1ccccc1'])}")
        print(f"A in t unlab: {len(target_unlabeled_df[target_unlabeled_df['scaffold'] == 'c1ccccc1'])}")
        print(f"A in t hold: {len(target_holdout_df[target_holdout_df['scaffold'] == 'c1ccccc1'])}")

        print(f"B in s: {len(source_df[source_df['scaffold'] == 'c1cn[nH]c1'])}")
        print(f"B in t unlab: {len(target_unlabeled_df[target_unlabeled_df['scaffold'] == 'c1cn[nH]c1'])}")
        print(f"B in t hold: {len(target_holdout_df[target_holdout_df['scaffold'] == 'c1cn[nH]c1'])}")
    
    
    source_df.sample(frac=1).to_csv(f'{folder_name}/source.csv', columns=['smiles', 'logp'], index=False)
    target_unlabeled_df.sample(frac=1).to_csv(f'{folder_name}/target_unlabeled.csv', columns=['smiles', 'logp'], index=False)
    target_holdout_df.sample(frac=1).to_csv(f'{folder_name}/target_holdout.csv', columns=['smiles', 'logp'], index=False)
    classification_df.sample(frac=1).to_csv(f'{folder_name}/classification.csv', columns=['smiles', 'label'], index=False)



def generate_datasets(run_tag, path_A_master, path_B_master, size_source, size_target_unlabeled, size_target_holdout, mix_fractions, num_trials):

    log = np.zeros((len(mix_fractions) * num_trials, 3))

    folder_index = 0
    for mix_fraction in mix_fractions:
        for i in range(num_trials):
            generate_data(path_A_master, path_B_master, size_source, size_target_unlabeled, size_target_holdout, mix_fraction, f'{run_tag}/{folder_index}')
            log[folder_index, :] = [folder_index, mix_fraction, i]
            folder_index += 1
    
    np.savetxt(f'{run_tag}/folder_index_log.csv', log, header='folder_index mix_fraction trial')