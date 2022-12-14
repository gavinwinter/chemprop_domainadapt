import os, re, json
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

figs, axes = plt.subplots(4,4, figsize=(9,7.5), sharey=True)
x_axes = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7]


scaffolds_to_indices = {
    'C1CCCCC1': 0,
    'C1CCNCC1': 1,
    'OeC(Nc1ccccc1)c1ccccc1': 2,
    'OeC(NCc1ccccc1)c1ccccc1': 3
}
parentdir = os.getcwd()

def single_plot(ax, x_axes, means_source_on_targets, means_target_on_targets, means_weighted_source_on_targets, x, y):
    if x == 3:
        ax.set_xlabel('Segregation fraction $x$')
    if y == 0:
        ax.set_ylabel('RMSE on holdout\n target data')
    
    # import pdb; pdb.set_trace()
    ax.plot(x_axes, means_source_on_targets, marker = 'o', label='Source regressor')
    # ax.fill_between(x_axes, np.array(mins_source_on_targets), np.array(maxes_source_on_targets), alpha=0.2)
    ax.fill_between(x_axes, np.array(means_source_on_targets) - np.array(stds_source_on_targets), np.array(means_source_on_targets) + np.array(stds_source_on_targets), alpha=0.2)


    ax.plot(x_axes, means_target_on_targets, marker = 'o', label='Target regressor')
    # ax.fill_between(x_axes, np.array(mins_target_on_targets), np.array(maxes_target_on_targets), alpha=0.2)
    ax.fill_between(x_axes, np.array(means_target_on_targets) - np.array(stds_target_on_targets), np.array(means_target_on_targets) + np.array(stds_target_on_targets), alpha=0.2)


    ax.plot(x_axes, means_weighted_source_on_targets, marker = 'o', label='Weighted source regressor')
    # ax.fill_between(x_axes, np.array(mins_weighted_source_on_targets), np.array(maxes_weighted_source_on_targets), np.array(alpha=0.2)
    ax.fill_between(x_axes, np.array(means_weighted_source_on_targets) - np.array(stds_weighted_source_on_targets), np.array(means_weighted_source_on_targets) + np.array(stds_weighted_source_on_targets), alpha=0.2)
    # ax.set_xlabel('Fraction of majority scaffold in source data')
    # ax.set_ylabel('RMSE on holdout target data')
    ax.invert_xaxis()
    
for i, parent_folder in tqdm(enumerate(os.listdir('remaining_smaller_runs'))):
    print(parent_folder)
    
    means_source_on_targets = []
    mins_source_on_targets = []
    maxes_source_on_targets = []
    stds_source_on_targets = []


    means_target_on_targets = []
    maxes_target_on_targets = []
    mins_target_on_targets = []
    stds_target_on_targets = []

    means_weighted_source_on_targets = []
    maxes_weighted_source_on_targets = []
    mins_weighted_source_on_targets = []
    stds_weighted_source_on_targets = []

    for i, folder in tqdm(enumerate(range(0,60))):
        print(i)
        if i == 0:
            source_on_targets = []
            target_on_targets = []
            weighted_source_on_targets = []
        
        try:
            results = json.load(open(os.path.join(parentdir, 'remaining_smaller_runs', parent_folder, str(folder), 'results.json')))
        except:
            print('Skipping ', os.path.join(parentdir, 'remaining_smaller_runs', parent_folder, str(folder), 'results.json'))
            continue
        source_on_targets.append(results['source_on_target'])
        target_on_targets.append(results['target_on_target'])
        weighted_source_on_targets.append(results['weighted_source_on_target'])

        if i % 10 == 9:
            mean_source_on_targets = np.mean(source_on_targets)
            max_source_on_targets = np.max(source_on_targets)
            min_source_on_targets = np.min(source_on_targets)
            std_source_on_targets = np.std(source_on_targets)

            mean_target_on_targets = np.mean(target_on_targets)
            max_target_on_targets = np.max(target_on_targets)
            min_target_on_targets = np.min(target_on_targets)
            std_target_on_targets = np.std(target_on_targets)

            mean_weighted_source_on_targets = np.mean(weighted_source_on_targets)
            max_weighted_source_on_targets = np.max(weighted_source_on_targets)
            min_weighted_source_on_targets = np.min(weighted_source_on_targets)
            std_weighted_source_on_targets = np.std(weighted_source_on_targets)

            means_source_on_targets.append(mean_source_on_targets)
            mins_source_on_targets.append(min_source_on_targets)
            maxes_source_on_targets.append(max_source_on_targets)
            stds_source_on_targets.append(std_source_on_targets)

            means_target_on_targets.append(mean_target_on_targets)
            mins_target_on_targets.append(min_target_on_targets)
            maxes_target_on_targets.append(max_target_on_targets)
            stds_target_on_targets.append(std_target_on_targets)

            means_weighted_source_on_targets.append(mean_weighted_source_on_targets)
            mins_weighted_source_on_targets.append(min_weighted_source_on_targets)
            maxes_weighted_source_on_targets.append(max_weighted_source_on_targets)
            stds_weighted_source_on_targets.append(std_weighted_source_on_targets)
            # import pdb; pdb.set_trace()

    scaffolds = re.search(r'smaller_(.*)_(.*)_run', parent_folder).groups()
    # import pdb; pdb.set_trace()
    x = scaffolds_to_indices[scaffolds[0]]
    y = scaffolds_to_indices[scaffolds[1]]
    single_plot(axes[x][y], x_axes, means_source_on_targets, means_target_on_targets, means_weighted_source_on_targets, x, y)

for i in range(4):
    figs.delaxes(axes[i][i])

figs.tight_layout()
# figs.supxlabel('Majority scaffold in target data')
# figs.supylabel('Majority scaffold in source data')
handles, labels = axes[0][1].get_legend_handles_labels()
print(handles, labels)
figs.legend(handles, labels, loc='upper left', frameon=False)

plt.savefig('grid_results.svg')
plt.savefig('grid_results.jpg', dpi=300)

