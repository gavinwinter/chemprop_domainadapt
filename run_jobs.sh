#!/bin/bash
#SBATCH -J domain_adapt
#SBATCH -o domain_adapt-%j.out
#SBATCH -t 24-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --gres=gpu:volta:1
#SBATCH --mem=100gb
#SBATCH --array=0-59

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
cat $0
echo ""

source /etc/profile
module load anaconda/2021a
source activate chemprop

python train_domain_adapt.py --folder_index $SLURM_ARRAY_TASK_ID --parent_folder_name 'remaining_smaller_runs/smaller_OeC(NCc1ccccc1)c1ccccc1_OeC(Nc1ccccc1)c1ccccc1_run'