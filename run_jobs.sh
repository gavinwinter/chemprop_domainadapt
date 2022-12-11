#!/bin/bash
#SBATCH -J domain_adapt
#SBATCH -o domain_adapt-%j.out
#SBATCH -t 24-00:00:00
#SBATCH -n 1
#SBATCH -N 1
#SBATCH --mem=300gb
#SBATCH --array=0-9

echo "Date              = $(date)"
echo "Hostname          = $(hostname -s)"
echo "Working Directory = $(pwd)"
echo ""
cat $0
echo ""

source /etc/profile
module load anaconda/2021a
source activate chemprop

python train_domain_adapt.py --folder_index $SLURM_ARRAY_TASK_ID