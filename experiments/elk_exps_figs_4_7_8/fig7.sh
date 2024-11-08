#!/usr/bin/bash
#SBATCH --job-name=fig7
#SBATCH --error=fig7_%j_%a.err
#SBATCH --out=fig7_%j_%a.out
#SBATCH --time=47:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL

$SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 fig7.py $@
