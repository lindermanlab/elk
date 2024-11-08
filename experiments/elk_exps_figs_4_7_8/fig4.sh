#!/usr/bin/bash
#SBATCH --job-name=fig4
#SBATCH --error=fig4_%j_%a.err
#SBATCH --out=fig4_%j_%a.out
#SBATCH --time=23:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL

$SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 fig4.py $@
