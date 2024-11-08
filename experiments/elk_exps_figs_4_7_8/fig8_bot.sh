#!/usr/bin/bash
#SBATCH --job-name=fig8Bot
#SBATCH --error=fig8Bot_%j_%a.err
#SBATCH --out=fig8Bot_%j_%a.out
#SBATCH --time=0:59:59
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mail-type=ALL
#SBATCH -C GPU_SKU:A100_SXM4

$SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 fig8_bot.py $@