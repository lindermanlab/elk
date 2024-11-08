#!/usr/bin/bash
#SBATCH --job-name=time_fig2
#SBATCH --error=time_fig2%j_%a.err
#SBATCH --out=time_fig2%j_%a.out
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=47:59:59
#SBATCH --constraint='GPU_SKU:V100_SXM2&GPU_MEM:16GB'
#SBATCH --mail-type=ALL

for i in {1..240}
do
    $SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 timing_exp.py $@
done