#!/usr/bin/bash
#SBATCH --job-name=mem_fig2
#SBATCH --error=mem_fig2%j_%a.err
#SBATCH --out=mem_fig2%j_%a.out
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=47:59:59
#SBATCH --constraint='GPU_SKU:V100_SXM2&GPU_MEM:16GB'
#SBATCH --mail-type=ALL

for i in {1..144}
do
    $SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 mem_exp.py $@
done