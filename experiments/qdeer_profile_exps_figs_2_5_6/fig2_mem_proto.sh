#!/usr/bin/bash
#SBATCH --job-name=proto_mem_fig2
#SBATCH --error=proto_mem_fig2%j_%a.err
#SBATCH --out=proto_mem_fig2%j_%a.out
#SBATCH -G 1
#SBATCH --time=00:29:59
#SBATCH --constraint='GPU_SKU:V100_SXM2&GPU_MEM:16GB'
#SBATCH --mail-type=ALL

for i in {1..144}
do
    $SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 mem_exp.py $@
done