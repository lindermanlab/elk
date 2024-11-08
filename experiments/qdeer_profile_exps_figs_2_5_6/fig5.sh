#!/usr/bin/bash
#SBATCH --job-name=fig5
#SBATCH --error=fig5%j_%a.err
#SBATCH --out=fig5%j_%a.out
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=00:15:59
#SBATCH --constraint='GPU_SKU:V100_SXM2&GPU_MEM:16GB'
#SBATCH --mail-type=ALL

echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"

$SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 fig5.py $@

"Job ended at: $(date)"