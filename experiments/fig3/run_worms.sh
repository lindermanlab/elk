#!/usr/bin/bash
#SBATCH --job-name=worms
#SBATCH --error=worms%j_%a.err
#SBATCH --out=worms%j_%a.out
#SBATCH -G 1
#SBATCH --cpus-per-gpu=4
#SBATCH --time=23:59:59
#SBATCH --constraint='GPU_SKU:V100_SXM2&GPU_MEM:16GB'
#SBATCH --mail-type=ALL


NEPOCHS=3000
PATIENCE=1000
QUASI_FLAG=""
SCAN_FLAG=""

echo "Running on partition: $SLURM_JOB_PARTITION"
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node list: $SLURM_NODELIST"

# Parse command line arguments for --quasi flag
for arg in "$@"
do
    case $arg in
        --quasi)
        QUASI_FLAG="--quasi"
        shift # Remove --quasi from processing
        ;;
    esac

    case $arg in
        --use_scan)
        SCAN_FLAG="--use_scan"
        shift 
        ;;
    esac
done

$SCRATCH/deer-replication-attempt-10-27-2024/venvs/elkv1/bin/python3 eigenworms.py --nchannel 1 --precision 32 --batch_size 4 --version 0 --seed 0 --lr 3e-5 --nepochs $NEPOCHS --ninps 6 --nstates 32 --nsequence 17984 --nclass 5 --nlayer 5 --dset eigenworms --patience $PATIENCE --patience_metric accuracy $QUASI_FLAG $SCAN_FLAG

"Job ended at: $(date)"

