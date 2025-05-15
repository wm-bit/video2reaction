#!/bin/bash
#SBATCH -G 1                 # Number of GPUs
#SBATCH --nodes=1             # Always set to 1!
#SBATCH --gres=gpu:1          # This needs to match num GPUs.
#SBATCH --ntasks-per-node=1   # This needs to match num GPUs. default 8
#SBATCH --mem=150000           # Requested Memory
#SBATCH -p gypsum-2080ti     # Partition
#SBATCH -t 01-00              # Job time limit
#SBATCH -o ./log/slurm-cten-vaaneterase-%j-%a.out       # %j = job ID
#SBATCH --array=0-4

cd /work/v2r  # your own directory

eval "$(conda shell.bash hook)"
conda activate torch  # your conda environment

python scripts/train_cten.py --config vaanet.yaml --seed_idx ${SLURM_ARRAY_TASK_ID} --audio_mode average --is_erasing 0 --loss CECADLoss