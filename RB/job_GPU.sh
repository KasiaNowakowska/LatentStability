#!/bin/bash
#SBATCH --job-name=CAE   # Job name
#SBATCH --time=01:00:00               # Request runtime (hh:mm:ss)
#SBATCH --partition=gpu     # Request the GPU partition
#SBATCH --gres=gpu:1        # Request a single GPU
#SBATCH --mem-per-cpu=6G           # Request 8GB memory per CPU core
#SBATCH --output=slurm-%x-%j.out  # Output file: slurm-JobName-JobID.out
#SBATCH --error=slurm-%x-%j.out   # Redirects stderr to stdout file

# Load any necessary modules
module load miniforge
source activate base
source activate CAE2

# Execute your application
pwd

python3 train_cae.py