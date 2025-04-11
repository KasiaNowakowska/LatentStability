#!/bin/bash
#SBATCH --job-name=CAE   # Job name
#SBATCH --time=00:10:00               # Request runtime (hh:mm:ss)
#SBATCH --mem=5G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=2
#SBATCH --output=slurm-%x-%A-%a.out  # Output: slurm-JobName-ArrayJobID_TaskID.out
#SBATCH --error=slurm-%x-%A-%a.out   # Redirects stderr to stdout file

# Load any necessary modules
module load miniforge
source activate base
source activate CAE2

# Execute your application
pwd

python3 train_cae.py