#!/bin/bash -l
#SBATCH --job-name=example   # Job name
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
#SBATCH --nodes=1               # Total number of nodes 
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
#SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --time=03:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=15000M
#SBATCH --gres=gpu:1
#SBATCH --gpus=1


srun python sac.py --length $1
