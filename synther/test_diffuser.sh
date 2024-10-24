#!/bin/bash -l
#SBATCH --job-name=example   # Job name
#SBATCH --output=examplejob.o%j # Name of stdout output file
#SBATCH --error=examplejob.e%j  # Name of stderr error file
##SBATCH --nodes=1               # Total number of nodes 
##SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
##SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=30000M
#SBATCH --gres=gpu:1
#SBATCH --gpus=1

# echo "file $1"
# echo "config $2"
# echo "checkpoints_path $3"
# echo "name $4"
# echo "diffusion.path $5"
# echo "pole_length $6"
# echo "dataset $7"
# echo "context_aware $8"
# echo "seed $9"
# echo "segment ${10}"

srun python ./corl/algorithms/$1.py \
        --config $2 \
        --checkpoints_path $3 \
        --name $4 \
        --diffusion.path $5 \
        --pole_length $6 \
        --dataset $7 \
        --context_aware $8 \
        --seed $9 \
        --segment ${10}

# srun python ./corl/algorithms/edac.py \
#         --config ./corl/yaml/edac/cartpole/cartpole_swingup.yaml \
#         --checkpoints_path ./corl_logs/ \
#         --name "diffuser-front-10.0-1" \
#         --diffusion.path "./results-front/5m_samples.npz_0.5.npz" \
#         --pole_length 0.7 \
#         --dataset "dm-cartpole-train-length-front-v0" \
#         --context_aware 1 \
#         --seed 1