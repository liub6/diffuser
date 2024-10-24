#!/bin/bash -l
#SBATCH --job-name=example   # Job name
##SBATCH --nodes=1               # Total number of nodes 
##SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 16 total (2x8)
##SBATCH --gpus-per-node=1       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=4
#SBATCH --time=6:00:00       # Run time (d-hh:mm:ss)
#SBATCH --mem=30000M
#SBATCH --gres=gpu:1


srun python ./corl/algorithms/td3_bc.py \
        --config $1 \
        --checkpoints_path $2 \
        --name $3 \
        --diffusion.path $4 \
        --pole_length $5 \
        --dataset $6 \
        --context_aware $7 \
        --seed $8 \
        --segment $9

# srun python ./corl/algorithms/edac.py \
#         --config ./corl/yaml/edac/cartpole/cartpole_swingup.yaml \
#         --checkpoints_path ./corl_logs/ \
#         --name "diffuser-front-10.0-1" \
#         --diffusion.path "./results-front/5m_samples.npz_0.5.npz" \
#         --pole_length 0.7 \
#         --dataset "dm-cartpole-train-length-front-v0" \
#         --context_aware 1 \
#         --seed 1