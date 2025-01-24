#!/bin/bash

##SBATCH --partition=iris
#SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/train_NEW_LOW2_RANDSTART-%j.out                                      
#SBATCH --job-name=train_diverse

#SBATCH --time=30:00:00                                              # Max job length is 5 days
#SBATCH --cpus-per-task=20                                          # Request 8 CPUs for this task
#SBATCH --mem-per-cpu=4G                                            # Request 8GB of memory

#SBATCH --nodes=1                                                   # Only use one node (machine)
#SBATCH --gres=gpu:1                                                # Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris9,iris-hp-z8,iris-hgx-1
############################################################################################################

# Input type
INPUT="low_dim"  
MODEL="diffusion_policy_cnn"  
TASK="square"
# Planning horizon

HORIZON=12  #
SEED=40
FILE="low_dim_square_diverse_diffusion_policy_cnn_config.yaml"

export HYDRA_FULL_ERROR=1

# Execute the training script
python train.py --config-dir=config --config-name="${FILE}" \
    training.seed=${SEED} \
    training.device=cuda:0 \
    hydra.run.dir="outputs/NEW_LOW2_RANDSTART_DIVERSITY_\${now:%Y.%m.%d}/\${now:%H.%M.%S}_${MODEL}_${HORIZON}_${TASK}"
