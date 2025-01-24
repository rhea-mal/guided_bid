#!/bin/bash

############################################################################################################

##SBATCH --partition=iris
#SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/train-dp-%j.out 										
#SBATCH --job-name=train

#SBATCH --time=36:00:00 											# Max job length is 5 days
#SBATCH --cpus-per-task=20 											# Request 8 CPUs for this task
#SBATCH --mem-per-cpu=4G 											# Request 8GB of memory

#SBATCH --nodes=1 													# Only use one node (machine)
#SBATCH --gres=gpu:1 												# Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris3,iris4,iris6,iris-hp-z8,iris-hgx-1

date

############################################################################################################

source /sailhome/rheamal/.bashrc
conda activate bid

export PYTHONUNBUFFERED=1

############################################################################################################

INPUT=low_dim
MODEL=diffusion_policy_cnn

TASK=pusht
# TASK=block_pushing
# TASK=tool_hang_ph
# TASK=transport_mh
# TASK=kitchen

AH=10
OH=10

############################################################################################################

CFG=${INPUT}_${TASK}_ah${AH}_oh${OH}_${MODEL}_config.yaml

############################################################################################################

HYDRA_FULL_ERROR=1 python train.py --config-dir=config --config-name=${CFG} \
	training.seed=42 training.device=cuda:0 hydra.run.dir='outputs/${now:%Y.%m.%d}/${now:%H.%M}_${name}_th${horizon}_oh${n_obs_steps}_${task_name}'

date