#!/bin/bash

############################################################################################################

##SBATCH --partition=iris
#SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/eval-%j.out 										
#SBATCH --job-name=eval

#SBATCH --time=3:00:00 												# Max job length is 5 days
#SBATCH --cpus-per-task=20 											# Request 8 CPUs for this task
#SBATCH --mem-per-cpu=4G 											# Request 8GB of memory

#SBATCH --nodes=1 													# Only use one node (machine)
#SBATCH --gres=gpu:1 												# Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris3,iris4,iris-hp-z8,iris-hgx-1

date

############################################################################################################

source /sailhome/rheamal/.bashrc
conda activate bid

export PYTHONUNBUFFERED=1

############################################################################################################

# bash script/eval_random.sh

# bash script/eval_coherence.sh

# bash script/eval_contrast.sh

bash script/eval_bid.py
date