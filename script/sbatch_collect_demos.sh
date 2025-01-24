#!/bin/bash

############################################################################################################

#SBATCH --partition=iris
##SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/collect_NEW_LOW2_RANDSTART_demos-%j.out
#SBATCH --job-name=low2demos

#SBATCH --time=9:30:00                                              # Max job length is 5 days
#SBATCH --cpus-per-task=20                                          # Request 8 CPUs for this task
#SBATCH --mem=80G                                                   # Request 8GB of memory

#SBATCH --nodes=1                                                   # Only use one node (machine)
#SBATCH --gres=gpu:1                                                # Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris4,iris5,iris-hgx-1

############################################################################################################

python collect_demos_og.py