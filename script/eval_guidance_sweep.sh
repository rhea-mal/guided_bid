#!/bin/bash
############################################################################################################
#SBATCH --partition=iris
##SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid_lerobot
#SBATCH --output slurm/guidance_weight_sweep.out
#SBATCH --job-name=guidance_weight_sweep

#SBATCH --time=9:30:00                                              # Max job length is 5 days
#SBATCH --cpus-per-task=20                                          # Request 20 CPUs for this task
#SBATCH --mem=80G                                                   # Request 80GB of memory

#SBATCH --nodes=1                                                   # Only use one node (machine)
#SBATCH --gres=gpu:1                                                # Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris4,iris-hgx-1
############################################################################################################
# Sweep over different guidance_weight values
WEIGHTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

# Base command with common parameters
BASE_CMD="python lerobot/scripts/eval.py \
    -p /iris/u/rheamal/bid_lerobot/lerobot/ckpt \
    eval.n_episodes=500 \
    eval.batch_size=50 \
    eval.use_async_envs=true \
    use_amp=true \
    wandb.enable=true \
    --sampler=guidance \
    --ah_test=1 \
    --temperature=0.5"

# Output directory base
OUTPUT_DIR="outputs/eval"

# Sweep loop
for GUIDANCE_WEIGHT in "${WEIGHTS[@]}"
do
    echo "Running evaluation with guidance_weight=${GUIDANCE_WEIGHT}"
    CMD="${BASE_CMD} --guidance_weight=${GUIDANCE_WEIGHT} --out-dir=${OUTPUT_DIR}/guidance_weight_${GUIDANCE_WEIGHT}"
    echo "Executing: ${CMD}"
    eval ${CMD}
done

echo "Guidance weight sweep completed!"
