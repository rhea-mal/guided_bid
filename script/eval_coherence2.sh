#!/bin/bash

############################################################################################################

#SBATCH --partition=iris
##SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/LOW_RAND_guided_nsamples_s0.out
#SBATCH --job-name=GUIDEDLOW

#SBATCH --time=9:30:00                                              # Max job length is 5 days
#SBATCH --cpus-per-task=20                                          # Request 8 CPUs for this task
#SBATCH --mem=80G                                                   # Request 8GB of memory

#SBATCH --nodes=1                                                   # Only use one node (machine)
#SBATCH --gres=gpu:1                                                # Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris9,iris-hgx-1

############################################################################################################

# Print the list of nodes allocated to the job
echo "Nodes allocated for this job: $SLURM_NODELIST"

date

############################################################################################################

source /sailhome/rheamal/.bashrc
conda activate bid

export PYTHONUNBUFFERED=1

############################################################################################################
NOISE=0.0

METHOD=coherence
TH=16
OH=2
OUTDIR=result
TASK=square_mh
#HIGH
# DIR=outputs/HIGH_RANDSTART_DIVERSITY_2025.01.18/23.17.00_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.720.ckpt'

#MEDIUM
# DIR=outputs/MEDIUM_RANDSTART_DIVERSITY_2025.01.18/23.20.35_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.740.ckpt'

#LOW
DIR=outputs/LOW_RANDSTART_DIVERSITY_2025.01.18/02.28.08_diffusion_policy_cnn_12_square/checkpoints
CKPT='epoch=0250-test_mean_score=0.860.ckpt'

#NONE
# DIR=outputs/RANDSTART_DIVERSITY_2025.01.17/18.26.54_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.580.ckpt'

GUIDANCE=0

# ===================================================================================================

AH=1
DECAY=0.5
PERTURB=0
SEED=1

NSAMPLE_VALUES="1"

for NSAMPLE in $NSAMPLE_VALUES; do
    echo "METHOD=${METHOD} GUIDED LOW DIVERSITY DATA SEED=${SEED}"
    echo "TASK=${TASK} GUIDANCE=${GUIDANCE} NSAMPLE=$NSAMPLE SEED=${SEED}"
        
    python eval_bid.py --checkpoint ${DIR}/${CKPT} \
        --output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH}_SEED${SEED} \
        --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} \
        --decay ${DECAY} --noise ${NOISE} --ntest 100 \
        --guidance ${GUIDANCE} --perturb ${PERTURB} --seed ${SEED}
    echo "TASK=${TASK} GUIDANCE=${GUIDANCE} NSAMPLE=$NSAMPLE SEED=$SEED"
    
done
date
