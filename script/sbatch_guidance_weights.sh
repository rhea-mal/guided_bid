#!/bin/bash

############################################################################################################

#SBATCH --partition=iris
##SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/square_guidance_sweep2.out
#SBATCH --job-name=Gsweep

#SBATCH --time=9:30:00                                              # Max job length is 5 days
#SBATCH --cpus-per-task=20                                          # Request 8 CPUs for this task
#SBATCH --mem=80G                                                   # Request 8GB of memory

#SBATCH --nodes=1                                                   # Only use one node (machine)
#SBATCH --gres=gpu:1                                                # Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris4,iris5,iris6,iris-hgx-1

############################################################################################################

# Print the list of nodes allocated to the job
echo "Nodes allocated for this job: $SLURM_NODELIST"

date

############################################################################################################

source /sailhome/rheamal/.bashrc
conda activate bid

export PYTHONUNBUFFERED=1

############################################################################################################
# NOISE=0.0

METHOD=coherence
TH=16
OH=2
OUTDIR=result/baseline_square

# ===================================================================================================
TASK=square_mh
#LOW
# DIR=outputs/LOW_RANDSTART_DIVERSITY_2025.01.18/02.28.08_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.860.ckpt'

#HIGH
# DIR=outputs/HIGH_RANDSTART_DIVERSITY_2025.01.18/23.17.00_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.720.ckpt'

# #MEDIUM
# DIR=outputs/MEDIUM_RANDSTART_DIVERSITY_2025.01.18/23.20.35_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.740.ckpt'

#NONE
# DIR=outputs/RANDSTART_DIVERSITY_2025.01.17/18.26.54_diffusion_policy_cnn_12_square/checkpoints
# CKPT='epoch=0250-test_mean_score=0.580.ckpt'

# TASK=pusht
# DIR=ckpt/${TASK}
# CKPT='epoch=0150-test_mean_score=0.909.ckpt'
# GUIDANCE=0

# TASK=block_pushing
# DIR=ckpt/${TASK}
# # CKPT='epoch=7550-test_mean_score=1.000.ckpt'
# CKPT='epoch=2250-test_mean_score=0.287.ckpt'
# GUIDANCE=25

# TASK="lift"
# DIR="/iris/u/rheamal/bid2/ckpt/lift"
# CKPT='epoch=0150-test_mean_score=0.960.ckpt'
# GUIDANCE=0

# TASK="kitchen"
# DIR="/iris/u/rheamal/bid2/ckpt/kitchen/checkpoints"
# CKPT="epoch=0030-test_mean_score=0.520.ckpt"
# GUIDANCE=0

TASK=square_mh
DIR=ckpt/${TASK}
CKPT='epoch=0150-test_mean_score=0.780.ckpt'

# TASK=transport_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=0880-test_mean_score=0.720.ckpt'

# TASK=tool_hang_ph
# DIR=ckpt/${TASK}
# CKPT='epoch=0850-test_mean_score=0.818.ckpt'

# ===================================================================================================

AH=1
DECAY=0.5
SEED=0
NSAMPLE=1
NOISE=0

GUIDANCE_VALUES="250 300 350"
for GUIDANCE in $GUIDANCE_VALUES; do
    echo "TASK=${TASK} GUIDANCE=${GUIDANCE} NSAMPLE=$NSAMPLE SEED=$SEED"
    python eval_bid.py --checkpoint ${DIR}/${CKPT} \
        --output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH} \
        --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} \
        --decay ${DECAY} --noise ${NOISE} --ntest 100 \
        --guidance ${GUIDANCE} --seed ${SEED}
    echo "TASK=${TASK} GUIDANCE=${GUIDANCE} NSAMPLE=$NSAMPLE SEED=$SEED"
    
done
date
