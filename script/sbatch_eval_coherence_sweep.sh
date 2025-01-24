#!/bin/bash

############################################################################################################

#SBATCH --partition=iris
##SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/rheamal/bid2
#SBATCH --output slurm/AHsweep_NEW_LOW2_350ckpts.out
#SBATCH --job-name=260AHSWEEP

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
DIVERSITY=LOW2
##TRIAL 1 - 350
if [ "$DIVERSITY" = "LOW2" ]; then
    DIR="outputs/NEW_LOW2_RANDSTART_DIVERSITY_2025.01.22/07.35.11_diffusion_policy_cnn_12_square/checkpoints"
    CKPT='epoch=0350-test_mean_score=0.760.ckpt'
elif [ "$DIVERSITY" = "LOW" ]; then
    DIR="outputs/LOW_RANDSTART_DIVERSITY_2025.01.18/02.28.08_diffusion_policy_cnn_12_square/checkpoints"
    CKPT='epoch=0380-test_mean_score=0.760.ckpt'
elif [ "$DIVERSITY" = "MED2" ]; then
    DIR="outputs/NEW_MEDIUM2_RANDSTART_DIVERSITY_2025.01.22/00.27.57_diffusion_policy_cnn_12_square/checkpoints"
    CKPT='epoch=0350-test_mean_score=0.700.ckpt'
elif [ "$DIVERSITY" = "MEDIUM" ]; then
    DIR="outputs/MEDIUM_RANDSTART_DIVERSITY_2025.01.18/23.20.35_diffusion_policy_cnn_12_square/checkpoints"
    CKPT='epoch=0390-test_mean_score=0.700.ckpt'
fi

# TRIAL2 - 260
# if [ "$DIVERSITY" = "LOW2" ]; then
#     DIR="outputs/NEW_LOW2_RANDSTART_DIVERSITY_2025.01.22/07.35.11_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0260-test_mean_score=0.680.ckpt'
# elif [ "$DIVERSITY" = "MED2" ]; then
#     DIR="outputs/NEW_MEDIUM2_RANDSTART_DIVERSITY_2025.01.22/00.27.57_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0260-test_mean_score=0.660.ckpt'
# elif [ "$DIVERSITY" = "MEDIUM" ]; then
#     DIR="outputs/MEDIUM_RANDSTART_DIVERSITY_2025.01.18/23.20.35_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0260-test_mean_score=0.680.ckpt' 
# fi

# # ## TRIAL 3 - 150
# if [ "$DIVERSITY" = "LOW2" ]; then
#     DIR="outputs/NEW_LOW2_RANDSTART_DIVERSITY_2025.01.22/07.35.11_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0150-test_mean_score=0.720.ckpt'
# elif [ "$DIVERSITY" = "LOW" ]; then
#     DIR="outputs/LOW_RANDSTART_DIVERSITY_2025.01.18/02.28.08_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0150-test_mean_score=0.740.ckpt'
# elif [ "$DIVERSITY" = "MED2" ]; then
#     DIR="outputs/NEW_MEDIUM2_RANDSTART_DIVERSITY_2025.01.22/00.27.57_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0200-test_mean_score=0.740.ckpt'
# elif [ "$DIVERSITY" = "MEDIUM" ]; then
#     DIR="outputs/MEDIUM_RANDSTART_DIVERSITY_2025.01.18/23.20.35_diffusion_policy_cnn_12_square/checkpoints"
#     CKPT='epoch=0160-test_mean_score=0.740.ckpt'
# fi


GUIDANCE=0

# ===================================================================================================

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
# GUIDANCE=250

# TASK="kitchen"
# DIR="/iris/u/rheamal/bid2/ckpt/kitchen/checkpoints"
# CKPT="epoch=0030-test_mean_score=0.520.ckpt"
# GUIDANCE=0

# TASK=square_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=0150-test_mean_score=0.780.ckpt'
# GUIDANCE=0

# TASK=transport_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=0880-test_mean_score=0.720.ckpt'
# GUIDANCE=0

# TASK=tool_hang_ph
# DIR=ckpt/${TASK}
# CKPT='epoch=0850-test_mean_score=0.818.ckpt'
# GUIDANCE=0

# ===================================================================================================

# AH=1
DECAY=0.5
PERTURB=0
SEED=1
NSAMPLE=1
# NSAMPLE_VALUES="1 2 4 8"
HORIZONS="16 8 4 2 1"

for AH in $HORIZONS; do
    echo "HORIZON=${AH} UNGUIDED ${DIVERSITY} DIVERSITY DATA SEED=${SEED}"
    echo "TASK=${TASK} GUIDANCE=${GUIDANCE} NSAMPLE=$NSAMPLE SEED=$SEED"
        
    python eval_bid.py --checkpoint ${DIR}/${CKPT} \
        --output_dir ${OUTDIR}/${TASK}_horizonsweep/${DIVERSITY}/${METHOD}_nsample${NSAMPLE}/horizon${AH}/ \
        --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} \
        --decay ${DECAY} --noise ${NOISE} --ntest 100 \
        --guidance ${GUIDANCE} --perturb ${PERTURB} --seed ${SEED}
    echo "TASK=${TASK} GUIDANCE=${GUIDANCE} NSAMPLE=$NSAMPLE SEED=$SEED"
    
done
date
