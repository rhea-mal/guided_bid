#!/bin/bash

NOISE=0.0

METHOD=coherence
TH=16
OH=2
OUTDIR=result/pusht

# ===================================================================================================

TASK=pusht
DIR=ckpt/${TASK}
CKPT='epoch=0150-test_mean_score=0.909.ckpt'
# CKPT='epoch=0100-test_mean_score=0.718.ckpt'
# CKPT='epoch=3250-test_mean_score=0.973.ckpt'

# TASK=kitchen
# DIR=ckpt/${TASK}/checkpoints
# # CKPT='epoch=1700-test_mean_score=0.580.ckpt'
# CKPT='epoch=0050-test_mean_score=0.511.ckpt'

# TASK=lift_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=0250-test_mean_score=1.000.ckpt'

# TASK=can_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=3200-test_mean_score=1.000.ckpt'

# TASK=square_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=0150-test_mean_score=0.780.ckpt'
# GUIDANCE=100

# TASK=transport_mh
# DIR=ckpt/${TASK}
# CKPT='epoch=3800-test_mean_score=0.773.ckpt'

# TASK=tool_hang_ph
# DIR=ckpt/${TASK}
# CKPT='epoch=0850-test_mean_score=0.818.ckpt'

# ===================================================================================================

AH=1
NSAMPLE=1
DECAY=0.5
PERTURB=0.5
DISTDIR="dist_test.txt"

echo "task: $TASK"

python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir ${OUTDIR}/${TASK}/${METHOD}_${NSAMPLE}_${DECAY}/${NOISE}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} --perturb ${PERTURB} --decay ${DECAY} --ntest 100 --seed 0 --dist_dir ${DISTDIR}
