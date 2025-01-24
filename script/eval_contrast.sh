#!/bin/bash

TASK=pusht
METHOD=contrast

TH=16
OH=2

DIR=ckpt/pusht/train_diffusion_unet_lowdim_th${TH}_oh${OH}_pusht_lowdim
# CKPT='epoch=3250-test_mean_score=0.973.ckpt'
CKPT='epoch=0100-test_mean_score=0.718.ckpt'

REF='epoch=0050-test_mean_score=0.250.ckpt'

############################################################################################################

# AH=1  				# close-loop action chunking
AH=8					# open-loop action chunking

NSAMPLE=15

python eval_bid.py --checkpoint ${DIR}/${CKPT} --output_dir result/${TASK}/${METHOD}/th${TH}_oh${OH}_ah${AH} --sampler ${METHOD} -ah ${AH} --nsample ${NSAMPLE} --reference ${DIR}/${REF}
