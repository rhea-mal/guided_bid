#!/bin/bash

############################################################################################################

# PushT
DIR=outputs/2024.05.02/17.42.24_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints
# CKPT='epoch=0050-test_mean_score=0.250.ckpt'
CKPT='epoch=0100-test_mean_score=0.718.ckpt'
# CKPT='epoch=0450-test_mean_score=0.851.ckpt'
# REF='epoch=0050-test_mean_score=0.250.ckpt'

# Block
# DIR=outputs/2024.05.02/17.42.20_train_diffusion_unet_lowdim_blockpush_lowdim_seed_abs/checkpoints
# CKPT='epoch=0050-test_mean_score=0.188.ckpt'
# CKPT='epoch=0400-test_mean_score=0.238.ckpt'
# CKPT='epoch=0450-test_mean_score=0.277.ckpt'
# REF='epoch=0050-test_mean_score=0.188.ckpt'

# Kitchen
# DIR=outputs/2024.05.02/19.27.29_train_diffusion_unet_lowdim_kitchen_lowdim/checkpoints
# CKPT='epoch=0050-test_mean_score=0.549.ckpt'
# CKPT='epoch=0100-test_mean_score=0.557.ckpt'
# CKPT='epoch=0450-test_mean_score=0.569.ckpt'


# echo ${CKPT}

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'random'
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'consensus' --nsample 10 --topk 3
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'contrastive' --nsample 10 --reference ${DIR}/${REF}

############################################################################################################

############################################################################################################

# DIR=data/experiments/low_dim/block_pushing/diffusion_policy_cnn/train_0/checkpoints
# DIR=data/experiments/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints
# CKPT=latest.ckpt

# DIR=data/experiments/low_dim/kitchen/diffusion_policy_cnn/train_0/checkpoints
# CKPT=latest.ckpt

# DIR=outputs/2024.05.02/19.27.29_train_diffusion_unet_lowdim_kitchen_lowdim/checkpoints
# CKPT='epoch=0100-test_mean_score=0.557.ckpt'

############################################################################################################

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'random'
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'consensus' --nsample 10 --topk 3
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'contrastive' --nsample 10


REFDIR=outputs/2024.05.02/17.42.24_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints
REF='epoch=0050-test_mean_score=0.250.ckpt'

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'random'

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'contrastive' --nsample 5 --reference ${REFDIR}/${REF}
python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'contrastive' --nsample 10 --reference ${REFDIR}/${REF}
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'contrastive' --nsample 20 --reference ${REFDIR}/${REF}
