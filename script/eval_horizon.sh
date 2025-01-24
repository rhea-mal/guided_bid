#!/bin/bash

############################################################################################################

# PushT
DIR=outputs/2024.05.02/17.42.24_train_diffusion_unet_lowdim_pusht_lowdim/checkpoints
# CKPT='epoch=0050-test_mean_score=0.250.ckpt'
# CKPT='epoch=0100-test_mean_score=0.718.ckpt'
CKPT='epoch=0450-test_mean_score=0.851.ckpt'
# REF='epoch=0050-test_mean_score=0.250.ckpt'

# DIR=outputs/2024.05.13/10.19.46_train_diffusion_unet_lowdim_4_pusht_lowdim/checkpoints/
# CKPT='epoch=0450-test_mean_score=0.836.ckpt'

# DIR=outputs/2024.05.13/10.19.46_train_diffusion_unet_lowdim_8_pusht_lowdim/checkpoints/
# CKPT='epoch=0450-test_mean_score=0.897.ckpt'

# DIR=outputs/2024.05.13/10.19.46_train_diffusion_unet_lowdim_12_pusht_lowdim/checkpoints/
# CKPT='epoch=0450-test_mean_score=0.839.ckpt'

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

# DIR=outputs/2024.05.13/10.19.46_train_diffusion_unet_lowdim_4_pusht_lowdim/checkpoints/
# CKPT='epoch=0450-test_mean_score=0.836.ckpt'

# DIR=outputs/2024.05.13/10.19.46_train_diffusion_unet_lowdim_8_pusht_lowdim/checkpoints/
# CKPT='epoch=0450-test_mean_score=0.897.ckpt'

# DIR=outputs/2024.05.13/10.19.46_train_diffusion_unet_lowdim_12_pusht_lowdim/checkpoints/
# CKPT='epoch=0450-test_mean_score=0.839.ckpt'

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'random' --ahorizon 2
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'random' --ahorizon 16
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'coherence' --nsample 10 --ahorizon 10
# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'ema' --ahorizon 4 --noise 5.0 --decay 0.5

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'cma' --ahorizon 4 --noise 5.0 --decay 0.5

# DIR=outputs/2024.05.13/10.23.30_train_diffusion_unet_lowdim_12_kitchen_lowdim/checkpoints
# DIR=outputs/2024.05.13/10.23.30_train_diffusion_unet_lowdim_8_kitchen_lowdim/checkpoints
# DIR=outputs/2024.05.13/10.23.27_train_diffusion_unet_lowdim_4_kitchen_lowdim/checkpoints
# CKPT='epoch=0100-test_mean_score=0.551.ckpt'


# TASK=tool_hang_ph
# DIR=data/experiments/low_dim/${TASK}/diffusion_policy_cnn/train_0/checkpoints
# CKPT=latest.ckpt

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'random' --ahorizon 1

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'fixed' --ahorizon 1

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'warmstart' --ahorizon 1

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'wma' --ahorizon 1

# python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'lowvar' --nsample 5 --decay 0.9 --ahorizon 1


python eval_bs.py --checkpoint ${DIR}/${CKPT} --output_dir data/eval/temp --sampler 'warmcoherence' --nsample 5 --decay 0.9 --ahorizon 1
