#!/bin/bash

############################################################################################################

INPUT=low_dim
MODEL=diffusion_policy_cnn

TASK=pusht
# TASK=block_pushing
# TASK=tool_hang_ph
# TASK=transport_mh
# TASK=kitchen

HORIZON=12
############################################################################################################

# FILE=${INPUT}_${TASK}_${MODEL}_config.yaml
FILE=${INPUT}_${TASK}_h${HORIZON}_${MODEL}_config.yaml

HYDRA_FULL_ERROR=1 python train.py --config-dir=commands --config-name=${FILE} \
    training.seed=42 training.device=cuda:0 hydra.run.dir='outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${horizon}_${task_name}'