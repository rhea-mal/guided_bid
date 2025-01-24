############################################################################################################

TASK=pusht
TASK=block_pushing
# TASK=tool_hang_ph
# TASK=transport_mh
TASK=kitchen

INPUT=low_dim

MODEL=diffusion_policy_cnn
# MODEL=behavior_transformer

############################################################################################################

SRCDIR=data/experiments/${INPUT}/${TASK}/${MODEL} 
SRCFILE=config.yaml

TARDIR=commands
TARFILE=${INPUT}_${TASK}_${MODEL}_config.yaml

cp ${SRCDIR}/${SRCFILE} ${TARDIR}/${TARFILE}
