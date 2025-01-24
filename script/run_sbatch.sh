#!/bin/bash

# sbatch --array=1 commands/sbatch_sampling_pusht.sh
# sbatch --array=2 commands/sbatch_sampling_pusht.sh
# sbatch --array=1,2,3,4,6,8 commands/sbatch_sampling_pusht.sh
# sbatch --array=10-14:2 commands/sbatch_sampling_pusht.sh
# sbatch --array=1,2,3,4,6,8,10,12,14 commands/sbatch_sampling_pusht.sh


# sbatch --array=5,10,20 commands/sbatch_sampling_pusht.sh
# sbatch --array=0,30,40,50 commands/sbatch_sampling_pusht.sh

# sbatch --array=1 commands/sbatch_sampling_pusht.sh
# sbatch --array=10,20,40 commands/sbatch_sampling_pusht.sh
# sbatch --array=20,40 commands/sbatch_sampling_pusht.sh
# sbatch --array=5,10,20 commands/sbatch_sampling_pusht.sh

# sbatch --array=1 commands/sbatch_sampling_block.sh
# sbatch --array=1,2,3,4,6,8 commands/sbatch_sampling_block.sh
# sbatch --array=10-16:2 commands/sbatch_sampling_block.sh
# sbatch --array=1,2,3,4,6,8,10,12,14 commands/sbatch_sampling_block.sh


# sbatch --array=2 commands/sbatch_sampling_kitchen.sh
# sbatch --array=1,2,3,4,6,8 commands/sbatch_sampling_kitchen.sh
# sbatch --array=10-16:2 commands/sbatch_sampling_kitchen.sh
# sbatch --array=1,2,3,4,6,8,10,12,14 commands/sbatch_sampling_kitchen.sh

# sbatch --array=0 commands/sbatch_sampling_kitchen.sh

# training
# sbatch --array=2,6,8,10,14 commands/sbatch_train_horizon.sh
# sbatch --array=2 commands/sbatch_train_horizon.sh
# sbatch --array=6 commands/sbatch_train_horizon.sh
sbatch --array=10 commands/sbatch_train_horizon.sh
# sbatch --array=14 commands/sbatch_train_horizon.sh
# sbatch --array=6,10 commands/sbatch_train_horizon.sh
# sbatch --array=2,6,10,14 commands/sbatch_train_horizon.sh


# contrast
# sbatch --array=5,10,15,20,25,30 commands/sbatch_contrast_pusht.sh
# sbatch --array=5,10,15,20,25,30 commands/sbatch_contrast_block.sh
