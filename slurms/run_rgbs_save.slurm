#!/bin/bash
# ==============================================================================
#
# Copyright (c) 2018 Dell Technologies
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# description     : This script will submit a horovod job
# author          : Cedric Castagnede
# company         : DellEMC
# mail            : cedric.castagnede@dell.com
# date            : 2019-07-03
#
# ==============================================================================

####Parameters to modify####

#SBATCH --job-name='poulainl_save_rgbs'
#SBATCH --nodes=001 # laisseer ça même si node2 dans partition
#SBATCH --partition=node2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=0-12:00:00
#SBATCH --begin=now #+26hours

#your working directories -> needed to give you permissions for your working directories 
##/!\##must be at least one level above your 'root directory' 
#ex -> /home/my_group/my_user/my_working_directory about my scripts 
#ex -> /scratch/my_group/my_user/my_working_directory about my data

HOME_DIR="/home/mrmn/poulainauzeaul/stylegan4arome/gan_horovod/" 
DATA_DIR="/scratch/mrmn/poulainauzeaul/Exp_StyleGAN/"

PYTHON_SCRIPT="${HOME_DIR}save_rgbs.py"


HOROVOD_CONTAINER="stylegan4arome_louis" #container name

############################

# Collect Slurm hostlist
SLURM_HOSTLIST=$(scontrol show hostnames|paste -d, -s)
echo "SLURM_HOSTLIST: ${SLURM_HOSTLIST}"


#your user and group IDs -> needed to give you permissions for your working directories 
UID="$((`id -u`))"
GID="$((`id -g`))"
#read the arguments of your python script if you have one (>sbatch horovod.slurm "ARGS")
ARGS=$1
ARGS2="${ARGS//|/ }"

# Prepare secondary nodes (if you use at least 2 nodes)
NB_SECONDARY_WORKERS=$((SLURM_JOB_NUM_NODES-1))
if (( ${NB_SECONDARY_WORKERS} > 0 ))
then 
    echo "Start containers of secondary nodes"
    echo srun -x $(hostname -s) --nodes=${NB_SECONDARY_WORKERS} --ntasks-per-node=1 --ntasks=${NB_SECONDARY_WORKERS} slurm-docker-run secondary $HOROVOD_CONTAINER $UID     $GID $HOME_DIR $DATA_DIR $PYTHON_SCRIPT $ARGS2
    srun -x $(hostname -s) --nodes=${NB_SECONDARY_WORKERS} --ntasks-per-node=1 --ntasks=${NB_SECONDARY_WORKERS} slurm-docker-run secondary $HOROVOD_CONTAINER $UID $GID     $HOME_DIR $DATA_DIR $PYTHON_SCRIPT $ARGS &
fi

# Start the job on the  primary node
sleep 5
echo "Start Horovod on the primary node..."
echo srun -w $(hostname -s) --nodes=1 --ntasks-per-node=1 --ntasks=1 /home/mrmn/poulainauzeaul/stylegan4arome/gan_horovod/slurms/slurm-docker-run_perso primary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $DATA_DIR $PYTHON_SCRIPT $ARGS2

srun -w $(hostname -s) --nodes=1 --ntasks-per-node=1 --ntasks=1 /home/mrmn/poulainauzeaul/stylegan4arome/gan_horovod/slurm-docker-run_perso primary $HOROVOD_CONTAINER $UID $GID $HOME_DIR $DATA_DIR $PYTHON_SCRIPT $ARGS
