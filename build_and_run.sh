#!/bin/sh

set -eu

CWD=stylegan4arome_2.0
SCRIPTDIR=/home/mrmn/brochetc/aWeatherCodebook/
DATA_DIR=/scratch/mrmn/brochetc/GAN_2D/datasets_full_indexing/IS_1_1.0_0_0_0_0_0_256_large_lt_done/
OUTPUT_DIR=/scratch/mrmn/brochetc/GAN_2D/Exp_Codebook/

DOCKFILE=$2

mkdir -p $DATA_DIR
#chmod -R g+w $SCRATCHDIR
mkdir -p $SCRIPTDIR
#chmod -R g+w $SCRIPTDIR
mkdir -p $OUTPUT_DIR
echo $DATA_DIR

build() {
docker build -f $DOCKFILE . --tag "$CWD" --build-arg dev_id=$(id -u) --build-arg labia_gid=$(id -g)
}



#"device=$SLURM_STEP_GPUS"
run() {
    shift
    docker run --ipc=host --rm --gpus device=$SLURM_STEP_GPUS \
        -v $SCRIPTDIR:$SCRIPTDIR \
	-v $OUTPUT_DIR:$OUTPUT_DIR \
	-v $DATA_DIR:$DATA_DIR \
        -it "$CWD"

}


case ${1:-build} in
	build) build ;;
	clean) clean ;;
	run) run "$@" ;;
	dev) dev "$@" ;;
	*) echo "$0: No command named '$1'";;
esac
