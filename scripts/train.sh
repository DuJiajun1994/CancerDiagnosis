#!/bin/bash
# Usage:
# bash scripts/train.sh GPU_ID NET DATA CONFIG
#
# Example:
# bash scripts/train.sh 0 inception_v3 cancer_not_annotated cfg2
# bash scripts/train.sh 0 vgg16 cancer_not_annotated cfg1

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATA=$3
CONFIG=$4

LOG="logs/cancer_diagnosis_${NET}_${DATA}_`date +'%Y_%m_%d_%H_%M_%S'`.txt"
exec &> >(tee -a "$LOG")
echo "Logging output to ${LOG}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
nohup time python lib/train.py --net ${NET} --data ${DATA} --cfg ${CONFIG} &
