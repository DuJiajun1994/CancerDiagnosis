#!/bin/bash
# Usage:
# bash scripts/test.sh GPU_ID NET DATA CONFIG VAR
#
# Example:
# bash scripts/test.sh 0 vgg16 40 cfg1 VAR

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATA=$3
CONFIG=$4
VAR=$5

LOG="logs/cancer_diagnosis_${VAR}_`date +'%Y_%m_%d_%H_%M_%S'`.txt"
exec &> >(tee -a "$LOG")
echo "Logging output to ${LOG}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
time python lib/test.py --net ${NET} --data ${DATA} --cfg ${CONFIG} --var ${VAR}
