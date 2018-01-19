#!/bin/bash
# Usage:
# bash scripts/test.sh GPU_ID NET DATA CONFIG
#
# Example:
# bash scripts/test.sh 0 cancer_not_annotated_vgg16_100000 cancer_not_annotated cfg1
# bash scripts/test.sh 0 cancer_not_annotated_inception_v3_98000 cancer_not_annotated cfg2

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATA=$3
CONFIG=$4

LOG="logs/cancer_diagnosis_${NET}_`date +'%Y_%m_%d_%H_%M_%S'`.txt"
exec &> >(tee -a "$LOG")
echo "Logging output to ${LOG}"

export CUDA_VISIBLE_DEVICES=${GPU_ID}
time python lib/test.py --net ${NET} --data ${DATA} --cfg ${CONFIG}
