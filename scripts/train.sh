#!/usr/bin/env bash
# Usage:
# ./scripts/train.sh GPU_ID NET DATASET
#
# Example:
# ./scripts/train.sh

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
DATA=$3
CONFIG=$4


LOG="../logs/cancer_diagnosis_${NET}_${DATA}_`date +'%Y_%m_%d_%H_%M_%S'`.txt"
exec &> >(tee -a "$LOG")
echo "Logging output to ${LOG}"

time python ../lib/train.py --gpu GPU_ID --net NET --data DATA --cfg CONFIG