#!/bin/bash
#SBATCH --account=ml4science
#SBATCH --partition=dgx_normal_q #dgx_normal_q #a100_normal_q
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --time=8:00:00 # 24 hours
#SBATCH --gres=gpu:1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/codefiles22/.conda/envs/env/lib

DATASET=$1
PVALUES=$2
MASKING=$3
SOURCE_FILE=$4

python -u data_generator_ARC.py \
    --dataset $DATASET \
    --pvalues $PVALUES \
    --masking_type $MASKING \
    --source_file $SOURCE_FILE