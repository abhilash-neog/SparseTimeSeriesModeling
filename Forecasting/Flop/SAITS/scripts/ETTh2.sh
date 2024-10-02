#!/bin/bash
device=0
input_data_path="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/ETTh2/"
pvalue="p8"

input_path="${input_data_path}${pvalue}/"

echo "Running for pvalue = ${pvalue}"
python saits_imputer.py \
    --masking_type mcar \
    --dataset ETTh2 \
    --input_data_path $input_path \
    --device $device
