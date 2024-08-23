#!/bin/bash
masking_type=$1
dataset=$2
device=$3
input_data_path="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/${dataset}/"
fractions="p8"
periodic_fractions="a1 a2 a3 a4 a5 a6 a7 a8 a9"
patch_fractions="patch1 patch2 patch3 patch4 patch5 patch6 patch7 patch8 patch9"
if [[ "$masking_type" == "mcar" ]]; then
    selected_fractions=$fractions
elif [[ "$masking_type" == "periodic" ]]; then
    selected_fractions=$periodic_fractions
elif [[ "$masking_type" == "patch" ]]; then
    selected_fractions=$patch_fractions
else
    echo "Invalid masking_type. Please provide either 'mcar', 'periodic', or 'patch'."
    exit 1
fi

for pvalue in $selected_fractions; do
    input_path="${input_data_path}${pvalue}/"
    echo "Running for pvalue = ${pvalue}"
    python saits_imputer.py \
        --masking_type $masking_type \
        --dataset $dataset \
        --input_data_path $input_path \
        --device $device
done