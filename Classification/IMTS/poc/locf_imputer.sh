dataset=$1
device=$2

masking_type="mcar"
input_data_path="/raid/abhilash/synthetic_datasets/${dataset}/"
output_data_path="/raid/sepideh/Project_MissTSM/ts_synthetic_datasets/${dataset}/"
fractions="p6 p7 p8 p9"
periodic_fractions="a6 a7 a8 a9"
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
    out_path="${output_data_path}${pvalue}/"
    echo "Running for pvalue = ${pvalue}"
    python locf_imputation.py \
        --masking_type $masking_type \
        --dataset $dataset \
        --input_data_path $input_path \
        --output_data_path $out_path \
        --device $device
done