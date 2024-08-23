masking_type=$1
dataset=$2
input_data_path="/projects/ml4science/time_series/ts_synthetic_datasets/synthetic_datasets/${dataset}/"
fractions="p1 p2 p3 p4 p5 p6 p7 p8 p9 p9"
periodic_fractions="a1 a2 a3 a4 a5 a6 a7 a8 a9"
patch_fractions="patch1 patch2 patch3 patch4 patch5 patch6 patch7 patch8 patch9"

if [[ "$masking_type" == "mcar" ]]; then
    fractions=$fractions
elif [[ "$masking_type" == "periodic" ]]; then
    fractions=$periodic_fractions
elif [[ "$masking_type" == "patch" ]]; then
    fractions=$patch_fractions
else
    echo "Invalid masking_type. Please provide either 'mcar', 'periodic', or 'custom'."
    exit 1
fi

for pvalue in $fractions; do
    input_path="${input_data_path}${pvalue}/"
    echo "Running for pvalue = ${pvalue}"
    python knn_impute.py \
        --masking_type $masking_type \
        --dataset $dataset \
        --input_data_path $input_path
done