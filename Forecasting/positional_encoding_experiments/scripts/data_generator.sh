DATASET=$1
PVALUES=$2
MASKING=$3
SOURCE_FILE=$4

python -u data_generator.py \
    --dataset $DATASET \
    --pvalues $PVALUES \
    --masking_type $MASKING \
    --source_file $SOURCE_FILE