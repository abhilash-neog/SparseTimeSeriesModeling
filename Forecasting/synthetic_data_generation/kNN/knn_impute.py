import numpy as np
import pandas as pd
import argparse
import os

from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Imputation')
parser.add_argument('--masking_type', default="mcar", type=str)
parser.add_argument('--input_data_path', default="", type=str)
parser.add_argument('--dataset', default="ETTh1", type=str)

args = parser.parse_args()

masking_type = args.masking_type
dataset = args.dataset

dataset_path = args.input_data_path
filename = f"v{{}}_{masking_type}_{dataset.lower()}.csv"

out_filename = f"v{{}}_{masking_type}_{dataset.lower()}_imputed_kNN.csv"
out_path = dataset_path

for trial in tqdm(range(5)):
    # Example time-series data
    in_file=filename.format(trial)
    data = pd.read_csv(os.path.join(dataset_path, in_file))

    date_col_name = data.columns[0]
    feature_cols = data.columns[1:]
    date_col = data.date

    data = data[feature_cols]

    # Normalize data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # Initialize KNN Imputer
    imputer = KNNImputer(n_neighbors=10)

    # Perform imputation
    data_imputed = imputer.fit_transform(data_scaled)

    # Inverse transform to get back to original scale
    data_imputed = scaler.inverse_transform(data_imputed)

    # Convert back to DataFrame
    data_imputed = pd.DataFrame(data_imputed, columns=feature_cols)
    data_imputed.insert(0, date_col_name, date_col)

    '''
    SAVE
    '''
    out_file = out_filename.format(trial)
    data_imputed.to_csv(out_path+out_file, index=False)