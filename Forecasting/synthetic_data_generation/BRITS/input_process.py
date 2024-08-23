import os
import numpy as np
import pandas as pd
import json

def to_time_bin(x):
    # Convert timestamp to an hourly bin (assuming hourly data)
    return pd.to_datetime(x).hour

def parse_data(row, attributes):
    values = row[attributes].values
    return values

def parse_delta(masks, dir_, attributes):
    if dir_ == 'backward':
        masks = masks[::-1]
    deltas = []
    for h in range(len(masks)):
        if h == 0:
            deltas.append(np.ones(len(attributes)))
        else:
            deltas.append(np.ones(len(attributes)) + (1 - masks[h]) * deltas[h-1])
    return np.array(deltas)

def parse_rec(values, masks, evals, eval_masks, dir_, attributes):
    deltas = parse_delta(masks, dir_, attributes)
    forwards = pd.DataFrame(values).fillna(method='ffill').fillna(0.0).to_numpy()
    rec = {
        'values': np.nan_to_num(values).tolist(),
        'masks': masks.astype('int32').tolist(),
        'evals': np.nan_to_num(evals).tolist(),
        'eval_masks': eval_masks.astype('int32').tolist(),
        'forwards': forwards.tolist(),
        'deltas': deltas.tolist()
    }
    return rec

def convert_to_serializable(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj

def parse_series(series, timestamp_col, attributes, mean, std, fs):
    series = series.copy()  # Avoid SettingWithCopyWarning by working on a copy
    series['time_bin'] = series[timestamp_col].apply(to_time_bin)
    
    evals = series[attributes].values
    evals = (evals - mean) / std
    
    shp = evals.shape
    values = evals.copy()
    
    masks = ~np.isnan(values)
    
    eval_masks = masks
    
    evals = evals.reshape(shp)
    
    values = values.reshape(shp)
    
    masks = masks.reshape(shp)
    
    eval_masks = eval_masks.reshape(shp)
    
    rec = {}
    rec['forward'] = parse_rec(values, masks, evals, eval_masks, dir_='forward', attributes=attributes)
    rec['backward'] = parse_rec(values[::-1], masks[::-1], evals[::-1], eval_masks[::-1], dir_='backward', attributes=attributes)
    rec['time_bin'] = series['time_bin'].iloc[0]
    rec['is_train'] = np.random.randint(0, 2)
    rec = convert_to_serializable(rec)
    
    # print(f"rec = {rec}")
    fs.write(json.dumps(rec) + '\n')

# Split the dataset into sequences (e.g., 48-hour sequences)
def process(data_path, dataset, jsonfilename):
    
    # Load the ETTh2 dataset
    # data_path = '/raid/abhilash/synthetic_datasets/ETTh2/a1/v0_periodic_etth2.csv'  # Adjust this to the actual path of your dataset
    data = pd.read_csv(data_path)

    # Define the timestamp column and attributes
    timestamp_col = data.columns[0]  # Assuming the first column is 'timestamp'
    attributes = data.columns[1:]  # Assuming the rest are attributes

    # Calculate mean and std for these attributes
    mean = data[attributes].mean().values
    std = data[attributes].std().values
    
    json_path = "/raid/abhilash/BRITS/json/"
    
    # Create the output directory
    # if not os.path.exists('./json'):
    #     os.makedirs('./json')

    fs = open(json_path+jsonfilename, 'w')
    sequence_length = 48
    
    for i in range(0, len(data) - sequence_length + 1, sequence_length):
        series = data.iloc[i:i+sequence_length]
        print(f'Processing series starting at index {i}')
        try:
            parse_series(series, timestamp_col, attributes, mean, std, fs)
        except Exception as e:
            print(e)
            continue

    fs.close()
    return mean, std