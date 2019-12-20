# methods.py
import numpy as np
import os
import pandas as pd
from tensorflow.keras.preprocessing import sequence

from .constants import ACTIVITY_LIST
from .constants import ACTIVITY_MAP
from .constants import DATA_DIRECTORY
from .constants import RAW_DATA_DIRECTORY
from .constants import DATA_DICT


def map_activities():
    with open(os.path.join(DATA_DIRECTORY, 'activity_key.txt')) as activity_key:
        for line in activity_key:
            key, value = line.partition("=")[::2]
            if (key != '') and (value != ''):
                ACTIVITY_MAP[key.strip()] = value.strip().strip('\n')


def create_file_list(device_type, sensor_type):
    sensor_directory = os.path.join(RAW_DATA_DIRECTORY, device_type, sensor_type)
    sensor_files = []
    with os.scandir(sensor_directory) as entries:
        for entry in entries:
            if entry.is_file() and 'data' in entry.name:
                sensor_files.append(os.path.join(sensor_directory, entry.name))
    return sensor_files


def load_data(sensor_files):
    for sensor_file in sensor_files:
        df = pd.read_csv(sensor_file, header=None,
                         names=['subject_id', 'activity_code', 'timestamp', 'x', 'y', 'z'],
                         lineterminator=';')
        for activity in ACTIVITY_LIST:
            sub_df = df.loc[df['activity_code'] == ACTIVITY_MAP[activity], ['x', 'y', 'z']].copy()
            if len(sub_df) > 0:
                DATA_DICT['dfs'].append(sub_df)
                DATA_DICT['labels'].append(ACTIVITY_LIST.index(activity))
    return pd.concat(DATA_DICT['dfs'], axis=0)


def create_features(max_sequence_length):
    X = None
    for df in DATA_DICT['dfs']:
        sequences = [df.iloc[:, i].values.tolist() for i in range(df.values.shape[1])]
        sensor_array_padded = sequence.pad_sequences(sequences, maxlen=max_sequence_length)
        reshaped_array = sensor_array_padded.reshape(-1, sensor_array_padded.shape[1], sensor_array_padded.shape[0])
        if X is None:
            X = reshaped_array
        else:
            X = np.concatenate((X, reshaped_array), axis=0)
    return X


