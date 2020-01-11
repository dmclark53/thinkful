# methods.py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam


def set_params(param_dict, param, value):
    param_dict[param] = value


def build_mlp_model(metrics=None, params=None):
    model = Sequential([
        Dense(512, input_shape=(params['input_shape'],), activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=1e-3), metrics=metrics)

    return model


def build_mlp_model_2(metrics=None, params=None):
    model = Sequential([
        Dense(16, input_shape=(params['input_shape'],), activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=1e-3), metrics=metrics)

    return model


def build_lstm_model(metrics=None, params=None):
    model = Sequential([
        Embedding(params['vocab_size'], params['embedding_size'], input_length=params['input_length']),
        # Dropout(params['drop_out_prob'], input_shape=(params['embedding_size'], )),
        LSTM(params['lstm_units']),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=1e-3), metrics=metrics)

    return model


def build_embedding_model(metrics=None, params=None):
    model = Sequential([
        Embedding(params['vocab_size'], params['embedding_size'], input_length=params['input_length']),
        Flatten(),
        Dense(250, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    return model


def print_test_results(model, results):
    for name, value in zip(model.metrics_names, results):
        print(f'{name}: {value}')

