# methods.py
from tensorflow.keras.backend import clear_session
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential


def set_params(param_dict, param, value):
    param_dict[param] = value


def build_mlp_model(metrics=None, params=None):
    clear_session()

    model = Sequential([
        Dense(512, input_shape=(params['input_shape'],), activation='relu'),
        Dropout(0.3),
        Dense(512, activation='relu'),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=1e-3), metrics=metrics)

    model.summary()

    return model


def build_mlp_model_2(metrics=None, params=None):
    clear_session()

    model = Sequential([
        Dense(16, input_shape=(params['input_shape'],), activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=1e-3), metrics=metrics)

    model.summary()

    return model


def build_lstm_model(metrics=None, params=None):
    clear_session()

    model = Sequential([
        Embedding(params['vocab_size'], params['embedding_size'], input_length=params['input_length']),
        # Dropout(params['drop_out_prob'], input_shape=(params['embedding_size'], )),
        LSTM(params['lstm_units']),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss=BinaryCrossentropy(), optimizer=Adam(lr=1e-3), metrics=metrics)

    model.summary()

    return model


def build_embedding_model(metrics=None, params=None):
    clear_session()

    model = Sequential([
        Embedding(params['vocab_size'], params['embedding_size'], input_length=params['input_length']),
        Flatten(),
        Dense(250, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=metrics)

    model.summary()

    return model


def print_test_results(model, results):
    print('Test Metrics:')
    for name, value in zip(model.metrics_names, results):
        if len(name) > 3:
            name = name.title()
        else:
            name = name.upper()
        print(f' * {name}: {value}')

