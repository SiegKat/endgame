# Below is a final consolidated version of your Optuna file, now including:
# 1. LSTM objective
# 2. TCN objective
# 3. Transformer objective
# 4. run_optuna_tuning to pick which objective to run based on arch_type.
#
# This snippet is intended to be placed in something like:
# hyperparam_optimization/optuna_search.py
# and assumes:
# - You have create_sequences_single_step(...) in data_preparation/sequence_builders.py
# - global scaled_array, close_index are defined, or you pass them as arguments.
# - You have installed keras-tcn for TCN.
# - You have a minimal Transformer-based approach.

import optuna
from optuna.trial import TrialState
import numpy as np
import math

# from data_preparation.sequence_builders import create_sequences_single_step
# global scaled_array, close_index  # Typically you'd pass these in or define them above.

########################################################
# 1) LSTM Objective
########################################################

def objective_lstm(trial):
    """
    Objective function for LSTM hyperparameter optimization using Optuna.
    """
    # Suggest hyperparams
    n_units_1 = trial.suggest_int("n_units_1", 32, 128, step=32)
    n_units_2 = trial.suggest_int("n_units_2", 32, 128, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    seq_len = 30  # or trial.suggest_int("seq_len", 20, 60, step=10)

    # Access or pass in scaled_array, close_index
    global scaled_array, close_index
    X_all, y_all = create_sequences_single_step(
        scaled_array, seq_len=seq_len, target_col_index=close_index
    )

    train_size = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:], y_all[train_size:]

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam

    model = Sequential()
    model.add(
        Bidirectional(
            LSTM(n_units_1, return_sequences=True),
            input_shape=(seq_len, scaled_array.shape[1])
        )
    )
    model.add(Dropout(dropout_rate))
    model.add(Bidirectional(LSTM(n_units_2, return_sequences=False)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # single-step forecast

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )

    val_loss = history.history['val_loss'][-1]
    return val_loss


########################################################
# 2) TCN Objective
########################################################

def objective_tcn(trial):
    """
    Objective function for TCN hyperparameter optimization using Optuna.
    """
    n_units_1 = trial.suggest_int("n_units_1", 32, 128, step=32)
    n_units_2 = trial.suggest_int("n_units_2", 32, 128, step=32)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    seq_len = 30

    global scaled_array, close_index
    X_all, y_all = create_sequences_single_step(
        scaled_array, seq_len=seq_len, target_col_index=close_index
    )
    train_size = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:], y_all[train_size:]

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tcn import TCN  # ensure you have 'pip install keras-tcn'

    model = Sequential()
    # First TCN layer
    model.add(
        TCN(
            n_units_1,  # number of filters
            return_sequences=True,
            input_shape=(seq_len, scaled_array.shape[1])
        )
    )
    model.add(Dropout(dropout_rate))
    # Second TCN layer
    model.add(TCN(n_units_2, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )

    val_loss = history.history['val_loss'][-1]
    return val_loss


########################################################
# 3) Transformer Objective
########################################################

from tensorflow.keras.layers import (Input, Dense, Dropout, LayerNormalization,
                                     MultiHeadAttention, GlobalAveragePooling1D)
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


def transformer_block(x, num_heads=4, d_model=64, ff_dim=256, rate=0.1):
    """
    A single Transformer block with:
    - MultiHeadAttention
    - Residual + LayerNorm
    - FeedForward + Residual + LayerNorm
    """
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn_output)

    # Feed Forward
    ffn = tf.keras.Sequential([
        Dense(ff_dim, activation='relu'),
        Dense(d_model),
    ])
    ffn_output = ffn(out1)
    ffn_output = Dropout(rate)(ffn_output)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ffn_output)
    return out2


def objective_transformer(trial):
    """
    Objective function for Transformer hyperparameter optimization using Optuna.
    """
    # Suggest hyperparams
    d_model = trial.suggest_int("d_model", 32, 128, step=32)  # dimension for MHA
    num_heads = trial.suggest_int("num_heads", 2, 8, step=2)
    ff_dim = trial.suggest_int("ff_dim", 64, 256, step=64)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    num_blocks = trial.suggest_int("num_blocks", 1, 3, step=1)

    seq_len = 30  # or trial.suggest_int("seq_len", 20, 60, step=10)

    global scaled_array, close_index
    X_all, y_all = create_sequences_single_step(
        scaled_array, seq_len=seq_len, target_col_index=close_index
    )

    train_size = int(len(X_all) * 0.8)
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:], y_all[train_size:]

    # Build the Transformer model
    inputs = Input(shape=(seq_len, scaled_array.shape[1]))
    # Project to d_model
    x = Dense(d_model)(inputs)

    for _ in range(num_blocks):
        x = transformer_block(x, num_heads=num_heads, d_model=d_model, ff_dim=ff_dim, rate=dropout_rate)

    # Pool and final output
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)
    outputs = Dense(1)(x)

    model = Model(inputs, outputs)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        verbose=0
    )

    val_loss = history.history['val_loss'][-1]
    return val_loss


########################################################
# 4) run_optuna_tuning
########################################################

def run_optuna_tuning(n_trials=20, arch_type='lstm'):
    """
    Creates an Optuna study and calls the correct objective function
    based on arch_type.
    """
    study = optuna.create_study(direction='minimize')

    if arch_type == 'lstm':
        study.optimize(objective_lstm, n_trials=n_trials)
    elif arch_type == 'tcn':
        study.optimize(objective_tcn, n_trials=n_trials)
    elif arch_type == 'transformer':
        study.optimize(objective_transformer, n_trials=n_trials)
    else:
        raise ValueError(f"Unknown arch_type: {arch_type}")

    print("Number of finished trials:", len(study.trials))
    best_trial = study.best_trial
    print(f"Best trial val_loss: {best_trial.value}")
    print("Best params:", best_trial.params)
    return study


# Example usage:
# study = run_optuna_tuning(n_trials=30, arch_type='transformer')
# best_params = study.best_trial.params
# Then build a final model with best_params.