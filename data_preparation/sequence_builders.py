#######################################################
# FILE: data_preparation/sequence_builders.py
#######################################################

import numpy as np

def create_sequences_single_step(data, seq_len=30, target_col_index=3):
    """
    Prepares data for single-step forecasting.

    Parameters
    ----------
    data : np.array
        Shape (N, num_features). The scaled dataset (e.g., from MinMaxScaler).
    seq_len : int, optional
        Number of time steps in each sequence, by default 30.
    target_col_index : int, optional
        Which column in 'data' is the target (e.g., 'Close'), by default 3.

    Returns
    -------
    X : np.array
        Shape (N - seq_len, seq_len, num_features).
    y : np.array
        Shape (N - seq_len,). The single-step target for each sequence.
    """
    X, y = [], []
    for i in range(seq_len, len(data)):
        # Take the past seq_len rows as features
        X.append(data[i-seq_len:i, :])
        # The target is the value at time i in the target_col_index
        y.append(data[i, target_col_index])

    X = np.array(X)
    y = np.array(y)
    return X, y


def create_sequences_multi_step(data, seq_len=30, horizon=5, target_col_index=3):
    """
    Prepares data for multi-step forecasting.

    E.g., if horizon=5, you're predicting the next 5 bars (t+1 to t+5).

    Parameters
    ----------
    data : np.array
        Shape (N, num_features). The scaled dataset.
    seq_len : int, optional
        Number of time steps in each input sequence, by default 30.
    horizon : int, optional
        How many steps ahead to predict, by default 5.
    target_col_index : int, optional
        Which column is the target, by default 3.

    Returns
    -------
    X : np.array
        Shape (M, seq_len, num_features).
    y : np.array
        Shape (M, horizon).
        The next 'horizon' target values (e.g., [t+1, ..., t+horizon]).
    """
    X, y = [], []
    for i in range(seq_len, len(data) - horizon + 1):
        X.append(data[i - seq_len : i, :])
        # Collect next 'horizon' values
        future_targets = data[i : i + horizon, target_col_index]
        y.append(future_targets)

    X = np.array(X)
    y = np.array(y)
    return X, y


def create_sequences_direction(data, seq_len=30, alpha=0.0, close_index=3):
    """
    Prepares data for a classification approach (direction: up or down).

    A typical label is 1 if (Close_{t+1} / Close_t - 1) >= alpha, else 0.

    Parameters
    ----------
    data : np.array
        Shape (N, num_features). The scaled dataset.
    seq_len : int, optional
        Number of time steps in each sequence, by default 30.
    alpha : float, optional
        Threshold for deciding up vs. down, by default 0.0 (strictly 'is next close > current close?').
    close_index : int, optional
        Which column is 'Close', by default 3.

    Returns
    -------
    X : np.array
        Shape (N - seq_len - 1, seq_len, num_features).
    y : np.array
        Shape (N - seq_len - 1,). 1 or 0 depending on up vs. down label.
    """
    X, y = [], []
    for i in range(seq_len, len(data) - 1):
        # Past seq_len rows
        X.append(data[i - seq_len : i, :])
        # Direction label
        current_close = data[i - 1, close_index]
        next_close    = data[i, close_index]
        pct_diff      = (next_close - current_close) / current_close
        label = 1 if pct_diff >= alpha else 0
        y.append(label)

    X = np.array(X)
    y = np.array(y)
    return X, 


def invert_close_scaling(pred_scaled, scaler, close_index, n_features):
    """
    Inverse-transforms only the 'Close' column after making a prediction on scaled data.

    Parameters
    ----------
    pred_scaled : np.array
        Shape (M, 1) or (M,) - scaled predictions of the 'Close' column.
    scaler : MinMaxScaler or similar
        The same scaler object that was fit on the entire feature set.
    close_index : int
        The column index in the original data that corresponds to 'Close'.
    n_features : int
        The total number of features scaled with 'scaler'.

    Returns
    -------
    inv_close : np.array
        Shape (M,), the predictions back in the original scale of 'Close'.

    Notes
    -----
    - Because 'scaler' was fit on all columns, we must reconstruct an array
      of shape (M, n_features) with zeros, then place pred_scaled in the 'Close'
      column, and finally call scaler.inverse_transform(...) to get the real scale.
    """
    # Create a temp array of zeros
    zeros = np.zeros((pred_scaled.shape[0], n_features))
    # Insert the scaled predictions into the 'close_index' column
    zeros[:, close_index] = pred_scaled.ravel()
    # Inverse transform
    inv = scaler.inverse_transform(zeros)
    # Return just the close column
    return inv[:, close_index]
