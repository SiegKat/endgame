import math
import numpy as np
from sklearn.metrics import mean_squared_error

# from data_preparation.sequence_builders import create_sequences_single_step
# from your_model_code import build_lstm_model or other builder

def walk_forward_validation(
    data, 
    seq_len, 
    close_index, 
    model_builder_func, 
    train_sizes, 
    val_size
):
    """
    Performs a simple walk-forward (or expanding window) validation.

    For each 'train_end' in train_sizes:
      1) Train on [0..train_end-1] of data
      2) Validate on [train_end..train_end+val_size-1]
      3) Build a fresh model using model_builder_func
      4) Train for a few epochs
      5) Compute RMSE on the validation set
      6) Store results in a list

    Parameters
    ----------
    data : np.array
        Shape (N, num_features). Typically scaled data (e.g., from MinMaxScaler).
    seq_len : int
        The sequence length for create_sequences_single_step(...).
    close_index : int
        Which column in data is the target (e.g., 'Close').
    model_builder_func : callable
        A function that, given input_shape=(seq_len, num_features), returns
        a compiled Keras model. e.g. build_lstm_model(input_shape).
    train_sizes : list of int
        The end indices for each training set. e.g., [500, 1000, 1500].
    val_size : int
        How many data points for validation after each train_end index.

    Returns
    -------
    results : list of (train_end, rmse_val)
        A list of tuples with the final RMSE on the validation set for each step.

    Notes
    -----
    - This approach trains a new model for each walk-forward iteration.
    - If you prefer an *expanding window*, ensure your train_sizes are cumulative
      (e.g. 500, 800, 1000...). If you want a rolling window, you can adapt the code.
    - Make sure 'data' includes at least train_end + val_size rows at each step.
    """
    results = []
    for train_end in train_sizes:
        # 1) Split data
        train_data = data[:train_end]
        val_data   = data[train_end : train_end + val_size]

        # If not enough data in val, break early
        if len(val_data) < seq_len:
            break

        # 2) Build sequences for train/val
        X_train, y_train = create_sequences_single_step(
            train_data, seq_len=seq_len, target_col_index=close_index
        )
        X_val, y_val = create_sequences_single_step(
            val_data, seq_len=seq_len, target_col_index=close_index
        )

        # 3) Build a fresh model
        model = model_builder_func(input_shape=(seq_len, data.shape[1]))

        # 4) Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0)

        # 5) Predict and compute RMSE
        y_pred_val = model.predict(X_val)
        rmse_val = math.sqrt(mean_squared_error(y_val, y_pred_val))
        results.append((train_end, rmse_val))

        print(f"Train End: {train_end}, Val RMSE: {rmse_val:.4f}")

    return results
