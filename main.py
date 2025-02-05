#######################################################
# FILE: main.py
#######################################################

import pandas as pd
import numpy as np

# 1) Feature Engineering
from feature_engineering.compute_indicators import compute_technical_indicators

# 2) Data Prep
from data_preparation.sequence_builders import create_sequences_single_step
# If you have an invert_close_scaling or multi-step sequence creation, import it here too.

# 3) MC Dropout LSTM (optional example)
from mc_dropout_utils.mc_dropout import build_lstm_model_mc, mc_dropout_predict

# 4) Backtesting or Walk-Forward (optional)
# from backtesting.backtester import backtest_with_mc_dropout
# from walk_forward.walk_forward import walk_forward_validation

from sklearn.preprocessing import MinMaxScaler

def main():
    """
    A minimal pipeline showing how to:
      1) Load data (from CSV or other source)
      2) Compute technical indicators
      3) Scale data
      4) Create sequences for single-step forecasting
      5) Build an MC Dropout LSTM model
      6) Train and optionally do inference
    """

    # ---------------------------------------------------
    # 1) LOAD DATA
    # ---------------------------------------------------
    # For demonstration, assume we have a CSV with columns:
    #   Date, Open, high, low, Close, volume, ...
    csv_path = "your_data.csv"  # change to your real path
    df = pd.read_csv(csv_path)

    # Optional: parse datetime if needed
    # df['Date'] = pd.to_datetime(df['Date'])

    # ---------------------------------------------------
    # 2) FEATURE ENGINEERING
    # ---------------------------------------------------
    # compute_technical_indicators modifies df in-place, adding indicators.
    df = compute_technical_indicators(df)
    # df now has columns [Open, high, low, Close, volume, RSI, MACD, etc.]

    # ---------------------------------------------------
    # 3) SCALING
    # ---------------------------------------------------
    # Extract numeric columns you want to scale (including your target 'Close')
    # For example, let's keep columns = ['Open', 'high', 'low', 'Close', 'volume', 'RSI', ...]
    # Make sure the order is consistent each time!
    columns_to_scale = ['Open', 'high', 'low', 'Close', 'volume', 'RSI', 'MACD']
    df_model = df[columns_to_scale].copy()

    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df_model.values)
    # scaled_array.shape = (N, num_features)

    # Decide which column index is 'Close' in columns_to_scale
    close_index = columns_to_scale.index('Close')

    # ---------------------------------------------------
    # 4) CREATE SEQUENCES (single-step)
    # ---------------------------------------------------
    seq_len = 30
    X_all, y_all = create_sequences_single_step(
        scaled_array,
        seq_len=seq_len,
        target_col_index=close_index
    )
    print("X_all shape:", X_all.shape, "y_all shape:", y_all.shape)

    # Train/Val split
    train_size = int(len(X_all)*0.8)
    X_train, y_train = X_all[:train_size], y_all[:train_size]
    X_val, y_val = X_all[train_size:], y_all[train_size:]
    print("Training samples:", len(X_train), "Validation samples:", len(X_val))

    # ---------------------------------------------------
    # 5) BUILD MC DROPOUT MODEL
    # ---------------------------------------------------
    model = build_lstm_model_mc(
        input_shape=(seq_len, scaled_array.shape[1]), 
        n_units=100,
        dropout_rate=0.2
    )
    model.summary()

    # ---------------------------------------------------
    # 6) TRAIN
    # ---------------------------------------------------
    model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # ---------------------------------------------------
    # 7) OPTIONAL INFERENCE / MC DROPOUT
    # ---------------------------------------------------
    # Example: do predictions on validation set with multiple passes
    mean_preds, std_preds = mc_dropout_predict(model, X_val, n_samples=50)
    print("MC Dropout Mean Predictions (first 5):", mean_preds[:5])
    print("MC Dropout Std Dev (first 5):", std_preds[:5])

    # If you want to invert scaling for 'Close':
    # from data_preparation.sequence_builders import invert_close_scaling
    # y_pred_inverted = invert_close_scaling(mean_preds.reshape(-1,1), scaler, close_index, scaled_array.shape[1])
    # Then compare with actual y_val (also invert if needed).

    # ---------------------------------------------------
    # 8) (Optional) BACKTEST OR EVALUATE
    # ---------------------------------------------------
    # You might call backtest_with_mc_dropout(...) or other custom logic
    # final_equity, eq_curve, signals, max_dd = backtest_with_mc_dropout(...)

    print("Done with main pipeline. Models trained, predictions generated.")

if __name__ == "__main__":
    main()
