from mc_dropout_utils.mc_dropout import build_lstm_model_mc

# Suppose your training data (X_train) has shape (N, seq_len, num_features).
# For example, seq_len = 30, num_features might be 6 (Open, High, Low, Close, Volume, RSI, etc.).
# So X_train.shape == (N, 30, num_features), y_train.shape == (N,).

num_features = X_train.shape[2]
seq_len = X_train.shape[1]  # typically 30

# 1) Build the MC Dropout model
model = build_lstm_model_mc(input_shape=(seq_len, num_features))

# 2) Train the model
# e.g. for 10 epochs
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val),
    verbose=1
)

# 3) Use the trained model for MC Dropout inference
from mc_dropout_utils.mc_dropout import mc_dropout_predict

# Suppose X_test has shape (M, 30, num_features)
mean_preds, std_preds = mc_dropout_predict(model, X_test, n_samples=50)

# mean_preds, std_preds both have shape (M,). 
# You can interpret std_preds as an approximation of prediction uncertainty.
