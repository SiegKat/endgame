def backtest_with_mc_dropout(
    model,
    X_data,
    prices,
    mc_samples=50,
    stop_loss_pct=0.02,
    risk_per_trade=0.01,
    initial_capital=10000
):
    """
    A naive single-step backtest that uses MC Dropout predictions.
    For each time step i in X_data:
      1) We do multiple forward passes (MC dropout).
      2) Compare predicted price (mean_pred) with current_price.
      3) If not in a position and mean_pred > current_price, we go long.
      4) If in a position, we hold until the next bar (or until stopped out).
      5) Track equity, signals, and drawdowns.

    Parameters
    ----------
    model : Keras model
        A trained model that supports MC Dropout inference.
    X_data : np.array
        Shape (N, seq_len, num_features). The input sequences to predict.
    prices : np.array
        Shape (N,). The actual current price at each step for deciding buy/sell.
    mc_samples : int, optional
        Number of dropout samples for each prediction, by default 50
    stop_loss_pct : float, optional
        Fraction below entry_price that triggers a stop, by default 0.02
    risk_per_trade : float, optional
        Fraction of capital to risk if stop is hit, by default 0.01
    initial_capital : float, optional
        Starting capital, by default 10000

    Returns
    -------
    final_equity : float
        The final capital after all trades
    equity_curve : list of float
        The capital value after each step
    signals : list of dict
        A record of BUY/SELL/STOP actions
    max_drawdown : float
        The maximum drawdown encountered in the backtest
    """
    capital = initial_capital
    equity_curve = []
    signals = []
    position_open = False
    position_size = 0
    entry_price = 0
    peak_equity = capital
    max_drawdown = 0

    # Import inside the function so it doesn't break other usage
    from mc_dropout_utils.mc_dropout import mc_dropout_predict

    for i in range(len(X_data)):
        # 1) MC Dropout prediction
        X_single = X_data[i:i+1]
        mean_pred_scaled, std_pred_scaled = mc_dropout_predict(
            model, X_single, n_samples=mc_samples
        )
        mean_pred = mean_pred_scaled[0]
        std_pred = std_pred_scaled[0]  # not strictly used in logic, but stored for analysis

        current_price = prices[i]

        # 2) If no position open, check if we want to buy
        if not position_open:
            if mean_pred > current_price:
                stop_price = current_price * (1 - stop_loss_pct)
                position_size = (capital * risk_per_trade) / (current_price - stop_price)
                entry_price = current_price
                position_open = True
                signals.append({
                    'index': i,
                    'type': 'BUY',
                    'entry_price': entry_price,
                    'position_size': position_size,
                    'stop_price': stop_price,
                    'mean_pred': mean_pred,
                    'std_pred': std_pred
                })

        # 3) If position is open, either get stopped out or close after 1 bar
        else:
            stop_price = entry_price * (1 - stop_loss_pct)
            if current_price <= stop_price:
                # Stopped out
                capital -= position_size * (entry_price - stop_price)
                position_open = False
                signals.append({
                    'index': i,
                    'type': 'STOP',
                    'exit_price': stop_price,
                    'capital': capital
                })
            else:
                # Close after 1 bar (naive approach)
                capital += position_size * (current_price - entry_price)
                position_open = False
                signals.append({
                    'index': i,
                    'type': 'SELL',
                    'exit_price': current_price,
                    'capital': capital
                })

        # 4) Update equity curve & drawdown
        equity_curve.append(capital)
        if capital > peak_equity:
            peak_equity = capital
        drawdown = (peak_equity - capital) / peak_equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown

    final_equity = capital
    return final_equity, equity_curve, signals, max_drawdown
