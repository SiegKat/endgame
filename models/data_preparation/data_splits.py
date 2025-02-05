#######################################################
# FILE: data_preparation/data_splits.py
#######################################################
import pandas as pd
import numpy as np

def train_test_by_year(df, train_years, test_year, date_col='Date'):
    """
    Splits a DataFrame into train/test segments based on calendar years.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a date column (by default named 'Date').
    train_years : tuple of (int, int)
        The inclusive year range for training, e.g. (2015, 2017) for 2015-01-01 to 2017-12-31.
    test_year : int
        The single year to be used for testing.
    date_col : str, optional
        Column name for the date field, by default 'Date'.

    Returns
    -------
    df_train : pd.DataFrame
        The training subset of df.
    df_test : pd.DataFrame
        The testing subset of df.

    Notes
    -----
    - Ensure df[date_col] is in a valid datetime format.
      If not, do something like: df[date_col] = pd.to_datetime(df[date_col]).
    - This approach uses simple calendar slicing. 
      For more flexible or partial-year splits, adapt accordingly.
    """
    # Make sure the date_col is datetime
    # If needed, uncomment:
    # df[date_col] = pd.to_datetime(df[date_col])

    mask_train = (
        (df[date_col] >= f"{train_years[0]}-01-01") &
        (df[date_col] <= f"{train_years[1]}-12-31")
    )
    mask_test  = (
        (df[date_col] >= f"{test_year}-01-01") &
        (df[date_col] <= f"{test_year}-12-31")
    )

    df_train = df.loc[mask_train].copy()
    df_test  = df.loc[mask_test].copy()
    return df_train, df_test


def multi_regime_backtest(df):
    """
    Demonstration of looping over multiple test years to do a simple multi-regime test.

    Example:
      - Train on 2015-2017, Test on 2018
      - Train on 2015-2018, Test on 2019
      - Train on 2015-2019, Test on 2020

    Notes
    -----
    This is just a placeholder example. You would:
      1) Call 'train_test_by_year(df, (2015, y-1), y)' to get train_df, test_df
      2) Do feature engineering on train_df, test_df
      3) Scale your columns, create sequences, etc.
      4) Build and train your model on train_df
      5) Evaluate on test_df (compute metrics, store in results)

    Returns
    -------
    results : list
        A placeholder. In practice, you'd fill it with your actual performance metrics.
    """
    years = [2018, 2019, 2020]
    results = []
    for y in years:
        # 1) Grab data
        train_df, test_df = train_test_by_year(df, (2015, y-1), y)

        # 2) feature engineering, e.g.
        # train_df = compute_technical_indicators(train_df)
        # test_df = compute_technical_indicators(test_df)

        # 3) scale data, create sequences...
        # e.g. X_train, y_train = create_sequences_single_step(...)
        # X_test, y_test = create_sequences_single_step(...)

        # 4) build model, train
        # model.fit(X_train, y_train)

        # 5) evaluate on test_df
        # metrics = evaluate(model, X_test, y_test)
        # results.append((y, metrics))

        pass

    return results
