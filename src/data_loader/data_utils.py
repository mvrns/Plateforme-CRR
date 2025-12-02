# src/data_loader/data_utils.py

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Tuple

def fetch_stock_history(ticker: str, start: str, end: str) -> pd.Series:
    """Fetches historical closing prices for a given ticker."""

    data = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, multi_level_index=False)
    
    # Error anticipation
    if data.empty:
        raise ValueError(f"Aucune donnée trouvée pour {ticker}. Vérifiez le ticker ou les dates.")
    
    # Case where data is a df
    if isinstance(data, pd.DataFrame):
        if 'Close' in data.columns:
            return data['Close'].dropna()
        else:
            return data.iloc[:, 0].dropna()
            
    return data.dropna()

def compute_log_returns(X: pd.Series) -> Tuple[np.ndarray, np.ndarray, float]:
    """Computes daily log-returns and the time step."""
    X_values = X.values
    dt = 1 / 252
    log_returns = np.diff(np.log(X_values))

    return X_values, log_returns, dt