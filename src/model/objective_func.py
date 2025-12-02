# src/model/objective_func.py

import numpy as np
from scipy.optimize import differential_evolution
from typing import Tuple
from src.data_loader.data_utils import fetch_stock_history, compute_log_returns
import hashlib

def _generate_dynamic_seed(ticker: str, start: str, end: str) -> int:
    """Generates a unique seed based on the ticker and period."""
    unique_str = f"{ticker}_{start}_{end}"
    h = hashlib.sha256(unique_str.encode("utf-8")).hexdigest()
    return int(h, 16) % (2**32)

def J_objective_robust(params: Tuple[float, float], 
                        X_j: np.ndarray, 
                        delta_X: np.ndarray,
                        delta_W_fixed: np.ndarray, 
                        dt: float,
                        N: float,
                        r: float) -> float:
    """Objective function."""
    u, d = params
    
    # 1. Security Restrictions
    growth = np.exp(r * dt)
    if u <= growth or d >= growth:
        return 1e10

    try:
        ln_u = np.log(u)
        ln_d = np.log(d)
        ln_ud = ln_u + ln_d       
        ln_u_div_d = ln_u - ln_d 
    except:
        return 1e10
        
    # 2. Model reconstruction 
    term_drift = 0.5 * ln_ud * X_j 
    
    term_diffusion = 0.5 * ln_u_div_d * X_j * delta_W_fixed

    # 3. Error calculation
    residuals = delta_X - term_drift - term_diffusion
    
    if np.isnan(residuals).any():
        return 1e10
        
    mse = np.mean(residuals ** 2)
    return np.sqrt(mse)


def optimize_u_d_from_ticker(ticker: str, start: str, end: str, r: float = 0.05) -> Tuple[float, float]:
    """
    Optimize u,d with dynamic bounds to ensure non-arbitrage.
    """
    
    # 1. Loading and Cleaning Data
    X_series = fetch_stock_history(ticker, start, end)
    X_values_raw, log_returns_raw, dt = compute_log_returns(X_series)
    
    n_returns = len(log_returns_raw)
    n_prices = len(X_values_raw)
    min_len = min(n_returns, n_prices - 1)
    
    if min_len < 10:
        print(f"CRITICAL ERROR: Not enough raw data.")
        return 1.1, 0.9

    log_returns_raw = log_returns_raw[:min_len]
    X_values_raw = X_values_raw[:min_len+1]

    mask = np.isfinite(log_returns_raw)
    log_returns = log_returns_raw[mask]
    X_j = X_values_raw[:-1][mask]
    delta_X = (X_values_raw[1:] - X_values_raw[:-1])[mask]
    
    if len(log_returns) < 10:
        return 1.1, 0.9

    # 2. Stats
    std_dev_history = np.std(log_returns)
    mean_history = np.mean(log_returns)
    if std_dev_history < 1e-9: std_dev_history = 1e-9
    
    delta_W_fixed = (log_returns - mean_history) / std_dev_history
    N = float(len(log_returns))

    # 3. Optimization with dynamic bounds
    
    growth_factor = np.exp(r * dt)
    
    min_u = growth_factor + 0.0005 
    max_d = growth_factor - 0.0005
    
    bounds = [(min_u, 1.4), (0.5, max_d)]
    
    seed = _generate_dynamic_seed(ticker, start, end)

    result = differential_evolution(
        J_objective_robust,
        bounds,
        args=(X_j, delta_X, delta_W_fixed, dt, N, r),
        maxiter=1000,
        popsize=20,
        tol=1e-7,
        seed=seed,
        strategy='best1bin',
        polish=True
    )

    u_opt, d_opt = result.x
    
    # 4. Safety net
    safe_margin = 0.0001
    if u_opt <= growth_factor:
        u_opt = growth_factor + safe_margin
        print(f"-> Forced correction u: {u_opt:.6f}")
        
    if d_opt >= growth_factor:
        d_opt = growth_factor - safe_margin
        print(f"-> Forced correction d: {d_opt:.6f}")

    # 5. Display
    print("=" * 60)
    print(f"ROBUST Calibration (+Dynamic Bounds) for {ticker}")
    print(f"Period: {start} -> {end} | N={int(N)} days")
    print("-" * 60)
    print(f"Optimal u       : {u_opt:.6f}")
    print(f"Optimal d       : {d_opt:.6f}")
    ln_u_div_d = np.log(u_opt / d_opt)
    implied_sigma = (1 / (2 * np.sqrt(dt))) * ln_u_div_d
    print(f"Implied Volatility (CRR): {implied_sigma:.4f}")
    print(f"Model Error (RMSE)      : {result.fun:.6f}")
    print("=" * 60)
    
    return u_opt, d_opt