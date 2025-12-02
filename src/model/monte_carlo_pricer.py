# src/model/monte_carlo_pricer.py

import numpy as np

def monte_carlo_option_price(
    S0: float, K: float, r: float, T: float, sigma: float, 
    N_simulations: int, N_pas: int, option_type: str = "call"
) -> float:
    """
    Calculates the price of a European option using Monte Carlo simulation (GBM).
    """
    
    dt = T / N_pas
    drift = (r - 0.5 * sigma**2) * dt
    vol_term = sigma * np.sqrt(dt)
    
    # Array to store the final prices of all simulations
    S_T = S0 * np.ones(N_simulations)
    
    # 1. Path Simulation
    for _ in range(N_pas):
        Z = np.random.normal(0, 1, N_simulations)
        S_T = S_T * np.exp(drift + vol_term * Z)
        
    # 2. Payoff Calculation 
    if option_type.lower() == "call":
        payoffs = np.maximum(0, S_T - K)
    elif option_type.lower() == "put":
        payoffs = np.maximum(0, K - S_T)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
    
    # 3. Discounting and Averaging
    average_payoff = np.mean(payoffs)
    price = average_payoff * np.exp(-r * T)
    
    return float(price)