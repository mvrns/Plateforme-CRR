# src/model/bs_pricer.py

import numpy as np
import scipy.stats as si 
import pandas as pd
import plotly.graph_objects as go

def black_scholes_price(S: float, K: float, r: float, T: float, sigma: float, option_type: str = "call") -> float:
    """
    Calculates the price of a European option using the Black-Scholes model.
    """
    
    if T <= 0 or sigma <= 0:
        return max(0, S - K) if option_type.lower() == "call" else max(0, K - S)

    # D1 and d2 calculation
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type.lower() == "call":
        price = S * si.norm.cdf(d1, 0.0, 1.0) - K * np.exp(-r * T) * si.norm.cdf(d2, 0.0, 1.0)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * si.norm.cdf(-d2, 0.0, 1.0) - S * si.norm.cdf(-d1, 0.0, 1.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")
        
    return float(price)


def plot_discretization_analysis(emp_prices: pd.DataFrame, bs_price: float, option_type: str):
    """
    Creates a Plotly graph to show the stability of the Empirical CRR price (fixed u_opt, d_opt)
    as a function of N, compared to the Black-Scholes price.
    """
    
    fig = go.Figure()
    
    # Plotting Empirical CRR prices vs N
    fig.add_trace(go.Scatter(
        x=emp_prices['N'], 
        y=emp_prices['CRR_Price'], 
        mode='lines+markers', 
        name='Empirical CRR Price (fixed u, d)',
        line=dict(color='blue')
    ))
    
    # Plotting Black-Scholes price (horizontal line)
    fig.add_trace(go.Scatter(
        x=emp_prices['N'], 
        y=[bs_price] * len(emp_prices), 
        mode='lines', 
        name=f'Black-Scholes Reference ({bs_price:.4f})',
        line=dict(color='red', dash='dash', width=2)
    ))
    
    fig.update_layout(
        title=f"Stability of Empirical CRR Price vs. N ({option_type.capitalize()})",
        xaxis_title="Number of Binomial Model Steps (N)",
        yaxis_title="European Option Price",
        template='plotly_white',
        height=500,
        hovermode="x unified"
    )
    
    return fig