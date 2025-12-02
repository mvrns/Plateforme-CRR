# src/model/greeks_calculator.py

import math
import numpy as np
from dataclasses import dataclass
import pandas as pd
from .option_pricer import price_option_euro_amer 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Data class for results
@dataclass
class GreeksResult:
    """Container for the main Greeks values at t=0."""
    Delta: float
    Gamma: float
    Vega: float      
    Theta: float     
    Rho: float       
    Price: float 

# Calculation of greeks
def calculate_greeks_at_t0(
    S0: float, K: float, r: float, u: float, d: float, n: int,
    option_type: str = "call", T: float = 1.0,
) -> GreeksResult:
    """
    Calculates Delta, Gamma, Vega, Theta, and Rho at t=0 using the Finite Difference Method.
    """
    
    # Perturbation steps
    epsilon_S = 0.5     
    epsilon_sigma = 0.01  
    epsilon_r = 0.0001    
    epsilon_T = T / 365   
    
    dt = T / n
    
    # Calculate implied sigma 
    try:
        sigma_implied = np.log(u / d) / (2 * np.sqrt(dt))
    except (ZeroDivisionError, ValueError):
        sigma_implied = 0.0

    # Current Price 
    res_current = price_option_euro_amer(S0, K, r, u, d, n, option_type, T)
    V_current = res_current.price_euro
    
    # 1. DELTA AND GAMMA CALCULATION
    S_up = S0 + epsilon_S
    S_down = S0 - epsilon_S
    
    V_up_S = price_option_euro_amer(S_up, K, r, u, d, n, option_type, T).price_euro
    V_down_S = price_option_euro_amer(S_down, K, r, u, d, n, option_type, T).price_euro
    
    Delta = (V_up_S - V_down_S) / (2 * epsilon_S)
    Gamma = (V_up_S - 2 * V_current + V_down_S) / (epsilon_S ** 2)


    # 2. VEGA CALCULATION
    sigma_up = sigma_implied + epsilon_sigma
    sigma_down = sigma_implied - epsilon_sigma
    
    u_vega_up = np.exp(sigma_up * np.sqrt(dt))
    d_vega_up = np.exp(-sigma_up * np.sqrt(dt))
    u_vega_down = np.exp(sigma_down * np.sqrt(dt))
    d_vega_down = np.exp(-sigma_down * np.sqrt(dt))

    V_up_sigma = price_option_euro_amer(S0, K, r, u_vega_up, d_vega_up, n, option_type, T).price_euro
    V_down_sigma = price_option_euro_amer(S0, K, r, u_vega_down, d_vega_down, n, option_type, T).price_euro
    
    Vega = (V_up_sigma - V_down_sigma) / (2 * epsilon_sigma)


    # 3. RHO CALCULATION
    r_up = r + epsilon_r
    r_down = r - epsilon_r

    V_up_r = price_option_euro_amer(S0, K, r_up, u, d, n, option_type, T).price_euro
    V_down_r = price_option_euro_amer(S0, K, r_down, u, d, n, option_type, T).price_euro
    
    Rho = (V_up_r - V_down_r) / (2 * epsilon_r)


    # 4. THETA CALCULATION
    T_up = T + epsilon_T
    T_down = T - epsilon_T
    
    V_T_down = price_option_euro_amer(S0, K, r, u, d, n, option_type, T_down).price_euro
    
    Theta = -(V_current - V_T_down) / epsilon_T


    return GreeksResult(
        Delta=Delta, 
        Gamma=Gamma, 
        Vega=Vega,
        Theta=Theta,
        Rho=Rho,
        Price=V_current
    )


# Function for plotting
def plot_greeks_vs_price(greeks_data: pd.DataFrame, option_type: str):
    """
    Creates an interactive Plotly graph to display Option Price and 5 Greeks
    as a function of the underlying price.
    """
    
    greeks_data = greeks_data.sort_values(by='S')

    fig = make_subplots(rows=6, cols=1, 
                        subplot_titles=(
                            'European Option Price', 
                            'Delta', 
                            'Gamma',
                            'Vega', 
                            'Theta',           
                            'Rho' 
                        ),
                        shared_xaxes=True,
                        vertical_spacing=0.05) 

    # Price
    fig.add_trace(go.Scatter(x=greeks_data['S'], y=greeks_data['Price'], mode='lines', name='Price', line=dict(color='blue')), row=1, col=1)
    # Delta
    fig.add_trace(go.Scatter(x=greeks_data['S'], y=greeks_data['Delta'], mode='lines', name='Delta', line=dict(color='red')), row=2, col=1)
    # Gamma
    fig.add_trace(go.Scatter(x=greeks_data['S'], y=greeks_data['Gamma'], mode='lines', name='Gamma', line=dict(color='green')), row=3, col=1)
    # Vega 
    fig.add_trace(go.Scatter(x=greeks_data['S'], y=greeks_data['Vega'], mode='lines', name='Vega', line=dict(color='gold')), row=4, col=1)
    # Theta 
    fig.add_trace(go.Scatter(x=greeks_data['S'], y=greeks_data['Theta'], mode='lines', name='Theta', line=dict(color='purple')), row=5, col=1)
    # Rho 
    fig.add_trace(go.Scatter(x=greeks_data['S'], y=greeks_data['Rho'], mode='lines', name='Rho', line=dict(color='cyan')), row=6, col=1)


    # Titles and axes
    fig.update_layout(
        title_text=f"Greeks Analysis vs. Underlying Price ({option_type.capitalize()})",
        height=1400, # Increased height to accommodate 6 rows
        template='plotly_white',
        showlegend=False
    )
    
    # X-axis title only on the last plot
    fig.update_xaxes(title_text='Underlying Price', row=6, col=1)
    
    # Y-axis titles
    fig.update_yaxes(title_text='Price', row=1, col=1)
    fig.update_yaxes(title_text='Delta', row=2, col=1)
    fig.update_yaxes(title_text='Gamma', row=3, col=1)
    fig.update_yaxes(title_text='Vega', row=4, col=1)
    fig.update_yaxes(title_text='Theta', row=5, col=1)
    fig.update_yaxes(title_text='Rho', row=6, col=1)
    
    # Detailed views 
    fig.update_traces(
        hovertemplate='S: %{x:.2f}<br>Value: %{y:.4f}<extra></extra>'
    )
    
    return fig