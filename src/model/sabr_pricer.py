import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .bs_pricer import black_scholes_price

def sabr_volatility(F, K, T, alpha, beta, rho, nu):
    if K <= 0 or F <= 0: return 0.0
    
    log_FK = np.log(F / K)
    
    if abs(log_FK) < 1e-8:
        term1 = alpha / (F ** (1 - beta))
        term2 = 1 + ( ((1 - beta)**2 / 24) * (alpha**2 / (F**(2 - 2*beta))) + (rho * beta * nu * alpha) / (4 * (F**(1 - beta))) + ((2 - 3 * rho**2) * nu**2) / 24 ) * T
        return term1 * term2

    z = (nu / alpha) * ((F * K) ** ((1 - beta) / 2)) * log_FK
    
    if abs(z) < 1e-8:
        x_z = 1.0
    else:
        sqrt_term = np.sqrt(1 - 2 * rho * z + z**2)
        x_z = np.log((sqrt_term + z - rho) / (1 - rho)) / z

    denominator = ((F * K) ** ((1 - beta) / 2)) * (1 + ((1 - beta)**2 / 24) * log_FK**2 + ((1 - beta)**4 / 1920) * log_FK**4)
    numerator = alpha * z
    
    correction = 1 + ( ((1 - beta)**2 / 24) * (alpha**2 / ((F * K)**(1 - beta))) + (rho * beta * nu * alpha) / (4 * ((F * K)**((1 - beta) / 2))) + ((2 - 3 * rho**2) * nu**2) / 24 ) * T
    
    implied_vol = (alpha / denominator) * (z / x_z) * correction
    
    return implied_vol

def plot_sabr_analysis(S0, K_target, r, T, option_type, alpha, beta, rho, nu):
    F0 = S0 * np.exp(r * T)
    
    strikes = np.linspace(F0 * 0.5, F0 * 1.5, 100)
    vols = []
    prices_sabr = []
    
    for K_val in strikes:
        vol = sabr_volatility(F0, K_val, T, alpha, beta, rho, nu)
        vols.append(vol)
        
        price = black_scholes_price(S0, K_val, r, T, vol, option_type)
        prices_sabr.append(price)

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=strikes, y=vols, name="SABR Implied Vol", line=dict(color='firebrick', width=3)),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=strikes, y=prices_sabr, name=f"Price ({option_type})", line=dict(color='blue', dash='dot')),
        secondary_y=True
    )

    fig.add_vline(x=K_target, line_dash="dash", line_color="green", annotation_text="Target Strike")

    fig.update_layout(
        title="SABR Model: Volatility Smile & Price Structure",
        template='plotly_white',
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Strike Price (K)")
    fig.update_yaxes(title_text="Implied Volatility", secondary_y=False)
    fig.update_yaxes(title_text="Option Price", secondary_y=True)
    
    vol_atm = sabr_volatility(F0, K_target, T, alpha, beta, rho, nu)
    final_price = black_scholes_price(S0, K_target, r, T, vol_atm, option_type)
    
    return fig, final_price, vol_atm