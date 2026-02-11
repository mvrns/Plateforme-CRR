# app.py
import datetime 
import streamlit as st
import numpy as np
import pandas as pd 
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.model.option_pricer import price_option_euro_amer, ReplicatingResult
from src.model.objective_func import optimize_u_d_from_ticker
from src.data_loader.data_utils import fetch_stock_history 
from src.model.greeks_calculator import calculate_greeks_at_t0, plot_greeks_vs_price 
from src.model.monte_carlo_pricer import monte_carlo_option_price
from src.model.bs_pricer import black_scholes_price, plot_discretization_analysis
from src.model.sabr_pricer import plot_sabr_analysis
from src.model.heston_pricer import plot_heston_analysis, heston_option_price_mc


# HEATMAP PLOTTING FUNCTION
def plot_heatmap_matrices_plotly(result: ReplicatingResult, n: int, option_type: str):
    """Creates an interactive Heatmap chart for Delta and Psi."""
    
    delta_data = result.Delta_amer
    psi_data = result.Psi_amer
    
    df_delta = pd.DataFrame(delta_data.T, index=[f"J={j}" for j in range(delta_data.shape[1])], columns=[f"i={i}" for i in range(delta_data.shape[0])]).iloc[::-1]
    df_psi = pd.DataFrame(psi_data.T, index=[f"J={j}" for j in range(psi_data.shape[1])], columns=[f"i={i}" for i in range(psi_data.shape[0])]).iloc[::-1]

    fig_combined = make_subplots(
        rows=1, cols=2, 
        subplot_titles=('Delta Matrix (Price Sensitivity)', 'Psi Matrix (Risk-Free Investment)')
    )

    fig_combined.add_trace(go.Heatmap(
        z=df_delta.values, x=df_delta.columns, y=df_delta.index, colorscale="RdYlBu", name='Delta',
        coloraxis='coloraxis1', hovertemplate='Time: %{x}<br>Up Moves: %{y}<br>Delta: %{z:.4f}<extra></extra>'
    ), row=1, col=1)
    
    fig_combined.add_trace(go.Heatmap(
        z=df_psi.values, x=df_psi.columns, y=df_psi.index, colorscale="Viridis", name='Psi',
        coloraxis='coloraxis2', hovertemplate='Time: %{x}<br>Up Moves: %{y}<br>Psi: %{z:.4f}<extra></extra>'
    ), row=1, col=2)
    
    fig_combined.update_layout(
        title_text=f"Dynamic Hedging ({option_type.capitalize()} American)",
        height=700, template='plotly_white',
        coloraxis1=dict(colorscale="RdYlBu", colorbar=dict(title="Delta (Shares)", x=0.45, len=0.9)),
        coloraxis2=dict(colorscale="Viridis", colorbar=dict(title="Psi (Risk-Free Amount)", x=1.0))
    )
    
    fig_combined.update_xaxes(title_text='Time (i: from 0 to N-1)')
    fig_combined.update_yaxes(title_text='Number of Up Moves (j)', row=1, col=1)
    fig_combined.update_yaxes(showticklabels=False, row=1, col=2) 
    
    return fig_combined


# MAIN CALCULATION LOGIC
def run_pricing_and_display(
    ticker: str, start: str, end: str, K: float, r: float, 
    n: int = 50, option_type: str = "call", T: float = 1.0,
):
    
    st.markdown("---")
    st.header("🔍 Calibration Status and Current Data")
    
    # 0. Calibration and Data Retrieval
    
    st.info(f"Attempting to calibrate u and d parameters for **{ticker}**...")
    
    try:
        u_opt, d_opt = optimize_u_d_from_ticker(ticker, start, end)
        st.success(f"Calibration successful. Optimal u = {u_opt:.6f}, d = {d_opt:.6f}")
    except ValueError as e:
        st.error(f"Calibration Error: {e}")
        return

    try:
        data = fetch_stock_history(ticker, start, end)
        last_price = float(data.iloc[-1])
        st.markdown(f"**Current Spot Price (S0) used:** **{last_price:.2f}**")
    except Exception:
        st.error("Failed to retrieve stock history.")
        return

    # 1. Implicit volatility and consistency calculation
    N_calibre = n 
    dt_calibre = T / N_calibre
    
    if u_opt <= 0 or d_opt <= 0:
        st.error("Cannot calculate implied volatility because u or d are non-positive.")
        return
    
    # Implied Sigma derived from calibration
    sigma_implied = np.log(u_opt / d_opt) / (2 * np.sqrt(dt_calibre))
    st.markdown(f"**Implied Model Volatility (σ) used for BS/MC:** `{sigma_implied:.4f}`")


    # 2. Main Calculation (CRR)
    result = price_option_euro_amer(
        S0=last_price, K=K, r=r, u=u_opt, d=d_opt, n=n, option_type=option_type, T=T
    )

    # 3. Black-Scholes Calculation
    bs_price = black_scholes_price(last_price, K, r, T, sigma_implied, option_type)

    # 4. Monte Carlo Calculation
    N_sims = 100000 
    N_steps = 252 
    mc_price = monte_carlo_option_price(
        S0=last_price, K=K, r=r, T=T, sigma=sigma_implied, N_simulations=N_sims, N_pas=N_steps, option_type=option_type
    )

    st.markdown("---")
    st.header("💰 Valuation and Hedging Results")
    
    # PRESENTATION 1: Price Comparison
    
    col_amer, col_euro, col_bs, col_mc = st.columns(4)
    
    col_amer.metric("American CRR Price", f"{result.price_amer:.4f}")
    col_euro.metric(f"European CRR Price (N={n})", f"{result.price_euro:.4f}")
    col_bs.metric("Black-Scholes Price", f"{bs_price:.4f}")
    col_mc.metric("Monte Carlo Price", f"{mc_price:.4f}", help=f"Based on {N_sims} simulations")
    
    # Initial Hedging t=0
    st.markdown("### Initial Hedging (t=0)")
    st.table(pd.DataFrame({
        'Parameter': ['American Delta', 'American Psi'],
        'Value': [f"{result.Delta_amer[0, 0]:.4f}", f"{result.Psi_amer[0, 0]:.4f}"]
    }))

    # PRESENTATION 2: Graphical Analysis
    
    # Expander 1: Greeks
    with st.expander("📈 Sensitivity Analysis (Greeks vs. Spot Price)", expanded=False):
        st.subheader(f"European Option Greeks Analysis")
        
        # Greeks Calculation 
        price_range = np.linspace(last_price * 0.6, last_price * 1.4, 100)
        greeks_list = []
        
        for S_val in price_range:
            greeks = calculate_greeks_at_t0(
                S0=S_val, K=K, r=r, u=u_opt, d=d_opt, n=n, option_type=option_type, T=T
            )
            greeks_list.append({
                        'S': S_val, 
                        'Price': greeks.Price, 
                        'Delta': greeks.Delta, 
                        'Gamma': greeks.Gamma,
                        'Vega': greeks.Vega,
                        'Theta': greeks.Theta,
                        'Rho': greeks.Rho
                    })
            
        greeks_df = pd.DataFrame(greeks_list)

        fig_greeks = plot_greeks_vs_price(greeks_df, option_type)
        st.plotly_chart(fig_greeks, use_container_width=True)

    # Expander 2: Hedging Strategy
    with st.expander("🗺️ Dynamic Hedging Strategy (Heatmaps)", expanded=False):
        st.subheader("Hedging Lattice Visualization (CRR)")
        fig_heatmap = plot_heatmap_matrices_plotly(result, n, option_type)
        st.plotly_chart(fig_heatmap, use_container_width=True)

    # Expander 3: Stability Analysis
    with st.expander("📊 Stability Analysis (CRR Price vs N)", expanded=False):
        st.subheader("Convergence Stability (based on Implied Volatility)")

        bs_price_ref = black_scholes_price(last_price, K, r, T, sigma_implied, option_type)
        st.markdown(f"**Black-Scholes Limit Price (Reference):** `{bs_price_ref:.4f}`")

        # Stability Loop (uses variable u and d for convergence)
        N_range = np.arange(10, 501, 10)
        stability_list = []

        for N_val in N_range:
            dt_stab = T / N_val
            
            u_stab = np.exp(sigma_implied * np.sqrt(dt_stab))
            d_stab = np.exp(-sigma_implied * np.sqrt(dt_stab))
            
            try:
                price_stab = price_option_euro_amer(
                    S0=last_price, K=K, r=r, u=u_stab, d=d_stab, n=N_val, option_type=option_type, T=T
                ).price_euro
                stability_list.append({'N': N_val, 'CRR_Price': price_stab})
            except ValueError:
                stability_list.append({'N': N_val, 'CRR_Price': np.nan})

        stability_df = pd.DataFrame(stability_list).dropna()

        if not stability_df.empty:
            fig_stability = plot_discretization_analysis(stability_df, bs_price_ref, option_type)
            st.plotly_chart(fig_stability, use_container_width=True)
        else:
            st.warning("Cannot plot stability. Check initial u and d parameters.")


    st.markdown("---")
    st.header("SABR Model Analysis")
    st.info("Ajust volatility parameters")

    col_sabr1, col_sabr2, col_sabr3, col_sabr4 = st.columns(4)
    with col_sabr1: 
        alpha = st.number_input("Alpha (Level)", 0.01, 2.0, 0.3, 0.05)
    with col_sabr2: 
        beta = st.number_input("Beta (Elasticity)", 0.0, 1.0, 0.5, 0.1)
    with col_sabr3: 
        rho = st.slider("Rho (Correlation)", -0.99, 0.99, -0.4, 0.05)
    with col_sabr4: 
        nu = st.number_input("Nu (Vol of Vol)", 0.0, 5.0, 0.4, 0.1)

    fig_sabr, price_sabr, vol_sabr = plot_sabr_analysis(
        last_price, K, r, T, option_type, alpha, beta, rho, nu
    )

    c_met1, c_met2 = st.columns(2)
    c_met1.metric(f"SABR Implied Vol (at K={K})", f"{vol_sabr:.2%}")
    c_met2.metric(f"SABR Price (at K={K})", f"{price_sabr:.4f}")

    st.plotly_chart(fig_sabr, use_container_width=True)

        st.markdown("---")
    st.header("Heston Model Analysis")
    st.info("Stochastic volatility (Heston) via Monte Carlo (Euler full truncation)")

    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns(5)
    with col_h1:
        v0 = st.number_input("v0 (initial variance)", 1e-6, 2.0, 0.04, 0.01)
    with col_h2:
        kappa = st.number_input("kappa (mean reversion)", 0.01, 20.0, 2.0, 0.5)
    with col_h3:
        theta = st.number_input("theta (long-run variance)", 1e-6, 2.0, 0.04, 0.01)
    with col_h4:
        xi = st.number_input("xi (vol of vol)", 0.001, 10.0, 0.5, 0.1)
    with col_h5:
        rho = st.slider("rho (corr)", -0.99, 0.99, -0.5, 0.05)

    h_steps = st.slider("Heston steps (Euler)", 50, 1000, 252, 50)
    h_sims = st.slider("Heston simulations", 10_000, 300_000, 50_000, 10_000)

    fig_heston, price_heston, vol_heston = plot_heston_analysis(
        last_price, K, r, T, option_type,
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        n_steps=h_steps, n_sims=h_sims, seed=42
    )

    c_hm1, c_hm2 = st.columns(2)
    c_hm1.metric(f"Heston Price (at K={K})", f"{price_heston:.4f}")
    if np.isnan(vol_heston):
        c_hm2.metric(f"BS Implied Vol (from Heston, at K={K})", "n/a")
    else:
        c_hm2.metric(f"BS Implied Vol (from Heston, at K={K})", f"{vol_heston:.2%}")

    st.plotly_chart(fig_heston, use_container_width=True)

# STREAMLIT CONFIGURATION AND MAIN
# REMPLACE TOUTE LA FONCTION main() PAR CELLE-CI :

def main():
    st.set_page_config(layout="wide", page_title="CRR Option Pricer")
    st.title("Binomial Option Pricer (CRR) & Hedging")

    # --- 1. INITIALISATION DE LA MÉMOIRE (SESSION STATE) ---
    if 'results_calculated' not in st.session_state:
        st.session_state.results_calculated = False

    # --- 2. SIDEBAR (PARAMÈTRES) ---
    with st.sidebar:
        st.header("Model Parameters")
        
        ticker = st.text_input("Stock Symbol (Ticker)", value="AAPL")
        option_type = st.selectbox("Option Type", options=['call', 'put'])
        K = st.number_input("Strike Price (K)", value=180.0, step=1.0)
        r = st.slider("Risk-Free Rate (r)", min_value=0.01, max_value=0.10, value=0.05, step=0.005)
        T = st.slider("Time to Maturity (years)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
        n = st.slider("Number of Steps (N) - Calibration/CRR", min_value=10, max_value=300, value=50, step=10)
        
        today = datetime.date.today()
        one_year_ago = today - datetime.timedelta(days=365)
        start_date_input = st.date_input("Start Date", value=one_year_ago, max_value=today)
        end_date_input = st.date_input("End Date", value=today, max_value=today)
        start_date_str = start_date_input.strftime("%Y-%m-%d")
        end_date_str = end_date_input.strftime("%Y-%m-%d")
        
    params = {
        'ticker': ticker, 'start': start_date_str, 'end': end_date_str, 
        'K': K, 'r': r, 'n': n, 
        'option_type': option_type, 'T': T
    }
    
    st.markdown("---")
    
    if st.button("CALCULATE PRICE AND ANALYSIS", type="primary"):
        st.session_state.results_calculated = True

    if st.session_state.results_calculated:
        if start_date_input >= end_date_input:
            st.error("The start date must be before the end date.")
        else:
            run_pricing_and_display(**params)

if __name__ == '__main__':
    main()