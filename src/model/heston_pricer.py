#src/model/heston_pricer.py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from .bs_pricer import black_scholes_price

#Heston path simulation (Euler full truncation)
def simulate_heston_paths(
    S0: float,
    r: float,
    T: float,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_steps: int,
    n_sims: int,
    seed: int | None = None,
):
    """
    Simulates Heston dynamics under Q:
        dS_t = r S_t dt + sqrt(v_t) S_t dW1_t
        dv_t = kappa (theta - v_t) dt + xi sqrt(v_t) dW2_t
        corr(dW1, dW2) = rho

    Discretization: Euler with 'full truncation' for v to keep it non-negative in practice.
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps

    S = np.full(n_sims, S0, dtype=float)
    v = np.full(n_sims, v0, dtype=float)

    sqrt_dt = np.sqrt(dt)

    for _ in range(n_steps):
        #correlated normals
        Z1 = rng.standard_normal(n_sims)
        Z2 = rng.standard_normal(n_sims)
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * (rho * Z1 + np.sqrt(max(1 - rho**2, 0.0)) * Z2)

        v_pos = np.maximum(v, 0.0)
        #v update
        v = v + kappa * (theta - v_pos) * dt + xi * np.sqrt(v_pos) * dW2
        v = np.maximum(v, 0.0)  # full truncation

        #S update
        S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * dW1)

    return S


#Heston MC pricer
def heston_option_price_mc(
    S0: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_steps: int = 252,
    n_sims: int = 100_000,
    seed: int | None = 42,
):
    """
    Prices a European option under Heston via Monte Carlo.
    """
    ST = simulate_heston_paths(
        S0=S0, r=r, T=T,
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        n_steps=n_steps, n_sims=n_sims, seed=seed
    )

    if option_type.lower() == "call":
        payoffs = np.maximum(ST - K, 0.0)
    elif option_type.lower() == "put":
        payoffs = np.maximum(K - ST, 0.0)
    else:
        raise ValueError("option_type must be 'call' or 'put'.")

    price = np.exp(-r * T) * np.mean(payoffs)
    return float(price)

#visualization (smile-like + price curve)
def plot_heston_analysis(
    S0: float,
    K_target: float,
    r: float,
    T: float,
    option_type: str,
    v0: float,
    kappa: float,
    theta: float,
    xi: float,
    rho: float,
    n_steps: int = 252,
    n_sims: int = 50_000,
    seed: int | None = 42,
):
    """
    Produces a simple "shape" analysis:
    - Heston MC prices vs strike
    - Convert those prices into a BS-implied volatility curve (numerical inversion)
    """
    strikes = np.linspace(S0 * 0.5, S0 * 1.5, 25)
    prices = []
    impvols = []

    #helper: implied vol by bisection using your BS pricer
    def implied_vol_bisect(target_price: float, K: float):
        lo, hi = 1e-6, 5.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            p_mid = black_scholes_price(S0, K, r, T, mid, option_type)
            if p_mid > target_price:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    for K in strikes:
        p = heston_option_price_mc(
            S0=S0, K=K, r=r, T=T, option_type=option_type,
            v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
            n_steps=n_steps, n_sims=n_sims, seed=seed
        )
        prices.append(p)

        #implied vol is optional; can fail for deep ITM/OTM numerically, so guard
        try:
            iv = implied_vol_bisect(p, K)
        except Exception:
            iv = np.nan
        impvols.append(iv)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=strikes, y=impvols, name="BS Implied Vol (from Heston prices)"),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=strikes, y=prices, name=f"Heston MC Price ({option_type})", line=dict(dash="dot")),
        secondary_y=True
    )

    fig.add_vline(x=K_target, line_dash="dash", line_color="green", annotation_text="Target Strike")

    fig.update_layout(
        title="Heston (MC): Price curve + implied-vol shape",
        template="plotly_white",
        hovermode="x unified"
    )
    fig.update_xaxes(title_text="Strike (K)")
    fig.update_yaxes(title_text="Implied Vol", secondary_y=False)
    fig.update_yaxes(title_text="Option Price", secondary_y=True)

    #also compute the target K price + implied vol at target
    price_target = heston_option_price_mc(
        S0=S0, K=K_target, r=r, T=T, option_type=option_type,
        v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho,
        n_steps=n_steps, n_sims=n_sims, seed=seed
    )
    try:
        vol_target = implied_vol_bisect(price_target, K_target)
    except Exception:
        vol_target = np.nan

    return fig, float(price_target), float(vol_target)
