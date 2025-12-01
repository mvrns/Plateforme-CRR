# app.py
import datetime 
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
from src.model.option_pricer import price_option_euro_amer, ReplicatingResult
from src.model.objective_func import optimize_u_d_from_ticker
from src.data_loader.data_utils import fetch_stock_history 

# Fonction de traçage Heatmap
def plot_heatmap_matrices(result: ReplicatingResult, n: int, option_type: str):

    delta_data = result.Delta_amer
    psi_data = result.Psi_amer
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7), sharey=True)
    fig.suptitle(f'Couverture Dynamique ({option_type.capitalize()} Américain)', fontsize=16)

    # Heatmap de Delta
    sns.heatmap(
        delta_data.T, ax=axes[0], cmap="RdYlBu", 
        cbar_kws={'label': 'Delta (Nombre d\'actions)'}, 
        linewidths=.5, linecolor='lightgray', annot=False
    )
    axes[0].set_title(r'Matrice $\Delta$ (Sensibilité au Prix)', fontsize=14)
    axes[0].set_xlabel('Temps (i: de 0 à N-1)')
    axes[0].set_ylabel('Nombre de Mouvements Haussiers (j)')
    axes[0].invert_yaxis()

    # Heatmap de Psi
    sns.heatmap(
        psi_data.T, ax=axes[1], cmap="viridis", 
        cbar_kws={'label': r'$\Psi$ (Montant sans risque)'},
        linewidths=.5, linecolor='lightgray', annot=False
    )
    axes[1].set_title(r'Matrice $\Psi$ (Investissement sans Risque)', fontsize=14)
    axes[1].set_xlabel('Temps (i: de 0 à N-1)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    return fig 

# Interface 
def run_pricing_and_display(
    ticker: str, start: str, end: str, K: float, r: float,
    n: int = 50, option_type: str = "call", T: float = 1.0,
):
    st.info(f"Optimizing u and d for {ticker}...")
    
    try:
        u_opt, d_opt = optimize_u_d_from_ticker(ticker, start, end)
    except ValueError as e:
        st.error(f"Erreur de Calibration: {e}")
        return
        
    st.success(f"Optimal u = {u_opt:.6f}, d = {d_opt:.6f}")
    
    try:
        data = fetch_stock_history(ticker, start, end)
        last_price = float(data.iloc[-1])
        st.write(f"Prix actuel de l'action (S0) = **{last_price:.2f}**")
    except Exception:
        st.error("Impossible de récupérer l'historique de l'action.")
        return

    result = price_option_euro_amer(
        S0=last_price, K=K, r=r, u=u_opt, d=d_opt, n=n, option_type=option_type, T=T
    )

    st.header(" Résultats de Valorisation")
    st.markdown(f"""
    * **Prix Américain:** `{result.price_amer:.4f}`
    * **Prix Européen:** `{result.price_euro:.4f}`
    """)
    
    st.header(" Couverture Initiale (t=0)")
    st.table(pd.DataFrame({
        'Paramètre': ['Delta Américain', 'Psi Américain'],
        'Valeur': [f"{result.Delta_amer[0, 0]:.4f}", f"{result.Psi_amer[0, 0]:.4f}"]
    }))

    st.header(" Stratégie de Couverture Dynamique (Heatmaps)")
    fig = plot_heatmap_matrices(result, n, option_type)
    st.pyplot(fig)


# Streamlit
def main():
    
    st.set_page_config(layout="wide", page_title="CRR Option Pricer")
    st.title("Binomial Option Pricer (CRR) & Hedging")

    st.sidebar.header("Paramètres du Modèle")
    
    # --- 1. Paramètres de l'Option ---
    st.sidebar.subheader("Option et Taux")
    ticker = st.sidebar.text_input("Symbole Boursier (Ticker)", value="AAPL")
    option_type = st.sidebar.selectbox("Type d'Option", options=['call', 'put'])
    K = st.sidebar.number_input("Prix d'Exercice (K)", value=180.0, step=1.0)
    r = st.sidebar.slider("Taux sans Risque (r)", min_value=0.01, max_value=0.10, value=0.05, step=0.005)
    
    # --- 2. Paramètres du Modèle ---
    st.sidebar.subheader("Horizon et Discrétisation")
    T = st.sidebar.slider("Temps à Maturité (années)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    n = st.sidebar.slider("Nombre d'Étapes (N)", min_value=10, max_value=200, value=50, step=10)
    
    # --- 3. Paramètres de Calibration ---
    st.sidebar.subheader("Calibration (Historique)")
    
    # Dates par défaut
    today = datetime.date.today()
    one_year_ago = today - datetime.timedelta(days=365)
    
    # Widgets d'entrée de date
    start_date_input = st.sidebar.date_input(
        "Date de Début (Calibration)", 
        value=one_year_ago,
        max_value=today
    )
    end_date_input = st.sidebar.date_input(
        "Date de Fin (Calibration)", 
        value=today,
        max_value=today
    )
    
    # Convertir les objets date en chaînes de caractères "YYYY-MM-DD"
    start_date_str = start_date_input.strftime("%Y-%m-%d")
    end_date_str = end_date_input.strftime("%Y-%m-%d")


    if st.sidebar.button("Calculer le Prix"):
        if start_date_input >= end_date_input:
            st.sidebar.error("La date de début doit être antérieure à la date de fin.")
        else:
            run_pricing_and_display(
                ticker=ticker, 
                start=start_date_str, 
                end=end_date_str,     
                K=K, r=r, n=n, option_type=option_type, T=T
            )

if __name__ == '__main__':
    main()