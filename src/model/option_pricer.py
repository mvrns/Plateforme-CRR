# src/model/option_pricer.py

import math
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Data class for results
@dataclass
class ReplicatingResult:
    price_amer: float
    price_euro: float
    V_amer: np.ndarray
    V_euro: np.ndarray
    S: np.ndarray
    Delta_amer: np.ndarray
    Psi_amer: np.ndarray
    Delta_euro: np.ndarray
    Psi_euro: np.ndarray

def price_option_euro_amer(
    S0: float, K: float, r: float, u: float, d: float, n: int,
    option_type: str = "call", T: float = 1.0,
) -> ReplicatingResult:
    """Prices American and European options using the Binomial (CRR) model"""
    
    dt = T / n
    # 1. Risk-neutral probability
    p = (math.exp(r * dt) - d) / (u - d)
    
    if not (0 <= p <= 1):
        raise ValueError("Risk-neutral probability p is not between 0 and 1. Check u, d, r, and dt.")

    # 2. Stock price tree
    S = np.zeros((n + 1, n + 1))
    for i in range(n + 1):
        for j in range(i + 1):
            S[i, j] = S0 * (u**j) * (d**(i - j))

    # 3. Value and coverage matrices
    V_amer = np.zeros((n + 1, n + 1))
    V_euro = np.zeros((n + 1, n + 1))
    Delta_amer = np.zeros((n, n)) 
    Psi_amer = np.zeros((n, n))   
    Delta_euro = np.zeros((n, n)) 
    Psi_euro = np.zeros((n, n)) 

    # 4. Payoff at maturity (i = n)
    for j in range(n + 1):
        payoff = max(0.0, S[n, j] - K) if option_type.lower() == "call" else max(0.0, K - S[n, j])
        V_amer[n, j] = payoff
        V_euro[n, j] = payoff

    discount = math.exp(-r * dt)

    # 5. Backward induction
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            cont_amer = discount * (p * V_amer[i + 1, j + 1] + (1 - p) * V_amer[i + 1, j])
            cont_euro = discount * (p * V_euro[i + 1, j + 1] + (1 - p) * V_euro[i + 1, j])

            exercise_value = max(0.0, S[i, j] - K) if option_type.lower() == "call" else max(0.0, K - S[i, j])

            V_amer[i, j] = max(exercise_value, cont_amer)
            V_euro[i, j] = cont_euro

            denom = S[i + 1, j + 1] - S[i + 1, j]
            
            # 5a. American hedging
            if denom != 0:
                Delta_amer[i, j] = (V_amer[i + 1, j + 1] - V_amer[i + 1, j]) / denom
            else:
                Delta_amer[i, j] = 0.0
            Psi_amer[i, j] = discount * (V_amer[i + 1, j] - Delta_amer[i, j] * S[i + 1, j])

            # 5b. European hedging
            if denom != 0:
                Delta_euro[i, j] = (V_euro[i + 1, j + 1] - V_euro[i + 1, j]) / denom
            else:
                Delta_euro[i, j] = 0.0
            Psi_euro[i, j] = discount * (V_euro[i + 1, j] - Delta_euro[i, j] * S[i + 1, j])


    return ReplicatingResult(
        price_amer=float(V_amer[0, 0]),
        price_euro=float(V_euro[0, 0]),
        V_amer=V_amer,
        V_euro=V_euro,
        S=S,
        Delta_amer=Delta_amer,
        Psi_amer=Psi_amer,
        Delta_euro=Delta_euro,
        Psi_euro=Psi_euro
    )