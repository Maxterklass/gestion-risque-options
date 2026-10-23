"""
Moteur de valorisation Black-Scholes, de calcul des Grecques et de simulations Monte Carlo.
Vectorisé avec NumPy (accepte aussi bien des scalaires que des tableaux NumPy).
Compatible Python 3.14+.
"""

import numpy as np
from scipy.stats import norm


def calc_d1(S, K: float, T: float, r: float, sigma: float):
    """
    Calcule le terme d1 du modèle de Black-Scholes (compatible scalaire et vecteur NumPy).
    Formule : d1 = [ln(S0 / K) + (r + sigma^2 / 2) * T] / (sigma * sqrt(T))
    """
    T_safe = np.maximum(T, 1e-8)
    sigma_safe = np.maximum(sigma, 1e-8)
    return (np.log(S / K) + (r + 0.5 * sigma_safe ** 2) * T_safe) / (sigma_safe * np.sqrt(T_safe))


def calc_d2(S, K: float, T: float, r: float, sigma: float):
    """
    Calcule le terme d2 du modèle de Black-Scholes.
    Formule : d2 = d1 - sigma * sqrt(T)
    """
    T_safe = np.maximum(T, 1e-8)
    return calc_d1(S, K, T_safe, r, sigma) - sigma * np.sqrt(T_safe)


def bs_price(S, K: float, T: float, r: float, sigma: float, opt_type: str):
    """
    Calcule le prix théorique d'une option européenne (Call ou Put).
    Formules :
        Call : C = S0 * Phi(d1) - K * exp(-r*T) * Phi(d2)
        Put  : P = K * exp(-r*T) * Phi(-d2) - S0 * Phi(-d1)
    """
    d1_val = calc_d1(S, K, T, r, sigma)
    d2_val = calc_d2(S, K, T, r, sigma)

    if opt_type.lower() == "call":
        return S * norm.cdf(d1_val) - K * np.exp(-r * T) * norm.cdf(d2_val)
    elif opt_type.lower() == "put":
        return K * np.exp(-r * T) * norm.cdf(-d2_val) - S * norm.cdf(-d1_val)
    else:
        raise ValueError("Le type d'option doit être 'call' ou 'put'.")


def bs_delta(S, K: float, T: float, r: float, sigma: float, opt_type: str):
    """
    Calcule le Delta unitaire d'une option européenne.
    Formules :
        Delta_call = Phi(d1)
        Delta_put  = Phi(d1) - 1
    """
    d1_val = calc_d1(S, K, T, r, sigma)
    if opt_type.lower() == "call":
        return norm.cdf(d1_val)
    else:
        return norm.cdf(d1_val) - 1.0


def bs_gamma(S, K: float, T: float, r: float, sigma: float):
    """
    Calcule le Gamma unitaire d'une option européenne (identique pour Call et Put).
    Formule : Gamma = phi(d1) / (S0 * sigma * sqrt(T))
    """
    T_safe = np.maximum(T, 1e-8)
    d1_val = calc_d1(S, K, T_safe, r, sigma)
    return norm.pdf(d1_val) / (S * sigma * np.sqrt(T_safe))


def bs_vega(S, K: float, T: float, r: float, sigma: float):
    """
    Calcule le Vega unitaire (pour une variation de +1% de volatilité, soit +0.01).
    """
    T_safe = np.maximum(T, 1e-8)
    d1_val = calc_d1(S, K, T_safe, r, sigma)
    return S * norm.pdf(d1_val) * np.sqrt(T_safe) / 100.0


def bs_theta(S, K: float, T: float, r: float, sigma: float, opt_type: str):
    """
    Calcule le Theta unitaire journalier (perte de valeur sur 1 jour calendaire).
    """
    T_safe = np.maximum(T, 1e-8)
    d1_val = calc_d1(S, K, T_safe, r, sigma)
    d2_val = calc_d2(S, K, T_safe, r, sigma)
    term1 = -(S * norm.pdf(d1_val) * sigma) / (2.0 * np.sqrt(T_safe))

    if opt_type.lower() == "call":
        theta_annual = term1 - r * K * np.exp(-r * T_safe) * norm.cdf(d2_val)
    else:
        theta_annual = term1 + r * K * np.exp(-r * T_safe) * norm.cdf(-d2_val)

    return theta_annual / 365.0


def bs_rho(S, K: float, T: float, r: float, sigma: float, opt_type: str):
    """
    Calcule le Rho unitaire (pour une variation de +1% du taux sans risque).
    """
    d2_val = calc_d2(S, K, T, r, sigma)
    if opt_type.lower() == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2_val) / 100.0
    else:
        return -K * T * np.exp(-r * T) * norm.cdf(-d2_val) / 100.0


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, opt_type: str) -> dict:
    """
    Fonction regroupant l'ensemble des Grecques unitaires d'une option.
    """
    return {
        "delta": float(bs_delta(S, K, T, r, sigma, opt_type)),
        "gamma": float(bs_gamma(S, K, T, r, sigma)),
        "vega": float(bs_vega(S, K, T, r, sigma)),
        "theta": float(bs_theta(S, K, T, r, sigma, opt_type)),
        "rho": float(bs_rho(S, K, T, r, sigma, opt_type)),
    }


def check_put_call_parity(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """
    Vérifie numériquement la relation de parité Call-Put européenne :
    C - P = S0 - K * exp(-r * T)
    """
    call_val = float(bs_price(S, K, T, r, sigma, "call"))
    put_val = float(bs_price(S, K, T, r, sigma, "put"))
    lhs = call_val - put_val
    rhs = float(S - K * np.exp(-r * T))
    diff = abs(lhs - rhs)

    return {
        "is_valid": bool(np.isclose(lhs, rhs, atol=1e-6)),
        "call": call_val,
        "put": put_val,
        "lhs": lhs,
        "rhs": rhs,
        "diff": diff,
    }


def implied_volatility(target_price: float, S: float, K: float, T: float, r: float, opt_type: str) -> float:
    """
    Calcule la volatilité implicite par la méthode de Newton-Raphson.
    """
    sigma = 0.25
    for _ in range(100):
        price = float(bs_price(S, K, T, r, sigma, opt_type))
        vega_raw = float(bs_vega(S, K, T, r, sigma)) * 100.0
        diff = target_price - price
        if abs(diff) < 1e-6:
            return float(sigma)
        if vega_raw < 1e-8:
            return float("nan")
        sigma += diff / vega_raw
        if sigma <= 0:
            sigma = 0.001
    return float("nan")


# =========================================================
# MOTEUR MONTE CARLO (VECTORISÉ NUMPY)
# =========================================================

def mc_portfolio_pricing(S0: float, T: float, r: float, sigma: float, portfolio_df, n_sims: int = 20000, seed: int = 42) -> dict:
    """
    Valorise le portefeuille d'options par Monte Carlo avec variables antithétiques (Z et -Z).
    Retourne le prix estimé, l'erreur standard et l'intervalle de confiance à 95%.
    """
    rng = np.random.default_rng(seed)
    half_n = max(1, n_sims // 2)
    Z = rng.standard_normal(half_n)

    drift = (r - 0.5 * sigma ** 2) * T
    diffusion = sigma * np.sqrt(T) * Z

    ST_pos = S0 * np.exp(drift + diffusion)
    ST_neg = S0 * np.exp(drift - diffusion)

    payoff_pos = np.zeros(half_n)
    payoff_neg = np.zeros(half_n)

    for _, row in portfolio_df.iterrows():
        pos_label = str(row["Type de position"])
        qty = float(row["Quantité"])
        K = float(row["Strike K (€)"])
        signe = 1.0 if "Long" in pos_label else -1.0

        if "Call" in pos_label:
            payoff_pos += signe * qty * np.maximum(ST_pos - K, 0.0)
            payoff_neg += signe * qty * np.maximum(ST_neg - K, 0.0)
        else:
            payoff_pos += signe * qty * np.maximum(K - ST_pos, 0.0)
            payoff_neg += signe * qty * np.maximum(K - ST_neg, 0.0)

    # Moyenne par paire antithétique actualisée au taux sans risque
    disc_pair_payoffs = np.exp(-r * T) * 0.5 * (payoff_pos + payoff_neg)
    mc_price = float(np.mean(disc_pair_payoffs))
    std_err = float(np.std(disc_pair_payoffs, ddof=1) / np.sqrt(half_n))

    return {
        "mc_price": mc_price,
        "std_err": std_err,
        "ci_low": mc_price - 1.96 * std_err,
        "ci_high": mc_price + 1.96 * std_err,
    }


def mc_var_cvar_full_reval(
    S0: float,
    T_init: float,
    r: float,
    sigma: float,
    portfolio_df,
    val_0: float,
    hedge_0: float,
    horizon_days: int = 1,
    n_sims: int = 20000,
    alpha: float = 0.99,
    seed: int = 42,
) -> dict:
    """
    Calcule la VaR et l'Expected Shortfall (CVaR) par réévaluation complète (Full Revaluation)
    du portefeuille à l'horizon choisi, avec et sans couverture Delta.
    """
    rng = np.random.default_rng(seed)
    dt = horizon_days / 252.0
    T_new = max(1e-6, T_init - (horizon_days / 365.0))

    Z = rng.standard_normal(n_sims)
    S_horizon = S0 * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    val_horizon = np.zeros(n_sims)
    for _, row in portfolio_df.iterrows():
        pos_label = str(row["Type de position"])
        qty = float(row["Quantité"])
        K = float(row["Strike K (€)"])
        signe = 1.0 if "Long" in pos_label else -1.0
        opt_type = "call" if "Call" in pos_label else "put"

        val_horizon += signe * qty * bs_price(S_horizon, K, T_new, r, sigma, opt_type)

    pnl_unhedged = val_horizon - val_0
    pnl_hedged = pnl_unhedged + hedge_0 * (S_horizon - S0)

    # Quantile de perte au seuil (1 - alpha)
    q_unhedged = float(np.percentile(pnl_unhedged, (1.0 - alpha) * 100.0))
    var_unhedged = -q_unhedged
    tail_u = pnl_unhedged[pnl_unhedged <= q_unhedged]
    cvar_unhedged = -float(np.mean(tail_u)) if len(tail_u) > 0 else var_unhedged

    q_hedged = float(np.percentile(pnl_hedged, (1.0 - alpha) * 100.0))
    var_hedged = -q_hedged
    tail_h = pnl_hedged[pnl_hedged <= q_hedged]
    cvar_hedged = -float(np.mean(tail_h)) if len(tail_h) > 0 else var_hedged

    return {
        "pnl_unhedged": pnl_unhedged,
        "pnl_hedged": pnl_hedged,
        "var_unhedged": var_unhedged,
        "cvar_unhedged": cvar_unhedged,
        "var_hedged": var_hedged,
        "cvar_hedged": cvar_hedged,
    }


def mc_discrete_delta_hedging(
    S0: float,
    T: float,
    r: float,
    sigma: float,
    portfolio_df,
    val_0: float,
    hedge_0: float,
    n_steps: int = 52,
    n_paths: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Simule la couverture dynamique (Delta-Hedging) en temps discret de t=0 à t=T
    sur `n_paths` trajectoires avec `n_steps` dates de rebalancement.
    Retourne les trajectoires simulées et la distribution de l'erreur finale de couverture (Tracking Error).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    time_grid = np.linspace(0.0, T, n_steps + 1)

    # Génération matricielle des trajectoires : dimension (n_paths, n_steps + 1)
    Z = rng.standard_normal((n_paths, n_steps))
    log_increments = (r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    S_paths = np.zeros((n_paths, n_steps + 1))
    S_paths[:, 0] = S0
    S_paths[:, 1:] = S0 * np.exp(np.cumsum(log_increments, axis=1))

    # Initialisation à t=0 du portefeuille autofinancé de couverture
    # On détient le portefeuille d'options (valeur val_0) + hedge_0 actions (-Delta_0).
    # Le compte cash B_0 finance la position initiale pour que la mise nette initiale soit nulle :
    shares_held = np.full(n_paths, hedge_0, dtype=float)
    cash_account = np.full(n_paths, -val_0 - hedge_0 * S0, dtype=float)

    # Rebalancement dynamique aux étapes k = 1, ..., n_steps - 1
    for step in range(1, n_steps):
        tau = T - time_grid[step]
        S_t = S_paths[:, step]

        # Capitalisation des intérêts sur le cash entre t_{k-1} et t_k
        cash_account *= np.exp(r * dt)

        # Calcul vectorisé du nouveau Delta du portefeuille sur les n_paths trajectoires
        delta_t = np.zeros(n_paths)
        for _, row in portfolio_df.iterrows():
            pos_label = str(row["Type de position"])
            qty = float(row["Quantité"])
            K = float(row["Strike K (€)"])
            signe = 1.0 if "Long" in pos_label else -1.0
            opt_type = "call" if "Call" in pos_label else "put"

            delta_t += signe * qty * bs_delta(S_t, K, tau, r, sigma, opt_type)

        new_shares = -delta_t
        shares_traded = new_shares - shares_held

        # Achat / vente des actions d'ajustement financé par le compte cash
        cash_account -= shares_traded * S_t
        shares_held = new_shares

    # Débouclage final à l'échéance T (step = n_steps)
    cash_account *= np.exp(r * dt)
    S_T = S_paths[:, -1]

    # Payoff final du portefeuille d'options à maturité T
    payoff_T = np.zeros(n_paths)
    for _, row in portfolio_df.iterrows():
        pos_label = str(row["Type de position"])
        qty = float(row["Quantité"])
        K = float(row["Strike K (€)"])
        signe = 1.0 if "Long" in pos_label else -1.0
        if "Call" in pos_label:
            payoff_T += signe * qty * np.maximum(S_T - K, 0.0)
        else:
            payoff_T += signe * qty * np.maximum(K - S_T, 0.0)

    # Erreur finale de couverture = Payoff_T + Liquidation des actions détenues + Compte Cash final
    hedging_error = payoff_T + shares_held * S_T + cash_account

    return {
        "time_grid": time_grid,
        "S_paths": S_paths,
        "hedging_error": hedging_error,
        "mean_error": float(np.mean(hedging_error)),
        "std_error": float(np.std(hedging_error, ddof=1)),
        "q05_error": float(np.percentile(hedging_error, 5)),
        "q95_error": float(np.percentile(hedging_error, 95)),
    }