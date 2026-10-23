"""
Application web Streamlit : Outil de gestion du risque d'une position d'options.
Respecte l'ensemble des consignes du projet M2 IREF (base + bonus + Monte Carlo).
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf

from pricing_engine import (
    bs_price,
    bs_greeks,
    check_put_call_parity,
    implied_volatility,
    mc_portfolio_pricing,
    mc_var_cvar_full_reval,
    mc_discrete_delta_hedging,
)

# ==========================================
# FONCTIONS UTILITAIRES & FORMATAGE EN CLAIR
# ==========================================

def format_hedge_order(shares_to_trade: float) -> str:
    """
    Traduit un nombre d'actions en une instruction de trading explicite en français.
    Exemple : -60 -> 'Vendre 60.00 actions'
    """
    if abs(shares_to_trade) < 1e-4:
        return "Aucune action à traiter (0.00 action)"
    elif shares_to_trade > 0:
        return f"Acheter {abs(shares_to_trade):.2f} actions"
    else:
        return f"Vendre {abs(shares_to_trade):.2f} actions"


def compute_portfolio(S: float, T: float, r: float, sigma: float, df: pd.DataFrame):
    """
    Calcule les métriques ligne par ligne ainsi que les valeurs agrégées du portefeuille.
    Le signe (+1 pour Long / Achat, -1 pour Short / Vente) s'applique au prix et à toutes les Grecques.
    """
    rows_details = []
    tot_val = tot_delta = tot_gamma = tot_vega = tot_theta = tot_rho = 0.0

    for idx, row in df.iterrows():
        pos_label = str(row["Type de position"])
        qty = float(row["Quantité"])
        K = float(row["Strike K (€)"])

        is_long = "Long" in pos_label
        opt_type = "call" if "Call" in pos_label else "put"
        signe = 1.0 if is_long else -1.0

        u_price = float(bs_price(S, K, T, r, sigma, opt_type))
        g = bs_greeks(S, K, T, r, sigma, opt_type)

        pos_val = signe * qty * u_price
        pos_delta = signe * qty * g["delta"]
        pos_gamma = signe * qty * g["gamma"]
        pos_vega = signe * qty * g["vega"]
        pos_theta = signe * qty * g["theta"]
        pos_rho = signe * qty * g["rho"]

        rows_details.append({
            "Ligne": f"#{idx + 1} - {pos_label} (K={K:.1f}€)",
            "Sens": "Achat (+1)" if is_long else "Vente (-1)",
            "Quantité": qty,
            "Prix Unitaire (€)": u_price,
            "Valeur Position (€)": pos_val,
            "Delta Unitaire": g["delta"],
            "Delta Position": pos_delta,
            "Couverture Ligne": format_hedge_order(-pos_delta),
            "Gamma Position": pos_gamma,
            "Vega Position": pos_vega,
            "Theta Position (€/j)": pos_theta,
        })

        tot_val += pos_val
        tot_delta += pos_delta
        tot_gamma += pos_gamma
        tot_vega += pos_vega
        tot_theta += pos_theta
        tot_rho += pos_rho

    summary = {
        "value": tot_val,
        "delta": tot_delta,
        "hedge": -tot_delta,
        "gamma": tot_gamma,
        "vega": tot_vega,
        "theta": tot_theta,
        "rho": tot_rho,
        "unit_price_first": rows_details[0]["Prix Unitaire (€)"] if rows_details else 0.0,
    }
    return pd.DataFrame(rows_details), summary


# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================

st.set_page_config(page_title="Desk Options - Gestion du Risque", layout="wide")

st.title("📈 Outil de Gestion du Risque d'une Position d'Options")
st.markdown(
    "Cet outil permet à un opérateur de marché ou à un analyste risque de **calculer instantanément "
    "la couverture en actions (Delta-Hedging)** d'une option ou d'un portefeuille d'options, "
    "de **suivre ses réajustements quand le marché bouge**, d'**identifier les zones d'instabilité (Gamma)**, "
    "et de **simuler par Monte Carlo le risque de couverture en temps discret**."
)

# --- Volet pédagogique pour comprendre les formules et les concepts ---
with st.expander("📖 Guide Pédagogique & Formules Mathématiques du Modèle (Cliquer pour dérouler)"):
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.markdown("#### 1. Formules de Black-Scholes")
        st.latex(r"C = S_0 \, \Phi(d_1) - K e^{-rT}\, \Phi(d_2), \qquad P = K e^{-rT}\, \Phi(-d_2) - S_0\, \Phi(-d_1)")
        st.latex(r"d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}")
        st.markdown("**Relation de Parité Call-Put (Vérification du moteur) :**")
        st.latex(r"C - P = S_0 - K e^{-rT}")
    with col_f2:
        st.markdown("#### 2. Les Indicateurs de Risque (Grecques)")
        st.latex(r"\Delta_{\text{call}} = \Phi(d_1), \qquad \Delta_{\text{put}} = \Phi(d_1) - 1")
        st.latex(r"\Gamma = \frac{\varphi(d_1)}{S_0 \, \sigma \sqrt{T}} \quad \text{(identique Call/Put unitaire)}")
        st.markdown(
            "- **Delta ($\Delta$)** : Variation du prix de la position quand l'action bouge de +1 €. "
            "Pour couvrir un Delta de $+60$, il faut **vendre 60 actions** (couverture $= -\Delta$).\n"
            "- **Gamma ($\Gamma$)** : Vitesse à laquelle le Delta bouge quand l'action varie. "
            "Plus le $|\Gamma|$ est élevé, plus la couverture devient **instable** et doit être réajustée fréquemment.\n"
            "- **Signe du Gamma** : Toujours **positif ($+$)** pour un achat d'option (Long) et **négatif ($-$)** pour une vente d'option (Short)."
        )

st.divider()

# ==========================================
# BARRE LATÉRALE : SAISIE DES PARAMÈTRES
# ==========================================

st.sidebar.header("⚙️ 1. Mode & Marché")

mode_outil = st.sidebar.radio(
    "Mode d'analyse",
    ["Position Unique (Standard)", "Portefeuille Multi-Options (Bonus)"],
    help="Choisissez 'Position Unique' pour analyser une seule option selon le cas de base, ou 'Portefeuille' pour agréger plusieurs options."
)

source_spot = st.sidebar.radio("Source du cours actuel (S0)", ["Saisie Manuelle", "Temps Réel (Yahoo Finance)"])

if source_spot == "Saisie Manuelle":
    S0 = st.sidebar.number_input("Cours actuel du sous-jacent S0 (€)", value=100.0, step=1.0)
else:
    ticker = st.sidebar.text_input("Ticker Yahoo Finance (ex: ESE.PA, BNP.PA, AAPL)", value="ESE.PA")
    try:
        hist = yf.Ticker(ticker).history(period="1y")
        if not hist.empty:
            live_s0 = float(hist["Close"].iloc[-1])
            log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
            hist_vol = float(log_returns.std() * np.sqrt(252) * 100.0)
            st.sidebar.success(f"Cours en direct : {live_s0:.2f} € | Vol. Hist. 1 an : {hist_vol:.1f} %")
            S0 = st.sidebar.number_input("Cours retenu S0 (€)", value=round(live_s0, 2), step=0.5)
        else:
            st.sidebar.error("Ticker introuvable ou aucune donnée disponible.")
            st.stop()
    except Exception:
        st.sidebar.error("Impossible de joindre Yahoo Finance. Vérifiez votre connexion ou le ticker.")
        st.stop()

T_init = st.sidebar.number_input("Maturité restante T (années)", value=0.5, step=0.05)
r_pct = st.sidebar.number_input("Taux sans risque annuel r (%)", value=3.0, step=0.1)

# Option de calcul de volatilité implicite
use_iv = st.sidebar.checkbox("Calculer σ via un prix d'option coté (Volatilité Implicite)")
if use_iv:
    mkt_price = st.sidebar.number_input("Prix de marché d'un Call ATM (€)", value=6.5, step=0.5)
    if S0 is not None and T_init is not None and S0 > 0 and T_init > 0 and mkt_price > 0:
        iv_val = implied_volatility(mkt_price, S0, S0, T_init, (r_pct or 0.0) / 100.0, "call")
        if np.isnan(iv_val):
            st.sidebar.error("Impossible de converger vers une volatilité implicite valide.")
            st.stop()
        sigma_pct = iv_val * 100.0
        st.sidebar.info(f"Volatilité implicite déduite : {sigma_pct:.2f} %")
    else:
        sigma_pct = 20.0
else:
    sigma_pct = st.sidebar.number_input("Volatilité annuelle σ (%)", value=20.0, step=1.0)

# --- Saisie de la position selon le mode choisi ---
if mode_outil == "Position Unique (Standard)":
    st.sidebar.markdown("---")
    st.sidebar.header("🎯 2. Caractéristiques de l'Option")
    pos_type_single = st.sidebar.selectbox(
        "Type de position",
        ["Long Call", "Short Call", "Long Put", "Short Put"]
    )
    qty_single = st.sidebar.number_input("Quantité d'options", value=100.0, step=10.0)
    K_single = st.sidebar.number_input("Strike K (€)", value=100.0, step=1.0)

    # Validation stricte des saisies (refus avec message clair sans planter)
    erreurs = []
    if S0 is None or S0 <= 0:
        erreurs.append("Le cours du sous-jacent (S) doit être strictement positif (> 0).")
    if K_single is None or K_single <= 0:
        erreurs.append("Le strike (K) doit être strictement positif (> 0).")
    if T_init is None or T_init <= 0:
        erreurs.append("La maturité restante (T) ne peut pas être nulle, négative ou vide.")
    if sigma_pct is None or sigma_pct <= 0:
        erreurs.append("La volatilité annuelle (σ) ne peut pas être nulle, négative ou vide.")
    if qty_single is None or qty_single <= 0:
        erreurs.append("La quantité d'options doit être strictement positive (non nulle et non vide).")
    if r_pct is None:
        erreurs.append("Le taux sans risque (r) ne peut pas être vide.")

    if erreurs:
        st.error("🚫 **Saisie invalide détectée — Calcul interrompu :**\n\n" + "\n".join([f"- {e}" for e in erreurs]))
        st.stop()

    portfolio_df = pd.DataFrame({
        "Type de position": [pos_type_single],
        "Quantité": [float(qty_single)],
        "Strike K (€)": [float(K_single)],
    })

else:
    erreurs = []
    if S0 is None or S0 <= 0:
        erreurs.append("Le cours du sous-jacent (S) doit être strictement positif (> 0).")
    if T_init is None or T_init <= 0:
        erreurs.append("La maturité restante (T) ne peut pas être nulle, négative ou vide.")
    if sigma_pct is None or sigma_pct <= 0:
        erreurs.append("La volatilité annuelle (σ) ne peut pas être nulle, négative ou vide.")
    if r_pct is None:
        erreurs.append("Le taux sans risque (r) ne peut pas être vide.")

    if erreurs:
        st.error("🚫 **Saisie invalide détectée — Calcul interrompu :**\n\n" + "\n".join([f"- {e}" for e in erreurs]))
        st.stop()

    st.subheader("🧰 Constructeur de Portefeuille Multi-Options (Bonus)")
    col_preset, col_info = st.columns([1, 2])
    with col_preset:
        strategie_preset = st.selectbox(
            "Charger une stratégie type :",
            [
                "Personnalisé / Call Spread (Défaut)",
                "Long Straddle (Achat Call + Achat Put ATM)",
                "Short Strangle (Vente Put OTM + Vente Call OTM)",
                "Butterfly Spread (1 Long K1, 2 Short K2, 1 Long K3)",
            ]
        )
    with col_info:
        st.info(
            "💡 Vous pouvez modifier les cellules, ajouter des lignes (`+` en bas du tableau) ou supprimer des lignes. "
            "Toutes les options portent sur le même sous-jacent."
        )

    k_base = round(float(S0), 0)
    if strategie_preset == "Long Straddle (Achat Call + Achat Put ATM)":
        init_df = pd.DataFrame({
            "Type de position": ["Long Call", "Long Put"],
            "Quantité": [100.0, 100.0],
            "Strike K (€)": [k_base, k_base],
        })
    elif strategie_preset == "Short Strangle (Vente Put OTM + Vente Call OTM)":
        init_df = pd.DataFrame({
            "Type de position": ["Short Put", "Short Call"],
            "Quantité": [100.0, 100.0],
            "Strike K (€)": [round(k_base * 0.95, 1), round(k_base * 1.05, 1)],
        })
    elif strategie_preset == "Butterfly Spread (1 Long K1, 2 Short K2, 1 Long K3)":
        init_df = pd.DataFrame({
            "Type de position": ["Long Call", "Short Call", "Long Call"],
            "Quantité": [100.0, 200.0, 100.0],
            "Strike K (€)": [round(k_base * 0.95, 1), k_base, round(k_base * 1.05, 1)],
        })
    else:
        init_df = pd.DataFrame({
            "Type de position": ["Long Call", "Short Call"],
            "Quantité": [50.0, 20.0],
            "Strike K (€)": [k_base, round(k_base * 1.05, 1)],
        })

    portfolio_df = st.data_editor(
        init_df,
        num_rows="dynamic",
        column_config={
            "Type de position": st.column_config.SelectboxColumn(
                "Type de position",
                options=["Long Call", "Short Call", "Long Put", "Short Put"],
                required=True,
            ),
            "Quantité": st.column_config.NumberColumn("Quantité", required=True),
            "Strike K (€)": st.column_config.NumberColumn("Strike K (€)", required=True),
        },
        use_container_width=True,
    )

    if portfolio_df.empty:
        st.error("🚫 Le portefeuille est vide. Veuillez saisir au moins une ligne d'option.")
        st.stop()

    for i, row in portfolio_df.iterrows():
        if pd.isna(row["Type de position"]) or pd.isna(row["Quantité"]) or pd.isna(row["Strike K (€)"]):
            st.error(f"🚫 **Ligne #{i+1} invalide** : Aucun champ ne doit être laissé vide.")
            st.stop()
        if float(row["Quantité"]) <= 0:
            st.error(f"🚫 **Ligne #{i+1} invalide** : La quantité ({row['Quantité']}) doit être strictement positive (> 0).")
            st.stop()
        if float(row["Strike K (€)"]) <= 0:
            st.error(f"🚫 **Ligne #{i+1} invalide** : Le strike K ({row['Strike K (€)']}) doit être strictement positif (> 0).")
            st.stop()

# Conversion en décimales pour les calculs
r = float(r_pct) / 100.0
sigma_init = float(sigma_pct) / 100.0

# ==========================================
# SECTION 1 : RÉSULTATS À L'ÉTAT INITIAL (S0)
# ==========================================

details_0, res_0 = compute_portfolio(S0, T_init, r, sigma_init, portfolio_df)

st.header("📊 1. État Initial de la Position & Couverture (au cours $S_0$)")

m1, m2, m3, m4, m5 = st.columns(5)
if len(portfolio_df) == 1:
    m1.metric("Prix Unitaire de l'Option", f"{res_0['unit_price_first']:.2f} €")
else:
    m1.metric("Nombre de lignes (Legs)", f"{len(portfolio_df)} options")

m2.metric("Valeur de la Position", f"{res_0['value']:.2f} €")
m3.metric("Delta de la Position (Δ)", f"{res_0['delta']:+.2f}")
m4.metric("Gamma de la Position (Γ)", f"{res_0['gamma']:+.4f}")
m5.metric("Couverture Initiale Requise", format_hedge_order(res_0["hedge"]))

sens_gamma_txt = (
    "positif (vous êtes acheteur net de convexité : les grands mouvements de marché vous sont favorables)"
    if res_0["gamma"] > 1e-6
    else "négatif (vous êtes vendeur net d'options : les forts décalages de cours dégradent votre couverture)"
    if res_0["gamma"] < -1e-6
    else "proche de zéro (couverture localement stable)"
)

st.info(
    f"💡 **Lecture intuitive pour l'opérateur :**\n"
    f"- **Exposition directionnelle (Delta)** : Si l'action sous-jacente varie de **+1,00 €**, la valeur de votre position d'options "
    f"varie d'environ **{res_0['delta']:+.2f} €**.\n"
    f"- **Action de couverture immédiate** : Pour neutraliser ce risque directionnel (être *Delta-Neutre* à $S_0 = {S0:.2f}$ €), "
    f"vous devez immédiatement **{format_hedge_order(res_0['hedge']).lower()}** sur le marché.\n"
    f"- **Stabilité de la couverture (Gamma)** : Votre Gamma est **{res_0['gamma']:+.4f}**, il est {sens_gamma_txt}."
)

col_table, col_parity = st.columns([2, 1])

with col_table:
    st.markdown("##### 📋 Détail par ligne d'option et agrégation")
    st.dataframe(
        details_0.style.format({
            "Quantité": "{:.0f}",
            "Prix Unitaire (€)": "{:.2f} €",
            "Valeur Position (€)": "{:+.2f} €",
            "Delta Unitaire": "{:+.4f}",
            "Delta Position": "{:+.2f}",
            "Gamma Position": "{:+.4f}",
            "Vega Position": "{:+.2f} €",
            "Theta Position (€/j)": "{:+.2f} €",
        }),
        use_container_width=True,
        hide_index=True,
    )

with col_parity:
    k_ref = float(portfolio_df.iloc[0]["Strike K (€)"])
    parity = check_put_call_parity(S0, k_ref, T_init, r, sigma_init)
    st.markdown(f"##### ✅ Vérification Parité Call-Put ($K = {k_ref:.1f}$ €)")
    st.latex(rf"C - P = {parity['call']:.4f} - {parity['put']:.4f} = {parity['lhs']:.4f}\text{{ €}}")
    st.latex(rf"S_0 - K e^{{-rT}} = {S0:.2f} - {k_ref:.1f} e^{{-{r:.3f}\times {T_init:.2f}}} = {parity['rhs']:.4f}\text{{ €}}")
    if parity["is_valid"]:
        st.success(f"Parité parfaitement vérifiée (Écart : {parity['diff']:.2e} €)")
    else:
        st.error("Écart détecté sur la parité Call-Put.")

st.divider()

# ==========================================
# SECTION 2 : CURSEURS & ÉVOLUTION DYNAMIQUE (S0 -> S1)
# ==========================================

st.header("🎚️ 2. Simulation de Marché & Réajustement de la Couverture")
st.write(
    "Déplacez le curseur du cours sous-jacent ($S_1$) pour observer comment la couverture évolue, "
    "et utilisez les curseurs bonus ($T$ et $\sigma$) pour voir les courbes se déformer en temps réel."
)

s_min_slider = round(float(S0 * 0.60), 2)
s_max_slider = round(float(S0 * 1.40), 2)

cs1, cs2, cs3, cs4 = st.columns([1.2, 1.2, 1.2, 1])
with cs1:
    S1 = st.slider(
        "Nouveau cours du sous-jacent S1 (€) [±40%]",
        min_value=s_min_slider,
        max_value=s_max_slider,
        value=round(float(S0), 2),
        step=max(0.1, round(float(S0 * 0.01), 2)),
    )
with cs2:
    T_sim = st.slider(
        "Maturité simulée T (années) [Bonus]",
        min_value=0.01,
        max_value=max(2.0, float(T_init * 2.0)),
        value=float(T_init),
        step=0.01,
    )
with cs3:
    sigma_sim_pct = st.slider(
        "Volatilité simulée σ (%) [Bonus]",
        min_value=1.0,
        max_value=100.0,
        value=float(sigma_pct),
        step=1.0,
    )
with cs4:
    courbe_choix = st.radio(
        "Courbe à afficher (Graphe 1) :",
        ["Couverture (-Delta)", "Delta de la position (+Delta)"],
        help="Représente la couverture (-Delta) ou le Delta en fonction du cours S."
    )

sigma_sim = sigma_sim_pct / 100.0

_, res_1 = compute_portfolio(S1, T_sim, r, sigma_sim, portfolio_df)

ajustement_actions = res_1["hedge"] - res_0["hedge"]
pnl_options_seules = res_1["value"] - res_0["value"]
pnl_actions_couverture = res_0["hedge"] * (S1 - S0)
pnl_portefeuille_couvert = pnl_options_seules + pnl_actions_couverture

u1, u2, u3, u4, u5 = st.columns(5)
u1.metric(
    "Nouvelle Valeur Position (S1)",
    f"{res_1['value']:.2f} €",
    f"{pnl_options_seules:+.2f} € (PnL Options)",
)
u2.metric(
    "Nouveau Delta (S1)",
    f"{res_1['delta']:+.2f}",
    f"{res_1['delta'] - res_0['delta']:+.2f} vs S0",
)
u3.metric(
    "Nouveau Gamma (S1)",
    f"{res_1['gamma']:+.4f}",
    f"{res_1['gamma'] - res_0['gamma']:+.4f} vs S0",
)
u4.metric(
    "Nouvelle Couverture Cible (S1)",
    format_hedge_order(res_1["hedge"]),
)
u5.metric(
    "PnL Position Couverte (Delta-Hedgée)",
    f"{pnl_portefeuille_couvert:+.2f} €",
    help="Gain ou perte total incluant les actions de couverture détenues depuis S0.",
)

if abs(ajustement_actions) < 1e-4:
    st.success(
        f"🔄 **Ajustement de couverture ($S_0 = {S0:.2f}$ € $\\rightarrow S_1 = {S1:.2f}$ €) :** "
        f"Aucun réajustement requis. Vous conservez votre position actuelle ({format_hedge_order(res_0['hedge']).lower()})."
    )
else:
    st.warning(
        f"🔄 **Ajustement de couverture à exécuter ($S_0 = {S0:.2f}$ € $\\rightarrow S_1 = {S1:.2f}$ €) :** "
        f"Pour rester couvert (Delta-Neutre), vous devez **{format_hedge_order(ajustement_actions).upper()}** "
        f"(passage d'une couverture de *{format_hedge_order(res_0['hedge'])}* à *{format_hedge_order(res_1['hedge'])}*)."
    )

# ==========================================
# SECTION 3 : GRAPHIQUES (COUVERTURE / DELTA & GAMMA SUR ±40%)
# ==========================================

st.header("📈 3. Évolution de la Couverture et du Gamma (±40% autour de $S_0$)")

S_grid = np.linspace(S0 * 0.60, S0 * 1.40, 200)

delta_init_curve, hedge_init_curve, gamma_init_curve = [], [], []
delta_sim_curve, hedge_sim_curve, gamma_sim_curve = [], [], []

for s_val in S_grid:
    _, r_init = compute_portfolio(s_val, T_init, r, sigma_init, portfolio_df)
    delta_init_curve.append(r_init["delta"])
    hedge_init_curve.append(r_init["hedge"])
    gamma_init_curve.append(r_init["gamma"])

    _, r_sim = compute_portfolio(s_val, T_sim, r, sigma_sim, portfolio_df)
    delta_sim_curve.append(r_sim["delta"])
    hedge_sim_curve.append(r_sim["hedge"])
    gamma_sim_curve.append(r_sim["gamma"])

fig_main, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(15, 5.5))

if courbe_choix == "Couverture (-Delta)":
    y_init_1 = hedge_init_curve
    y_sim_1 = hedge_sim_curve
    y_pt_0 = res_0["hedge"]
    y_pt_1 = res_1["hedge"]
    titre_g1 = "Évolution de la Couverture Requise (-Δ) en fonction de S"
    ylabel_g1 = "Couverture : Actions à détenir (-Δ) [actions]"
    couleur_g1 = "#1f77b4"
else:
    y_init_1 = delta_init_curve
    y_sim_1 = delta_sim_curve
    y_pt_0 = res_0["delta"]
    y_pt_1 = res_1["delta"]
    titre_g1 = "Évolution du Delta de la Position (Δ) en fonction de S"
    ylabel_g1 = "Delta de la position (Δ) [actions]"
    couleur_g1 = "#1f77b4"

ax_left.plot(
    S_grid, y_init_1, color="grey", linestyle="--", alpha=0.6,
    label=f"Courbe initiale (T={T_init:.2f}a, σ={sigma_pct:.0f}%)"
)
ax_left.plot(
    S_grid, y_sim_1, color=couleur_g1, linewidth=2.5,
    label=f"Courbe active (T={T_sim:.2f}a, σ={sigma_sim_pct:.0f}%)"
)
ax_left.axvline(S0, color="grey", linestyle=":", alpha=0.7, label=f"Cours initial S0 ({S0:.1f} €)")
ax_left.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax_left.plot(S0, y_pt_0, "ko", markersize=7, label=f"Position en S0 ({y_pt_0:+.1f} act.)")
ax_left.plot(S1, y_pt_1, "ro", markersize=9, label=f"Point courant S1 ({y_pt_1:+.1f} act.)")

ax_left.set_title(titre_g1, fontsize=11, fontweight="bold")
ax_left.set_xlabel("Cours du sous-jacent S (€)")
ax_left.set_ylabel(ylabel_g1)
ax_left.grid(True, alpha=0.3)
ax_left.legend(loc="best", fontsize=8.5)

gamma_arr = np.array(gamma_sim_curve)
idx_max_instability = int(np.argmax(np.abs(gamma_arr)))
s_instable = S_grid[idx_max_instability]
g_instable = gamma_arr[idx_max_instability]

ax_right.plot(
    S_grid, gamma_init_curve, color="grey", linestyle="--", alpha=0.6,
    label=f"Gamma initial (T={T_init:.2f}a, σ={sigma_pct:.0f}%)"
)
ax_right.plot(
    S_grid, gamma_sim_curve, color="#6a0dad", linewidth=2.5,
    label=f"Gamma actif (T={T_sim:.2f}a, σ={sigma_sim_pct:.0f}%)"
)

seuil_instable = 0.75 * np.max(np.abs(gamma_arr))
masque_instable = np.abs(gamma_arr) >= seuil_instable
ax_right.fill_between(
    S_grid, gamma_arr, 0, where=masque_instable,
    color="orange", alpha=0.28, label="Zone d'instabilité maximale (|Γ| élevé)"
)

ax_right.axvline(S0, color="grey", linestyle=":", alpha=0.7)
ax_right.axhline(0, color="black", linewidth=0.8, alpha=0.5)
ax_right.plot(S0, res_0["gamma"], "ko", markersize=7, label=f"Gamma en S0 ({res_0['gamma']:+.3f})")
ax_right.plot(S1, res_1["gamma"], "ro", markersize=9, label=f"Point courant S1 ({res_1['gamma']:+.3f})")
ax_right.annotate(
    f"Pic d'instabilité\n(S ≈ {s_instable:.1f} €)",
    xy=(s_instable, g_instable),
    xytext=(s_instable, g_instable * 0.7 if abs(g_instable) > 0 else 0.1),
    arrowprops=dict(arrowstyle="->", color="darkorange", lw=1.2),
    fontsize=8.5,
    fontweight="bold",
    color="darkorange",
    ha="center",
)

ax_right.set_title("Évolution du Gamma (Γ) et Zone d'Instabilité", fontsize=11, fontweight="bold")
ax_right.set_xlabel("Cours du sous-jacent S (€)")
ax_right.set_ylabel("Gamma de la position (Γ) [Δ / €]")
ax_right.grid(True, alpha=0.3)
ax_right.legend(loc="best", fontsize=8.5)

plt.tight_layout()
st.pyplot(fig_main)

st.divider()

# ==========================================
# SECTION 4 : ANALYSES QUANTITATIVES COMPLÉMENTAIRES
# ==========================================

st.header("🔬 4. Analyses Avancées : Attribution de PnL, Payoff & Stress-Test")

col_adv1, col_adv2, col_adv3 = st.columns(3)

with col_adv1:
    st.markdown("#### 🧩 Attribution du PnL ($S_0 \\rightarrow S_1$)")
    st.caption("Décomposition mathématique de la variation de valeur entre l'état initial et l'état simulé :")

    dS = S1 - S0
    dsigma = (sigma_sim - sigma_init) * 100.0
    dT_days = -(T_sim - T_init) * 365.0

    effet_delta = res_0["delta"] * dS
    effet_gamma = 0.5 * res_0["gamma"] * (dS ** 2)
    effet_vega = res_0["vega"] * dsigma
    effet_theta = res_0["theta"] * dT_days
    pnl_taylor = effet_delta + effet_gamma + effet_vega + effet_theta

    attrib_df = pd.DataFrame({
        "Composante de Risque": [
            "1. Effet Delta (Δ · δS)",
            "2. Effet Gamma (½ Γ · δS²)",
            "3. Effet Vega (ν · δσ)",
            "4. Effet Temps (Θ · δt)",
            "Total Approché (Taylor)",
            "PnL Réel Exact (Black-Scholes)",
        ],
        "Impact (€)": [
            effet_delta,
            effet_gamma,
            effet_vega,
            effet_theta,
            pnl_taylor,
            pnl_options_seules,
        ],
    })
    st.dataframe(
        attrib_df.style.format({"Impact (€)": "{:+.2f} €"}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "📌 *Remarque : Lorsque vous êtes couvert en Delta, l'effet 1 est neutralisé par vos actions. "
        "Votre PnL résiduel est alors quasiment égal à l'Effet Gamma (ligne 2) !*"
    )

with col_adv2:
    st.markdown("#### 🎯 Profil de Payoff & Effet de la Couverture")
    S_payoff = np.linspace(S0 * 0.60, S0 * 1.40, 150)

    pnl_unhedged_curve = []
    pnl_hedged_curve = []
    pnl_maturity_curve = []

    for s_p in S_payoff:
        _, r_p = compute_portfolio(s_p, T_sim, r, sigma_sim, portfolio_df)
        pnl_u = r_p["value"] - res_0["value"]
        pnl_h = pnl_u + res_0["hedge"] * (s_p - S0)
        pnl_unhedged_curve.append(pnl_u)
        pnl_hedged_curve.append(pnl_h)

        val_T0 = 0.0
        for _, row in portfolio_df.iterrows():
            q = float(row["Quantité"])
            k = float(row["Strike K (€)"])
            sgn = 1.0 if "Long" in str(row["Type de position"]) else -1.0
            if "Call" in str(row["Type de position"]):
                val_T0 += sgn * q * max(s_p - k, 0.0)
            else:
                val_T0 += sgn * q * max(k - s_p, 0.0)
        pnl_maturity_curve.append(val_T0 - res_0["value"])

    fig_pay, ax_pay = plt.subplots(figsize=(6, 4.5))
    ax_pay.plot(S_payoff, pnl_maturity_curve, "--", color="grey", alpha=0.7, label="PnL à l'échéance (T=0)")
    ax_pay.plot(S_payoff, pnl_unhedged_curve, color="#1f77b4", lw=2, label="PnL Non Couvert (Options seules)")
    ax_pay.plot(S_payoff, pnl_hedged_curve, color="#2ca02c", lw=2.2, label="PnL Couvert en Delta (Options + Actions)")
    ax_pay.axhline(0, color="black", lw=0.8)
    ax_pay.axvline(S0, color="grey", linestyle=":")
    ax_pay.plot(S1, pnl_portefeuille_couvert, "ro", markersize=7, label="Point S1 (Couvert)")
    ax_pay.set_xlabel("Cours du sous-jacent S (€)")
    ax_pay.set_ylabel("Gain / Perte Net (€)")
    ax_pay.set_title("Comparaison PnL : Position Nue vs Couverte en Delta", fontsize=10, fontweight="bold")
    ax_pay.grid(True, alpha=0.3)
    ax_pay.legend(fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_pay)

with col_adv3:
    st.markdown("#### 🔥 Matrice de Stress-Test (Spot × Volatilité)")
    s_stress = np.linspace(S0 * 0.80, S0 * 1.20, 7)
    v_stress = np.linspace(max(0.05, sigma_init * 0.5), sigma_init * 1.5, 7)

    stress_matrix = np.zeros((len(v_stress), len(s_stress)))
    for i, v_val in enumerate(v_stress):
        for j, s_val in enumerate(s_stress):
            _, r_st = compute_portfolio(s_val, T_sim, r, v_val, portfolio_df)
            stress_matrix[i, j] = r_st["value"] - res_0["value"]

    fig_hm, ax_hm = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(
        stress_matrix,
        xticklabels=[f"{x:.0f}€" for x in s_stress],
        yticklabels=[f"{v*100:.0f}%" for v in v_stress],
        cmap="RdYlGn",
        center=0,
        annot=True,
        fmt=".0f",
        cbar_kws={"label": "PnL (€)"},
        ax=ax_hm,
        annot_kws={"size": 7.5},
    )
    ax_hm.set_title("PnL de la Position (€) selon S et σ", fontsize=10, fontweight="bold")
    ax_hm.set_xlabel("Cours du Sous-jacent S (€)")
    ax_hm.set_ylabel("Volatilité Annuelle σ (%)")
    plt.tight_layout()
    st.pyplot(fig_hm)

st.divider()

# ==========================================
# SECTION 5 : BONUS MONTE CARLO (PRICING, VAR & COUVERTURE DISCRÈTE)
# ==========================================

st.header("🎲 5. Bonus Monte Carlo : Convergence, VaR Full Revaluation & Couverture en Temps Discret")
st.write(
    "En pratique, un trader ne peut pas réajuster sa couverture en continu ($dt \\to 0$). "
    "Cette section utilise la méthode de **Monte Carlo vectorisée** pour : "
    "**(1)** valider la convergence du prix avec réduction de variance antithétique, "
    "**(2)** calculer la VaR 99 % et la CVaR par réévaluation complète, et "
    "**(3)** simuler l'erreur réelle de couverture (*Tracking Error*) selon la fréquence de rebalancement."
)

mc_col1, mc_col2, mc_col3 = st.columns(3)
with mc_col1:
    n_sims_mc = st.select_slider(
        "Nombre de trajectoires Monte Carlo (Pricing & VaR)",
        options=[5000, 10000, 25000, 50000],
        value=25000,
    )
with mc_col2:
    horizon_var_days = st.selectbox(
        "Horizon de risque pour la VaR / CVaR",
        options=[1, 5, 10, 21],
        index=1,
        format_func=lambda d: f"{d} jour(s) de bourse",
    )
with mc_col3:
    freq_label = st.selectbox(
        "Fréquence de rebalancement du Delta-Hedging jusqu'à T",
        options=[
            ("Mensuel (12 rebalancements)", 12),
            ("Hebdomadaire (52 rebalancements)", 52),
            ("Quotidien (252 rebalancements)", 252),
        ],
        index=1,
        format_func=lambda x: x[0],
    )
    n_steps_hedge = freq_label[1]

# 1. Exécution des 3 moteurs Monte Carlo vectorisés
mc_pricing_res = mc_portfolio_pricing(S0, T_init, r, sigma_init, portfolio_df, n_sims=n_sims_mc)
mc_var_res = mc_var_cvar_full_reval(
    S0, T_init, r, sigma_init, portfolio_df,
    val_0=res_0["value"], hedge_0=res_0["hedge"],
    horizon_days=horizon_var_days, n_sims=n_sims_mc, alpha=0.99
)
mc_hedge_res = mc_discrete_delta_hedging(
    S0, T_init, r, sigma_init, portfolio_df,
    val_0=res_0["value"], hedge_0=res_0["hedge"],
    n_steps=n_steps_hedge, n_paths=1000
)

# Affichage des métriques clés Monte Carlo
k_mc1, k_mc2, k_mc3, k_mc4, k_mc5 = st.columns(5)

ecart_mc_bs = mc_pricing_res["mc_price"] - res_0["value"]
k_mc1.metric(
    "Prix Monte Carlo (Antithétique)",
    f"{mc_pricing_res['mc_price']:.2f} €",
    f"{ecart_mc_bs:+.2f} € vs Black-Scholes",
    help=f"Intervalle de confiance à 95% : [{mc_pricing_res['ci_low']:.2f} €, {mc_pricing_res['ci_high']:.2f} €]",
)
k_mc2.metric(
    f"VaR 99% Non Couverte ({horizon_var_days}j)",
    f"-{mc_var_res['var_unhedged']:.2f} €",
    f"CVaR 99% : -{mc_var_res['cvar_unhedged']:.2f} €",
    delta_color="inverse",
)
k_mc3.metric(
    f"VaR 99% Couverte en Delta ({horizon_var_days}j)",
    f"-{mc_var_res['var_hedged']:.2f} €",
    f"CVaR 99% : -{mc_var_res['cvar_hedged']:.2f} €",
    delta_color="normal",
    help="Perte maximale à 99% lorsque le portefeuille est couvert en actions : le risque directionnel disparaît !",
)
k_mc4.metric(
    "Erreur Moyenne de Couverture (à T)",
    f"{mc_hedge_res['mean_error']:+.2f} €",
    help="En théorie comme en simulation, l'espérance de l'erreur de couverture autofinancée est proche de 0 €.",
)
k_mc5.metric(
    "Dispersion de l'Erreur σ(ε_T)",
    f"{mc_hedge_res['std_error']:.2f} €",
    f"90% dans [{mc_hedge_res['q05_error']:+.0f}€, {mc_hedge_res['q95_error']:+.0f}€]",
    help="Écart-type du PnL final à l'échéance dû au fait que l'on ne couvre qu'à dates discrètes et non en continu.",
)

# Graphiques Monte Carlo
fig_mc, (ax_mc1, ax_mc2, ax_mc3) = plt.subplots(1, 3, figsize=(16, 4.8))

# --- Graphe MC 1 : Distribution PnL (Non couvert vs Couvert en Delta) ---
ax_mc1.hist(
    mc_var_res["pnl_unhedged"], bins=50, alpha=0.45, color="#1f77b4",
    label=f"Non Couvert (VaR 99% = -{mc_var_res['var_unhedged']:.0f} €)", density=True
)
ax_mc1.hist(
    mc_var_res["pnl_hedged"], bins=50, alpha=0.65, color="#2ca02c",
    label=f"Couvert en Delta (VaR 99% = -{mc_var_res['var_hedged']:.0f} €)", density=True
)
ax_mc1.axvline(-mc_var_res["var_unhedged"], color="#1f77b4", linestyle="--", lw=1.8)
ax_mc1.axvline(-mc_var_res["var_hedged"], color="#2ca02c", linestyle="--", lw=1.8)
ax_mc1.axvline(0, color="black", lw=0.8)
ax_mc1.set_title(f"Distribution du PnL à {horizon_var_days} jour(s) (Full Revaluation)", fontsize=10, fontweight="bold")
ax_mc1.set_xlabel("Gain / Perte sur l'horizon (€)")
ax_mc1.set_ylabel("Densité")
ax_mc1.legend(fontsize=8)
ax_mc1.grid(True, alpha=0.3)

# --- Graphe MC 2 : Échantillon de trajectoires du sous-jacent jusqu'à T ---
n_show = min(40, mc_hedge_res["S_paths"].shape[0])
for i in range(n_show):
    ax_mc2.plot(mc_hedge_res["time_grid"], mc_hedge_res["S_paths"][i, :], lw=0.8, alpha=0.35)
ax_mc2.axhline(S0, color="black", linestyle="--", lw=1.2, label=f"S0 = {S0:.1f} €")
ax_mc2.set_title(f"Trajectoires Simulées S_t ({n_steps_hedge} pas de rebalancement)", fontsize=10, fontweight="bold")
ax_mc2.set_xlabel("Temps écoulé t (années)")
ax_mc2.set_ylabel("Cours du sous-jacent S_t (€)")
ax_mc2.legend(fontsize=8)
ax_mc2.grid(True, alpha=0.3)

# --- Graphe MC 3 : Distribution de l'Erreur Finale de Couverture à l'échéance T ---
ax_mc3.hist(
    mc_hedge_res["hedging_error"], bins=45, color="#6a0dad", alpha=0.65, edgecolor="white",
    label=f"σ(Erreur) = {mc_hedge_res['std_error']:.1f} €"
)
ax_mc3.axvline(0, color="black", lw=1.2, label="Couverture parfaite (0 €)")
ax_mc3.axvline(mc_hedge_res["q05_error"], color="red", linestyle=":", lw=1.5, label="Intervalle 5% - 95%")
ax_mc3.axvline(mc_hedge_res["q95_error"], color="red", linestyle=":", lw=1.5)
ax_mc3.set_title("Erreur Finale de Couverture (Tracking Error à T)", fontsize=10, fontweight="bold")
ax_mc3.set_xlabel("PnL Net Autofinancé à l'Échéance (€)")
ax_mc3.set_ylabel("Nombre de trajectoires (sur 1 000)")
ax_mc3.legend(fontsize=8)
ax_mc3.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig_mc)

st.info(
    f"🎓 **Interprétation théorique (Finance Computationnelle) :**\n"
    f"- **Convergence Antithétique** : Le prix Monte Carlo (**{mc_pricing_res['mc_price']:.2f} €**) converge vers le prix analytique de Black-Scholes "
    f"(**{res_0['value']:.2f} €**) avec un intervalle de confiance à 95 % de **[{mc_pricing_res['ci_low']:.2f} €, {mc_pricing_res['ci_high']:.2f} €]**.\n"
    f"- **Réduction drastique de la VaR** : Sur le premier graphique, la distribution verte (position couverte en Delta) est beaucoup plus resserrée que la distribution bleue (position nue). "
    f"La VaR 99 % à {horizon_var_days} jour(s) passe de **{mc_var_res['var_unhedged']:.2f} €** à **{mc_var_res['var_hedged']:.2f} €**.\n"
    f"- **Pourquoi l'erreur de couverture à l'échéance n'est-elle pas exactement nulle ?** En rebalançant **{n_steps_hedge} fois** sur la durée de vie de l'option, "
    f"le Gamma génère un écart entre deux dates de couverture consécutives (écart-type de **{mc_hedge_res['std_error']:.2f} €**). "
    f"Essayez de passer la fréquence de *Mensuel (12)* à *Quotidien (252)* : vous verrez l'écart-type de l'erreur diminuer d'un facteur proche de $\\sqrt{{252/12}} \\approx 4,5$ !"
)