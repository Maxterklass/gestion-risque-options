# Outil de Gestion du Risque d'une Position d'Options
Réalisé par Thomas BRISSON et Maxime CHAPRON dans le cadre du cours "Finance Computationnelle"

Application interactive de finance computationnelle développée en Python (Streamlit) pour le suivi, la valorisation et la couverture dynamique (**Delta-Gamma Hedging**) d'une option unique ou d'un portefeuille multi-options sous le modèle de **Black-Scholes**.

---

## 🚀 Accès en Ligne (Application Live)

Pour tester l'outil directement sans aucune installation locale, cliquez sur le lien ci-dessous :

👉 **[Accéder à l'application Streamlit en ligne](https://gestion-risque-options-ft3z8qhqnqokkkcxdwgrzz.streamlit.app/)**  


---

## 📂 Architecture du Projet

Le code source est rigoureusement séparé entre la logique mathématique (backend) et l'interface utilisateur (frontend) :
* **`pricing_engine.py`** : Moteur quantitatif contenant les formules fermées de Black-Scholes (`bs_price`, `bs_greeks`), la vérification de la parité Call-Put, l'algorithme de Newton-Raphson pour la volatilité implicite, ainsi que les simulations Monte Carlo vectorisées (pricing antithétique, VaR en réévaluation complète et erreur de couverture en temps discret).
* **`app.py`** : Interface graphique interactive (Streamlit) permettant la saisie sécurisée des paramètres, l'affichage des ordres de couverture en clair (« Acheter X actions »), les curseurs de simulation et les visualisations graphiques (Delta, Gamma, Payoff, Heatmap de stress-test).
* **`requirements.txt`** : Liste des dépendances nécessaires au déploiement cloud de l'application.

---

## 🎯 Fonctionnalités clés de l'outil

1. **Moteur Analytique & Parité Call-Put** : Valorisation exacte et démonstration mathématique instantanée de la relation $C - P = S_0 - Ke^{-rT}$.
2. **Gestion de Portefeuille (Bonus M2)** : Constitution libre d'un portefeuille multi-options avec agrégation en temps réel des sensibilités (Delta net, Gamma net, Vega, Theta, Rho).
3. **Couverture en Clair & Gamma Hedging** : Traduction automatique du Delta en instructions d'achat ou de vente d'actions, et proposition de couverture du Gamma par option de référence.
4. **Simulation Dynamique & Attribution de PnL** : Curseurs interactifs pour faire varier le spot $S$, la maturité $T$ et la volatilité $\sigma$, avec décomposition des gains par un développement de Taylor (effets Delta, Gamma, Vega, Theta).
5. **Bonus Monte Carlo** : Validation du pricing par réduction de variance antithétique, calcul de la VaR 99% par réévaluation complète, et simulation de l'erreur de couverture (*Tracking Error*) selon la fréquence de rebalancement en temps discret.