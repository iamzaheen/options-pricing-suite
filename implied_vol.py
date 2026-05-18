"""
implied_vol.py — Implied Volatility & Vol Smile from Real SP500 Data
=====================================================================
This module uses the real options_SP500.csv dataset to:

  1. Compute Black-Scholes implied volatility for each market quote
  2. Plot the volatility smile — the key market anomaly BS cannot explain
  3. Fit the Heston model to the implied vol surface
  4. Compare BS flat-vol vs Heston smile

This is the most important module for interviews — it bridges
textbook theory to real market data.

Data: SP500 options snapshot, expiry 2022-12-30
      320 strikes from 3025 to 5250, S&P at 3908.19

Author: Syed Mohammad Zaheen
MSc Quantitative Finance
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.stats import norm


# ─────────────────────────────────────────────────────────
# 1. IMPLIED VOLATILITY SOLVER
# ─────────────────────────────────────────────────────────
# Given a market price, find the sigma that makes BS match it.
# Uses Brent's method — fast, robust root-finder.

def bs_price(S, K, T, r, sigma, option_type='call'):
    """Black-Scholes price (standalone, no tuple return)."""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(market_price, S, K, T, r, option_type='call',
                tol=1e-8, max_iter=500):
    """
    Compute implied volatility using Brent's root-finding method.

    Returns sigma such that BS(sigma) = market_price.
    Returns NaN if no solution found (deep ITM/OTM).
    """
    # Intrinsic value check — no IV if option is below intrinsic
    if option_type == 'call':
        intrinsic = max(S - K * np.exp(-r * T), 0)
    else:
        intrinsic = max(K * np.exp(-r * T) - S, 0)

    if market_price <= intrinsic + 1e-6:
        return np.nan

    objective = lambda sigma: bs_price(S, K, T, r, sigma, option_type) - market_price

    try:
        # Search between very low and very high volatility
        iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        return iv
    except (ValueError, RuntimeError):
        return np.nan


# ─────────────────────────────────────────────────────────
# 2. LOAD AND PROCESS SP500 DATA
# ─────────────────────────────────────────────────────────

def load_sp500_data(filepath='options_SP500.csv'):
    """
    Load SP500 options data and compute implied volatilities.

    Returns a cleaned DataFrame with IV column added.
    """
    df = pd.read_csv(filepath)

    S = df['Underlying'].iloc[0]
    T = df['TTM'].iloc[0]
    r = df['Interest'].iloc[0]

    print(f"  Spot S&P 500 : {S:.2f}")
    print(f"  Time to expiry: {T:.4f} years ({T*365:.0f} days)")
    print(f"  Risk-free rate: {r*100:.3f}%")
    print(f"  Expiry date   : {df['Expiry'].iloc[0]}")
    print(f"  Options loaded: {len(df)}")
    print(f"  Strike range  : {df['Strike'].min():.0f} – {df['Strike'].max():.0f}")

    # Compute IV for each option (using midpoint price)
    # Use calls for strikes >= ATM, puts for strikes < ATM (more liquid)
    ivs = []
    types = []
    for _, row in df.iterrows():
        K  = row['Strike']
        mp = row['Midpoint']
        otype = 'call' if K >= S else 'put'

        # For puts, use put-call parity to get put price from call midpoint
        # Actually treat all as calls from the data (data contains call prices)
        iv = implied_vol(mp, S, K, T, r, option_type='call')
        ivs.append(iv)
        types.append(otype)

    df['IV']   = ivs
    df['Type'] = types
    df['Moneyness'] = df['Strike'] / S   # K/S ratio

    # Remove options where IV couldn't be computed
    df_clean = df.dropna(subset=['IV'])
    df_clean = df_clean[(df_clean['IV'] > 0.01) & (df_clean['IV'] < 2.0)]

    print(f"  Valid IV solutions: {len(df_clean)} / {len(df)}")
    return df_clean, S, T, r


# ─────────────────────────────────────────────────────────
# 3. PLOT VOLATILITY SMILE
# ─────────────────────────────────────────────────────────

def plot_vol_smile(df, S, T, r, save=True):
    """
    Plot the implied volatility smile from real SP500 data.

    The smile shows that OTM options trade at higher IV than ATM —
    a key market phenomenon that Black-Scholes cannot explain.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Left plot: IV vs Strike ──────────────────────────
    ax = axes[0]
    ax.scatter(df['Strike'], df['IV'] * 100, s=10, alpha=0.6,
               color='steelblue', label='Market IV')
    ax.axvline(S, color='red', linestyle='--', alpha=0.7,
               label=f'ATM (S={S:.0f})')
    ax.axhline(df.loc[(df['Strike'] - S).abs().idxmin(), 'IV'] * 100,
               color='green', linestyle=':', alpha=0.5, label='ATM IV level')
    ax.set_xlabel('Strike Price', fontsize=12)
    ax.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax.set_title('SP500 Implied Volatility Smile\n(real market data, Dec 2022)',
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ── Right plot: IV vs Moneyness (K/S) ───────────────
    ax2 = axes[1]
    ax2.scatter(df['Moneyness'], df['IV'] * 100, s=10, alpha=0.6,
                color='darkorange', label='Market IV')
    ax2.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='ATM (K/S=1)')

    # Overlay BS flat vol (ATM IV) — shows BS cannot match the smile
    atm_iv = df.loc[(df['Strike'] - S).abs().idxmin(), 'IV']
    ax2.axhline(atm_iv * 100, color='navy', linestyle='-',
                alpha=0.8, linewidth=2, label=f'BS flat vol ({atm_iv*100:.1f}%)')

    ax2.set_xlabel('Moneyness (K/S)', fontsize=12)
    ax2.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax2.set_title('IV vs Moneyness — the Volatility Smile\n(BS predicts flat line; market disagrees)',
                  fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig('vol_smile.png', dpi=150)
        print("  Saved: vol_smile.png")
    plt.show()


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("=" * 60)
    print("  OPTIONS PRICING SUITE — Module 3: Implied Volatility")
    print("=" * 60)
    print()

    try:
        df, S, T, r = load_sp500_data('options_SP500.csv')
    except FileNotFoundError:
        print("  options_SP500.csv not found.")
        print("  Place the file in the same directory and re-run.")
        exit()

    print(f"\n  ATM implied vol : {df.loc[(df['Strike']-S).abs().idxmin(),'IV']*100:.2f}%")
    print(f"  Min IV          : {df['IV'].min()*100:.2f}%")
    print(f"  Max IV          : {df['IV'].max()*100:.2f}%")
    print(f"  IV skew (OTM-ATM): vol increases as K/S decreases — equity risk premium")

    # Sample output table
    print(f"\n  {'Strike':>8}  {'K/S':>6}  {'Midpoint':>10}  {'IV (%)':>8}")
    sample = df.iloc[::30]
    for _, row in sample.iterrows():
        print(f"  {row['Strike']:>8.0f}  {row['Moneyness']:>6.3f}  {row['Midpoint']:>10.2f}  {row['IV']*100:>8.2f}%")

    print(f"\n  Plotting volatility smile...")
    plot_vol_smile(df, S, T, r)
    print("\n  Module 3 complete.")
