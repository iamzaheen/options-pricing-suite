"""
pricer.py — Core Options Pricing Module
========================================
Implements three pricing methods from the MSc Computational Finance course:

  1. Black-Scholes  — closed-form analytical solution (Weeks 1-3)
  2. CRR Binomial Tree — Cox-Ross-Rubinstein model (Weeks 1-3)
  3. Monte Carlo — with antithetic variance reduction (Weeks 9-10)

Supports both European and American options.

Author: Syed Mohammad Zaheen
MSc Quantitative Finance
"""

import numpy as np
from scipy.stats import norm


def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0), 0, 0
        else:
            return max(K - S, 0), 0, 0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price, d1, d2


def greeks(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        return {'delta': 0, 'gamma': 0, 'vega': 0, 'theta': 0, 'rho': 0}
    _, d1, d2 = black_scholes(S, K, T, r, sigma, option_type)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100
    if option_type == 'call':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
        rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
        rho   = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta, 'rho': rho}


def crr_binomial(S, K, T, r, sigma, N=500, option_type='call', exercise='european'):
    dt = T / N
    u  = np.exp(sigma * np.sqrt(dt))
    d  = 1.0 / u
    p  = (np.exp(r * dt) - d) / (u - d)
    q  = 1.0 - p
    discount = np.exp(-r * dt)
    j   = np.arange(N + 1)
    S_T = S * (u ** (N - j)) * (d ** j)
    V   = np.maximum(S_T - K, 0) if option_type == 'call' else np.maximum(K - S_T, 0)
    for i in range(N - 1, -1, -1):
        V = discount * (p * V[:-1] + q * V[1:])
        if exercise == 'american':
            S_i = S * (u ** (i - np.arange(i + 1))) * (d ** np.arange(i + 1))
            intrinsic = np.maximum(S_i - K, 0) if option_type == 'call' else np.maximum(K - S_i, 0)
            V = np.maximum(V, intrinsic)
    return float(V[0])


def monte_carlo(S, K, T, r, sigma, option_type='call', N=200_000, antithetic=True, seed=42):
    rng = np.random.default_rng(seed)
    Z   = rng.standard_normal(N)
    def S_T(z): return S * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    def payoff(s): return np.maximum(s - K, 0) if option_type == 'call' else np.maximum(K - s, 0)
    if antithetic:
        disc = np.exp(-r * T) * (payoff(S_T(Z)) + payoff(S_T(-Z))) / 2
    else:
        disc = np.exp(-r * T) * payoff(S_T(Z))
    price = float(np.mean(disc))
    se    = float(np.std(disc) / np.sqrt(N))
    return price, se, price - 1.96 * se, price + 1.96 * se


if __name__ == '__main__':
    S, K, T, r, sigma = 3908.19, 3900.0, 0.3194, 0.0247, 0.22
    print("=" * 60)
    print("  OPTIONS PRICING SUITE — Module 1: Core Pricer")
    print("=" * 60)
    print(f"\n  S={S}, K={K}, T={T:.4f}y, r={r:.4f}, sigma={sigma:.2f}")
    for otype in ['call', 'put']:
        print(f"\n  {otype.upper()}")
        bs, d1, d2 = black_scholes(S, K, T, r, sigma, otype)
        crr_eur    = crr_binomial(S, K, T, r, sigma, N=500, option_type=otype, exercise='european')
        crr_am     = crr_binomial(S, K, T, r, sigma, N=500, option_type=otype, exercise='american')
        mc, se, lo, hi = monte_carlo(S, K, T, r, sigma, option_type=otype)
        print(f"  Black-Scholes        : {bs:>10.4f}")
        print(f"  CRR European (N=500) : {crr_eur:>10.4f}  diff={abs(bs-crr_eur):.4f}")
        print(f"  CRR American (N=500) : {crr_am:>10.4f}  early exercise premium={crr_am-crr_eur:.4f}")
        print(f"  Monte Carlo (N=200k) : {mc:>10.4f}  SE={se:.4f}")
        print(f"  95% CI               : [{lo:.4f}, {hi:.4f}]")
        g = greeks(S, K, T, r, sigma, otype)
        print(f"  Greeks: delta={g['delta']:+.4f} gamma={g['gamma']:.5f} vega={g['vega']:.4f} theta={g['theta']:.4f}")
    print("\n  Module 1 complete.")
