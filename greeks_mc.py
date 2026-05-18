"""
greeks_mc.py — Monte Carlo Greeks Estimation
=============================================
Implements three methods for computing option sensitivities (Greeks)
via Monte Carlo simulation, directly from Week 11 course content:

  1. Finite Difference (FD)             — bump-and-reprice
  2. Infinitesimal Perturbation (IP)    — pathwise derivative
  3. Likelihood Ratio (LR)              — score function method

Compares all three against the Black-Scholes closed-form benchmark.

Key insight: FD is simple but noisy. IP is cleaner because it
differentiates the payoff directly. LR works even for discontinuous
payoffs (like digital options). Each has its place in production systems.

Author: Syed Mohammad Zaheen
MSc Quantitative Finance
"""

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────
# CLOSED-FORM BENCHMARK
# ─────────────────────────────────────────────────────────

def bs_delta(S, K, T, r, sigma, option_type='call'):
    """Closed-form Black-Scholes delta."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1) if option_type == 'call' else norm.cdf(d1) - 1


def bs_vega(S, K, T, r, sigma):
    """Closed-form vega (same for calls and puts)."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T) / 100


# ─────────────────────────────────────────────────────────
# 1. FINITE DIFFERENCE METHOD
# ─────────────────────────────────────────────────────────
# Bump S by +h/2 and -h/2, take the difference.
# Central difference: error is O(h^2) vs O(h) for one-sided.

def delta_fd(S, K, T, r, sigma, N=50_000, h=1.0, seed=20):
    """
    Finite difference delta estimator.

    Z_n(s) = e^(-rT) * max(s*exp((r-sig^2/2)*T + sig*sqrt(T)*X) - K, 0)
    delta  = [Z_n(S+h/2) - Z_n(S-h/2)] / h
    """
    np.random.seed(seed)
    X    = np.random.normal(0, 1, N)
    term = (r - sigma**2 / 2) * T + sigma * np.sqrt(T) * X

    Z_up   = np.exp(-r * T) * np.maximum((S + h/2) * np.exp(term) - K, 0)
    Z_down = np.exp(-r * T) * np.maximum((S - h/2) * np.exp(term) - K, 0)
    Z_diff = (Z_up - Z_down) / h

    return float(Z_diff.mean()), float(np.var(Z_diff, ddof=1))


# ─────────────────────────────────────────────────────────
# 2. INFINITESIMAL PERTURBATION METHOD
# ─────────────────────────────────────────────────────────
# Differentiate inside the expectation (pathwise derivative).
# Z'_n(S) = 1_{S_T > K} * exp(-sigma^2*T/2 + sigma*sqrt(T)*X)
# Works because the payoff is continuous a.s.

def delta_ip(S, K, T, r, sigma, N=50_000, seed=20):
    """
    Infinitesimal perturbation (pathwise) delta estimator.
    Lower variance than FD because it uses the exact derivative path.
    """
    np.random.seed(seed)
    X    = np.random.normal(0, 1, N)
    term = (r - sigma**2 / 2) * T + sigma * np.sqrt(T) * X
    S_T  = S * np.exp(term)

    indicator = (S_T > K).astype(float)
    Z_prime   = indicator * np.exp(-sigma**2 * T / 2 + sigma * np.sqrt(T) * X)

    return float(Z_prime.mean()), float(np.var(Z_prime, ddof=1))


# ─────────────────────────────────────────────────────────
# 3. LIKELIHOOD RATIO METHOD
# ─────────────────────────────────────────────────────────
# Multiplies payoff by the score function (derivative of log-density).
# LR delta = e^(-rT) * max(S_T-K,0) * (X / (S*sigma*sqrt(T)))
# Works for ANY payoff, including discontinuous ones (digital options).

def delta_lr(S, K, T, r, sigma, N=50_000, seed=20):
    """
    Likelihood ratio (score function) delta estimator.
    Most general — works for discontinuous payoffs.
    """
    np.random.seed(seed)
    X    = np.random.normal(0, 1, N)
    term = (r - sigma**2 / 2) * T + sigma * np.sqrt(T) * X
    S_T  = S * np.exp(term)

    payoff = np.maximum(S_T - K, 0)
    score  = X / (S * sigma * np.sqrt(T))
    Z_lr   = np.exp(-r * T) * payoff * score

    return float(Z_lr.mean()), float(np.var(Z_lr, ddof=1))


# ─────────────────────────────────────────────────────────
# 4. IMPORTANCE SAMPLING FOR DEEP OTM OPTIONS
# ─────────────────────────────────────────────────────────
# From Week 11: for deep OTM options, standard MC is very noisy
# because most paths give zero payoff. Shift the sampling
# distribution toward the exercise region.

def mc_price_importance_sampling(S, K, T, r, sigma, N=10_000, seed=20):
    """
    Monte Carlo price with importance sampling for deep OTM calls.
    Shifts sampling mean to d = (log(K/S) - (r-sig^2/2)*T)/(sig*sqrt(T))
    so paths land in-the-money far more often.
    """
    np.random.seed(seed)

    # Optimal shift: put sampling centre at the exercise boundary
    d  = (np.log(K / S) - (r - sigma**2 / 2) * T) / (sigma * np.sqrt(T))
    mu = d  # shift mean of sampling distribution to d

    # Draw from shifted distribution N(mu, 1)
    Y    = np.random.normal(mu, 1, N)
    S_T  = S * np.exp((r - sigma**2 / 2) * T + sigma * np.sqrt(T) * Y)

    payoff = np.maximum(S_T - K, 0)

    # Likelihood ratio (Radon-Nikodym derivative) to correct for the shift
    lr_weight = np.exp(-mu * Y + 0.5 * mu**2)
    disc_payoffs = np.exp(-r * T) * payoff * lr_weight

    price     = float(np.mean(disc_payoffs))
    std_error = float(np.std(disc_payoffs) / np.sqrt(N))
    return price, std_error


# ─────────────────────────────────────────────────────────
# MAIN — comparison table
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    S, K, T, r, sigma = 3908.19, 3900.0, 0.3194, 0.0247, 0.22
    N = 50_000

    print("=" * 65)
    print("  OPTIONS PRICING SUITE — Module 4: MC Greeks")
    print("=" * 65)
    print(f"\n  Parameters: S={S}, K={K}, T={T:.4f}, r={r:.4f}, sigma={sigma}")
    print(f"  N = {N:,} simulations\n")

    # Closed-form benchmarks
    true_delta = bs_delta(S, K, T, r, sigma, 'call')
    true_vega  = bs_vega(S, K, T, r, sigma)

    print(f"  Closed-form delta  : {true_delta:.6f}")
    print(f"  Closed-form vega   : {true_vega:.6f}")

    print(f"\n  {'─'*60}")
    print(f"  DELTA ESTIMATION COMPARISON")
    print(f"  {'─'*60}")
    print(f"  {'Method':<30} {'Delta':>10}  {'Variance':>12}  {'Error':>10}")

    fd_d,  fd_v  = delta_fd(S, K, T, r, sigma, N=N)
    ip_d,  ip_v  = delta_ip(S, K, T, r, sigma, N=N)
    lr_d,  lr_v  = delta_lr(S, K, T, r, sigma, N=N)

    print(f"  {'Finite Difference (h=1.0)':<30} {fd_d:>10.6f}  {fd_v:>12.4f}  {abs(fd_d-true_delta):>10.6f}")
    print(f"  {'Infinitesimal Perturbation':<30} {ip_d:>10.6f}  {ip_v:>12.4f}  {abs(ip_d-true_delta):>10.6f}")
    print(f"  {'Likelihood Ratio':<30} {lr_d:>10.6f}  {lr_v:>12.4f}  {abs(lr_d-true_delta):>10.6f}")
    print(f"  {'Black-Scholes (true)':<30} {true_delta:>10.6f}  {'—':>12}  {'—':>10}")

    print(f"\n  Variance ranking: IP < FD < LR")
    print(f"  IP wins here because the call payoff is continuous.")
    print(f"  LR would win for a digital option (discontinuous payoff).")

    # Importance sampling for a deep OTM option
    print(f"\n  {'─'*60}")
    print(f"  IMPORTANCE SAMPLING — deep OTM call (K=5000)")
    print(f"  {'─'*60}")
    K_otm = 5000.0
    p_std, se_std = mc_price_importance_sampling(S, K_otm, T, r, sigma,
                                                 N=10_000, seed=20)
    # Standard MC for comparison
    np.random.seed(20)
    X = np.random.normal(0, 1, 10_000)
    S_T = S * np.exp((r - sigma**2/2)*T + sigma*np.sqrt(T)*X)
    pay = np.exp(-r*T) * np.maximum(S_T - K_otm, 0)
    p_naive = float(np.mean(pay))
    se_naive = float(np.std(pay) / np.sqrt(10_000))

    print(f"  Standard MC price   : {p_naive:.6f}  SE={se_naive:.6f}")
    print(f"  Importance sampling : {p_std:.6f}  SE={se_std:.6f}")
    if se_naive > 0:
        print(f"  Variance reduction  : {(se_naive/se_std)**2:.1f}x")

    print("\n  Module 4 complete.")
