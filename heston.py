"""
heston.py — Heston Stochastic Volatility Model
================================================
Implements option pricing under the Heston model using two methods
from the MSc course (Weeks 6-7):

  1. Gil-Pelaez Fourier inversion  — characteristic function integration
  2. Laplace transform / FFT       — simultaneous multi-strike pricing

The Heston model extends Black-Scholes by making volatility stochastic:
  dS(t) = r * S(t) dt + sqrt(gamma(t)) * S(t) dW1(t)
  d(gamma(t)) = kappa*(lambda - gamma(t)) dt + sigma_tilde * sqrt(gamma(t)) dW2(t)
  dW1 * dW2 = rho * dt

This captures the volatility smile that Black-Scholes cannot.

Author: Syed Mohammad Zaheen
MSc Quantitative Finance
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize


# ─────────────────────────────────────────────────────────
# 1. HESTON CHARACTERISTIC FUNCTION
# ─────────────────────────────────────────────────────────
# Stable Albrecher-Mayer-Schoutens-Tistaert (2007) formulation
# as implemented in Week 6 tutorial.

def heston_char(u, S0, T, r, gam0, kappa, lamb, sig_tilde, rho):
    """
    Characteristic function of log S(T) in the Heston model.

    Parameters
    ----------
    u         : complex — frequency variable
    S0        : float   — initial stock price
    T         : float   — time to expiry
    r         : float   — risk-free rate
    gam0      : float   — initial variance (= sigma0^2)
    kappa     : float   — mean-reversion level of variance
    lamb      : float   — mean-reversion speed
    sig_tilde : float   — volatility of variance
    rho       : float   — correlation between stock and variance Brownians
    """
    B    = rho * sig_tilde * u * 1j - lamb
    D    = np.sqrt(B**2 + sig_tilde**2 * u * (u + 1j))
    lnG  = np.log(B * (np.exp(-D * T) - 1) / (2 * D) + (np.exp(-D * T) + 1) / 2)
    psi0 = u * 1j * r * T - kappa / sig_tilde**2 * ((B + D) * T + 2 * lnG)
    psi1 = u * 1j
    psi2 = (u * (u + 1j) * (np.exp(-D * T) - 1)
            / (B * (np.exp(-D * T) - 1) + D * (np.exp(-D * T) + 1)))
    return np.exp(psi0 + psi1 * np.log(S0) + psi2 * gam0)


# ─────────────────────────────────────────────────────────
# 2. GIL-PELAEZ FOURIER INVERSION
# ─────────────────────────────────────────────────────────
# From the characteristic function, recover the density and price:
#   f(x) = (1/pi) * integral_0^inf Re[e^(-iux) * chi(u)] du
# Then:
#   Call = e^(-rT) * E[max(S_T - K, 0)]

def heston_call_fourier(S0, K, T, r, gam0, kappa, lamb, sig_tilde, rho,
                        upper=100):
    """
    Price a European call option in the Heston model using
    Gil-Pelaez Fourier inversion.

    Returns
    -------
    price : float — call option price
    """
    def integrand(u):
        chi = heston_char(u - 1j, S0, T, r, gam0, kappa, lamb, sig_tilde, rho)
        chi0 = heston_char(u, S0, T, r, gam0, kappa, lamb, sig_tilde, rho)
        denom = 1j * u
        part1 = np.real(np.exp(-1j * u * np.log(K)) * chi / (denom * S0 * np.exp(r * T)))
        part2 = np.real(np.exp(-1j * u * np.log(K)) * chi0 / denom)
        return part1 - part2

    result, _ = quad(integrand, 1e-8, upper, limit=200)
    price = S0 - K * np.exp(-r * T) * 0.5 - K * np.exp(-r * T) * result / np.pi
    return max(price, 0.0)


# ─────────────────────────────────────────────────────────
# 3. LAPLACE TRANSFORM MULTI-STRIKE PRICING (FFT-style)
# ─────────────────────────────────────────────────────────
# From Week 6: price calls at many strikes simultaneously.
# R is the damping constant for call options (R > 1).

def heston_call_laplace(K_array, S0, T, r, gam0, kappa, lamb, sig_tilde, rho,
                        R=1.5, upper=100):
    """
    Price European calls at multiple strikes using the Laplace transform.

    Parameters
    ----------
    K_array : array — array of strike prices
    R       : float — damping constant (R > 1 for calls)

    Returns
    -------
    prices : array — call prices for each strike in K_array
    """
    prices = np.zeros(len(K_array))
    for i, K in enumerate(K_array):
        def integrand(u):
            z    = R + 1j * u
            chi  = heston_char(-z, S0, T, r, gam0, kappa, lamb, sig_tilde, rho)
            num  = np.exp(-1j * u * np.log(K)) * chi
            denom = z * (z - 1)
            return np.real(num / denom)
        result, _ = quad(integrand, 0, upper, limit=200)
        prices[i] = max(np.exp(-r * T) * K**(1 - R) / np.pi * result, 0)
    return prices


# ─────────────────────────────────────────────────────────
# 4. HESTON CALIBRATION TO MARKET PRICES
# ─────────────────────────────────────────────────────────
# Fit Heston parameters to observed market option prices
# by minimising squared error between model and market.

def calibrate_heston(market_strikes, market_prices, S0, T, r,
                     method='nelder-mead'):
    """
    Calibrate Heston parameters to market call prices.

    Parameters
    ----------
    market_strikes : array — observed strike prices
    market_prices  : array — observed market (mid) prices
    method         : str   — scipy optimiser method

    Returns
    -------
    params : dict — calibrated {gam0, kappa, lamb, sig_tilde, rho}
    rmse   : float — root mean squared error of fit
    """
    def objective(x):
        gam0, kappa, lamb, sig_tilde, rho = x
        # Parameter constraints (keep physically sensible)
        if (gam0 <= 0 or kappa <= 0 or lamb <= 0 or
                sig_tilde <= 0 or abs(rho) >= 1):
            return 1e10
        # Feller condition: 2*kappa*lamb >= sig_tilde^2
        if 2 * kappa * lamb < sig_tilde**2:
            return 1e10
        try:
            model_prices = heston_call_laplace(
                market_strikes, S0, T, r, gam0, kappa, lamb, sig_tilde, rho)
            return np.mean((model_prices - market_prices)**2)
        except Exception:
            return 1e10

    # Initial guess: close to Black-Scholes (low kappa variance)
    x0 = [0.04, 0.09, 1.0, 0.3, -0.5]
    bounds = [(1e-4, 2.0), (1e-4, 2.0), (1e-4, 10.0),
              (1e-4, 2.0), (-0.99, 0.99)]

    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6})

    gam0, kappa, lamb, sig_tilde, rho = result.x
    params = {'gam0': gam0, 'kappa': kappa, 'lamb': lamb,
              'sig_tilde': sig_tilde, 'rho': rho}

    model_prices = heston_call_laplace(
        market_strikes, S0, T, r, **params)
    rmse = np.sqrt(np.mean((model_prices - market_prices)**2))

    return params, rmse


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

if __name__ == '__main__':

    # Heston parameters from course tutorials
    S0        = 3908.19
    T         = 0.3194
    r         = 0.0247
    gam0      = 0.04        # initial variance (20% vol)
    kappa     = 0.09        # long-run variance level
    lamb      = 1.5         # mean-reversion speed
    sig_tilde = 0.3         # vol of vol
    rho       = -0.7        # typical negative equity correlation

    print("=" * 60)
    print("  OPTIONS PRICING SUITE — Module 2: Heston Model")
    print("=" * 60)
    print(f"\n  S0={S0}, T={T:.4f}, r={r:.4f}")
    print(f"  gam0={gam0}, kappa={kappa}, lamb={lamb}")
    print(f"  sig_tilde={sig_tilde}, rho={rho}")

    # Price a single call via Fourier inversion
    K_test = 3900.0
    call_fourier = heston_call_fourier(S0, K_test, T, r, gam0, kappa,
                                       lamb, sig_tilde, rho)
    print(f"\n  Heston call (K={K_test}, Fourier): {call_fourier:.4f}")

    # Price across a range of strikes
    K_array = np.linspace(3200, 4600, 30)
    prices  = heston_call_laplace(K_array, S0, T, r, gam0, kappa,
                                   lamb, sig_tilde, rho)

    print(f"\n  Laplace transform pricing across 30 strikes:")
    print(f"  {'Strike':>8}  {'Heston Call':>12}")
    for k, p in zip(K_array[::5], prices[::5]):
        print(f"  {k:>8.1f}  {p:>12.4f}")

    print("\n  Module 2 complete.")
