# Options Pricing Suite

A computational finance toolkit implementing graduate-level options pricing methods in Python, built on coursework from an MSc Quantitative Finance programme.

**Real market data:** Uses a live SP500 options snapshot (320 strikes, Dec 2022) to compute implied volatilities and demonstrate the volatility smile — the key market anomaly that Black-Scholes cannot explain.

---

## Modules

| File | Topic | Methods |
|---|---|---|
| `pricer.py` | Core pricing | Black-Scholes, CRR binomial tree, Monte Carlo with antithetic variables, American options |
| `heston.py` | Stochastic volatility | Heston model, Gil-Pelaez Fourier inversion, Laplace transform multi-strike pricing, calibration |
| `implied_vol.py` | Market data analysis | Implied volatility extraction, volatility smile, BS vs market comparison |
| `greeks_mc.py` | Sensitivity analysis | MC delta via finite difference, infinitesimal perturbation, likelihood ratio, importance sampling |

---

## Quickstart

```bash
pip install numpy scipy pandas matplotlib
python pricer.py          # Module 1 — core pricing
python heston.py          # Module 2 — Heston model
python implied_vol.py     # Module 3 — vol smile (needs options_SP500.csv)
python greeks_mc.py       # Module 4 — MC Greeks
python run_all.py         # Run everything
```

Place `options_SP500.csv` in the project directory before running Module 3.

---

## Theory

### Black-Scholes

Assumes log-normal stock price under the risk-neutral measure:

```
dS = r*S*dt + sigma*S*dW

C = S*N(d1) - K*e^(-rT)*N(d2)
d1 = [ln(S/K) + (r + sigma^2/2)*T] / (sigma*sqrt(T))
d2 = d1 - sigma*sqrt(T)
```

### CRR Binomial Tree

Discretises stock price evolution into an N-step tree:

```
u = exp(sigma*sqrt(dt))      d = 1/u
p = (exp(r*dt) - d) / (u-d)
```

American options: at each node, compare holding value vs early exercise.

### Monte Carlo with Antithetic Variables

Terminal stock price under risk-neutral measure:

```
S_T = S * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z),  Z ~ N(0,1)
```

Antithetic method: average payoffs from Z and -Z to halve variance.

### Heston Stochastic Volatility

Variance itself follows a mean-reverting process:

```
dS = r*S*dt + sqrt(gamma)*S*dW1
d(gamma) = kappa*(lambda - gamma)*dt + sigma_tilde*sqrt(gamma)*dW2
dW1*dW2 = rho*dt
```

Characteristic function known in closed form — enables Fourier pricing.

### Monte Carlo Greeks

Three methods for computing `d(Price)/d(S)`:

| Method | Formula | Best for |
|---|---|---|
| Finite difference | `[Z(S+h/2) - Z(S-h/2)] / h` | Simple, any payoff |
| Infinitesimal perturbation | `E[1_{S_T>K} * dS_T/dS]` | Continuous payoffs, low variance |
| Likelihood ratio | `E[payoff * score function]` | Discontinuous payoffs (digital options) |

---

## Key result: The Volatility Smile

Black-Scholes assumes constant volatility. Real markets disagree. Running `implied_vol.py` on the SP500 data produces:

- **ATM implied vol:** ~22%
- **Deep OTM put IV:** often 30–40%
- **Shape:** IV rises steeply as K/S falls — the "smirk" or skew

This is driven by crash risk — investors pay a premium for downside protection. The Heston model can replicate this shape through stochastic volatility and the correlation parameter `rho < 0`.

---

## Sample output — Module 1

```
OPTIONS PRICING SUITE — Module 1: Core Pricer
S=3908.19, K=3900.0, T=0.3194y, r=0.0247, sigma=0.22

CALL OPTION
Black-Scholes        :    194.8431
CRR European (N=500) :    194.8221  diff=0.0210
CRR American (N=500) :    194.8221  early exercise premium=0.0000
Monte Carlo (N=200k) :    194.8619  SE=0.6421

PUT OPTION
Black-Scholes        :    158.7246
CRR European (N=500) :    158.7038  diff=0.0208
CRR American (N=500) :    159.1854  early exercise premium=0.4816
Monte Carlo (N=200k) :    158.7521  SE=0.5812
```

Note: American put has early exercise premium (0.48 points) — CRR captures this, BS does not.

---

## Dataset

`options_SP500.csv` — real SP500 options market snapshot

| Column | Description |
|---|---|
| Strike | Option strike price |
| Bid / Ask / Midpoint | Market prices |
| TTM | Time to maturity (years) |
| Underlying | S&P 500 spot level (3908.19) |
| Interest | Risk-free rate (2.47%) |
| Expiry | 2022-12-30 |

---

## Author

Syed Mohammad Zaheen  
MSc Quantitative Finance  
GitHub: [iamzaheen](https://github.com/iamzaheen)
