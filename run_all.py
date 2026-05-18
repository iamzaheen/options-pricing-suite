"""
run_all.py — Run the full Options Pricing Suite
================================================
Runs all four modules in sequence and prints a summary.
Place options_SP500.csv in the same directory before running.

Usage:
    python run_all.py
"""

import sys
import time

print("=" * 65)
print("  OPTIONS PRICING SUITE — Full Run")
print("  MSc Quantitative Finance — Syed Mohammad Zaheen")
print("=" * 65)

modules = [
    ("Module 1: Core Pricer      (BS + CRR + Monte Carlo)", "pricer"),
    ("Module 2: Heston Model     (Fourier + Laplace)",       "heston"),
    ("Module 3: Implied Vol      (real SP500 data)",          "implied_vol"),
    ("Module 4: MC Greeks        (FD + IP + LR)",            "greeks_mc"),
]

for label, mod in modules:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}\n")
    t0 = time.time()
    try:
        exec(open(f"{mod}.py").read())
    except FileNotFoundError as e:
        print(f"  Skipped: {e}")
    print(f"\n  Time: {time.time()-t0:.1f}s")

print(f"\n{'='*65}")
print("  All modules complete.")
print(f"{'='*65}\n")
