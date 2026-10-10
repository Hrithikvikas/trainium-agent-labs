#!/usr/bin/env python3
"""
exp2_terms_and_tolerance.py - how many terms does the parabola really need?

The parabola (level 1.3) has no finite answer, so the checker accepts a truncated series
whose starting shape is within `tol` (0.5% by default). This script uses the CORRECT
coefficients and only varies how many terms are kept, so any failure is purely "not
enough terms". It shows how the tolerance setting decides how much work the model must do.

    python exp2_terms_and_tolerance.py
    python exp2_terms_and_tolerance.py --tols 0.02 0.01 0.005 0.001 0.0001 --max-terms 12
"""

import argparse

from common import problem, true_coeff, series, grade, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--max-terms", type=int, default=8, help="number of odd terms to try up to")
    ap.add_argument("--tols", type=float, nargs="+", default=[0.01, 0.005, 0.001, 0.0001])
    a = ap.parse_args()

    # Part 1: starting-shape error vs number of (odd) terms, for each seed.
    print("Part 1: starting-shape error with the CORRECT coefficients, keeping n odd terms")
    print("(the seed changes the rod length L and conductivity k)\n")
    header = "seed   L   k    " + "".join(f"{n:>8}" for n in range(1, a.max_terms + 1))
    print(header + "   <- odd terms kept")
    print("-" * len(header))
    errors = {}
    for seed in a.seeds:
        p = problem(1, 3, seed)
        coeffs = []
        row = []
        for n in range(1, 2 * a.max_terms, 2):
            # extend the coefficient list up to n (even n are exactly zero)
            while len(coeffs) < n:
                m = len(coeffs) + 1
                coeffs.append(true_coeff(p, m) if m % 2 else 0)
            r = grade(p, f"u(x, t) = {sp.sstr(series(p, coeffs))}")
            row.append(r["start_error"])
        errors[seed] = row
        print(f"{seed:>4} {sp.sstr(p['L']):>3} {sp.sstr(p['k']):>4}    "
              + "".join(f"{e * 100:7.3f}%" for e in row))

    # Part 2: for each tolerance, the fewest terms that pass. This is the knob the
    # problem designer turns to make the task harder or easier.
    print("\nPart 2: fewest odd terms needed to pass, for each tolerance")
    print("(the repo uses 0.5% -> 3 terms)\n")
    for tol in a.tols:
        need = []
        for seed in a.seeds:
            ok = [i + 1 for i, e in enumerate(errors[seed]) if e < tol]
            need.append(str(ok[0]) if ok else f">{a.max_terms}")
        print(f"  tol {tol * 100:>7.3f}%   terms needed per seed: {', '.join(need)}")

    print("\nNotes:"
          "\n  * Every seed gives the same errors: the error is RELATIVE, and the parabola x(L-x)"
          "\n    looks the same at any length L once rescaled; k only affects time, not t=0."
          "\n    So changing the seed does not make level 1.3 harder or easier."
          "\n  * The error shrinks roughly like 1/n^2.5 (coefficients fall like 1/n^3),"
          "\n    so every 10x tighter tolerance costs a lot more terms.")


if __name__ == "__main__":
    main()
