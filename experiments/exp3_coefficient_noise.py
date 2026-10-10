#!/usr/bin/env python3
"""
exp3_coefficient_noise.py - how accurate do the model's numbers need to be?

A language model writes numbers like 1.03 instead of 32/pi**3, and it makes small arithmetic
slips. This script takes the correct 3-term answer to the parabola, then
  (a) rounds every coefficient to d significant figures, and
  (b) adds random relative error of e% to every coefficient,
and measures how often the checker still says SOLVED. It tells you how much slack the
reward function gives, i.e. how close "close enough" is.

    python exp3_coefficient_noise.py
    python exp3_coefficient_noise.py --terms 4 --trials 200
"""

import argparse
import random

from common import problem, true_coeff, series, grade, sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="which problem instance")
    ap.add_argument("--terms", type=int, default=3, help="odd terms in the base answer")
    ap.add_argument("--trials", type=int, default=100, help="random trials per noise level")
    ap.add_argument("--noise", type=float, nargs="+",
                    default=[0, 0.1, 0.5, 1, 2, 5, 10, 20, 50])
    a = ap.parse_args()

    p = problem(1, 3, a.seed)
    n_max = 2 * a.terms - 1
    exact = [true_coeff(p, n) if n % 2 else 0 for n in range(1, n_max + 1)]
    vals = [float(c) for c in exact]
    print(f"{p['name']}: correct coefficients for n = 1..{n_max}:")
    print("   " + ", ".join(f"c{n}={v:.5g}" for n, v in enumerate(vals, start=1)))

    def check(cs):
        return grade(p, f"u(x, t) = {sp.sstr(series(p, cs))}")

    # (a) rounding to d significant figures
    print("\n(a) every coefficient rounded to d significant figures")
    for d in (1, 2, 3, 4, 6):
        cs = [sp.Float(f"{v:.{d}g}") if v else 0 for v in vals]
        r = check(cs)
        print(f"   {d} s.f.: reward {r['reward']:.1f}   start error {r['start_error'] * 100:.3f}%   "
              f"e.g. c1 = {cs[0]}")

    # (b) random relative noise on every coefficient
    print(f"\n(b) random error of up to +/-e% on every coefficient, {a.trials} trials each")
    print("     e%   solved   mean reward   mean start error")
    rng = random.Random(0)
    for e in a.noise:
        solved, rewards, errs = 0, [], []
        for _ in range(a.trials if e else 1):
            cs = [v * (1 + rng.uniform(-e, e) / 100) if v else 0 for v in vals]
            r = check([sp.Float(c) for c in cs])
            solved += r["reward"] == 1.0
            rewards.append(r["reward"])
            errs.append(r["start_error"])
        n = len(rewards)
        print(f"  {e:>5g}   {solved / n * 100:5.0f}%      {sum(rewards) / n:.2f}          "
              f"{sum(errs) / n * 100:.3f}%")

    print("\nReading it: the 3-term answer already uses 0.34% of the 0.5% budget just from"
          "\ntruncation, so only a little coefficient error fits on top of it. Try --terms 5,"
          "\nwhich leaves more budget for sloppy numbers.")


if __name__ == "__main__":
    main()
