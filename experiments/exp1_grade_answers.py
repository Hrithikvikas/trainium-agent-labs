#!/usr/bin/env python3
"""
exp1_grade_answers.py - feed the checker a gallery of right and wrong answers.

Question: does the checker give the score we expect for each kind of mistake, and is its
feedback useful? This is the "unit test" view of the reward function.

    python exp1_grade_answers.py                       # level 1.3, the parabola
    python exp1_grade_answers.py --level 1 --sub 1     # insulated end
    python exp1_grade_answers.py --reveal              # feedback with target numbers shown
    python exp1_grade_answers.py --answer "u(x, t) = exp(-pi**2*t/2)*sin(pi*x/2)"   # your own
"""

import argparse

import sympy as sp

from common import pdecheck, problem, describe, true_coeff, series, show, x, t


def gallery(p):
    """(label, answer text) pairs. Built from the problem so they work for any seed."""
    L, k, f = p["L"], p["k"], p["f"]
    s = sp.sstr
    out = []

    if p["exact"] is not None:
        # Problems with a finite closed-form answer (level 0, level 1.1 and 1.2).
        u = p["exact"]
        out += [
            ("correct, exact", f"u(x, t) = {s(u)}"),
            ("correct, buried in reasoning text",
             f"Separating variables gives X''/X = T'/(kT) ...\nSo the answer is\nu(x, t) = {s(u)}"),
            ("forgot the time decay  (u = starting shape)", f"u(x, t) = {s(f)}"),
            ("decays at the wrong rate  (rate doubled)",
             f"u(x, t) = {s(u.subs(t, 2 * t))}"),
            ("habitual sin(n*pi*x/L) wave",
             f"u(x, t) = {s(sp.exp(-k * (sp.pi / L) ** 2 * t) * sp.sin(sp.pi * x / L))}"),
            ("cos instead of sin", f"u(x, t) = {s(u.replace(sp.sin, sp.cos))}"),
            ("amplitude off by 10%", f"u(x, t) = {s(sp.Rational(11, 10) * u)}"),
        ]
    else:
        # The parabola: answer is an infinite series, the checker wants a good truncation.
        c = [true_coeff(p, n) for n in range(1, 8)]          # exact c_1 .. c_7
        odd = [ci if n % 2 else 0 for n, ci in enumerate(c, start=1)]
        three = odd[:5]                                       # n = 1, 3, 5
        out += [
            ("correct: 3 odd terms, exact", f"u(x, t) = {s(series(p, three))}"),
            ("correct: 3 odd terms, decimals to 4 s.f.",
             f"u(x, t) = {s(series(p, [sp.N(v, 4) for v in three]))}"),
            ("correct: 4 odd terms", f"u(x, t) = {s(series(p, odd[:7]))}"),
            ("too few: 1 term", f"u(x, t) = {s(series(p, odd[:1]))}"),
            ("too few: 2 terms", f"u(x, t) = {s(series(p, odd[:3]))}"),
            ("coefficients doubled", f"u(x, t) = {s(series(p, [2 * v for v in three]))}"),
            ("wrong sign on c_3", f"u(x, t) = {s(series(p, [three[0], 0, -three[2], 0, three[4]]))}"),
            ("the guess Qwen made: 1.6, -0.8, 0.4, -0.2, 0.1",
             f"u(x, t) = {s(series(p, [1.6, -0.8, 0.4, -0.2, 0.1]))}"),
            ("right coefficients, WRONG decay (exponent uses n*pi, sine uses n*pi/L)",
             f"u(x, t) = {s(series(p, three, freq_override=lambda n: n * sp.pi))}"),
            ("forgot the time decay  (u = starting shape)", f"u(x, t) = {s(f)}"),
        ]

    # Formatting mistakes the README says the model really made.
    out += [
        ("LaTeX sum  (unreadable)", r"u(x, t) = \sum_{n=1}^\infty A_n e^{-k n^2 t} \sin(n \pi x)"),
        ("unknown constant left in", "u(x, t) = A*exp(-2*pi**2*t)*sin(pi*x/2)"),
        ("product notation", "u(x, t) = X(x)*T(t)"),
        ("no answer line at all", "The temperature decays exponentially over time."),
    ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", type=int, default=1)
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reveal", action="store_true",
                    help="set REVEAL_COEFFICIENTS=True to see the feedback the model used to copy")
    ap.add_argument("--answer", action="append",
                    help="grade your own answer instead of the gallery (may repeat)")
    a = ap.parse_args()

    pdecheck.REVEAL_COEFFICIENTS = a.reveal
    p = problem(a.level, a.sub, a.seed)
    print(describe(p))
    print("Score = 0.4 equation + 0.2 left end + 0.2 right end + 0.2 starting shape")

    items = [(f"your answer #{i + 1}", ans) for i, ans in enumerate(a.answer)] if a.answer \
        else gallery(p)
    rewards = [show(label, p, ans)["reward"] for label, ans in items]
    print(f"\n{len(rewards)} answers graded; {sum(r == 1.0 for r in rewards)} scored 1.0")


if __name__ == "__main__":
    main()
