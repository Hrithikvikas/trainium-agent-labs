#!/usr/bin/env python3
"""
exp4_parser_edge_cases.py - does the checker read every CORRECT answer correctly?

Every reply below contains a mathematically correct answer to level 1.1 (insulated end),
just written the way a model might write it: with ^ for powers, with the pi symbol, inside
backticks, with a <think> block, with a restated line at the end, and so on.

A reward function that marks a correct answer as wrong teaches the model the wrong lesson,
so any line below that does not score 1.0 is a weakness in the checker worth fixing (or a
formatting rule worth putting in the prompt).

    python exp4_parser_edge_cases.py
"""

from common import problem, describe, grade, sp


def main():
    p = problem(1, 1, 0)
    print(describe(p))
    good = sp.sstr(p["exact"])
    print(f"correct answer: u(x, t) = {good}\n")
    # For seed 0 this is exp(-pi**2*t/4)*sin(pi*x/2).

    cases = [
        # (label, reply text, what a fair checker should give)
        ("baseline, sympy style", f"u(x, t) = {good}", 1.0),
        ("no spaces", "u(x,t)=exp(-pi**2*t/4)*sin(pi*x/2)", 1.0),
        ("^ for powers", "u(x, t) = exp(-pi^2*t/4)*sin(pi*x/2)", 1.0),
        ("pi symbol", "u(x, t) = exp(-π**2*t/4)*sin(π*x/2)", 1.0),
        ("LaTeX \\pi", "u(x, t) = exp(-\\pi**2*t/4)*sin(\\pi*x/2)", 1.0),
        ("implicit multiplication", "u(x, t) = exp(-pi**2 t/4) sin(pi x/2)", 1.0),
        ("trailing full stop", "u(x, t) = exp(-pi**2*t/4)*sin(pi*x/2).", 1.0),
        ("in backticks", "u(x, t) = `exp(-pi**2*t/4)*sin(pi*x/2)`", 1.0),
        ("in $ $", "u(x, t) = $exp(-pi**2*t/4)*sin(pi*x/2)$", 1.0),
        ("decimal for pi^2/4", "u(x, t) = exp(-2.4674011*t)*sin(pi*x/2)", 1.0),
        ("e**(...) instead of exp", "u(x, t) = E**(-pi**2*t/4)*sin(pi*x/2)", 1.0),
        ("e^(...) with lowercase e", "u(x, t) = e^(-pi^2*t/4)*sin(pi*x/2)", 1.0),
        ("np.exp / np.sin", "u(x, t) = np.exp(-np.pi**2*t/4)*np.sin(np.pi*x/2)", 1.0),
        ("math.exp", "u(x, t) = math.exp(-math.pi**2*t/4)*math.sin(math.pi*x/2)", 1.0),
        ("inside <think> then answer",
         "<think>try u(x, t) = sin(pi*x)</think>\nu(x, t) = exp(-pi**2*t/4)*sin(pi*x/2)", 1.0),
        ("bold markdown answer", "**u(x, t) = exp(-pi**2*t/4)*sin(pi*x/2)**", 1.0),
        ("answer, then a restated wrong line",
         "u(x, t) = exp(-pi**2*t/4)*sin(pi*x/2)\n\nCheck: at t=0, u(x, t) = sin(pi*x/2)", 1.0),
        ("answer, then a trailing comment",
         "u(x, t) = exp(-pi**2*t/4)*sin(pi*x/2)   # decays slowly", 1.0),
        ("wrapped onto two lines",
         "u(x, t) = exp(-pi**2*t/4)*\n          sin(pi*x/2)", 1.0),
        ("Unicode minus sign", "u(x, t) = exp(−pi**2*t/4)*sin(pi*x/2)", 1.0),
    ]

    unfair = []
    print(f"{'reward':>6}  {'fair':>4}  case")
    for label, reply, want in cases:
        r = grade(p, reply)
        flag = "" if r["reward"] == want else "   <-- checker gets this wrong"
        if flag:
            unfair.append(label)
        print(f"{r['reward']:6.1f}  {want:4.1f}  {label}{flag}")
        if flag:
            print(f"              read as: {r['expr']!r}")
            print(f"              feedback: {r['feedback'][:150]}")

    print(f"\n{len(cases) - len(unfair)}/{len(cases)} correct answers were scored correctly.")
    if unfair:
        print("Correct answers the checker rejects (ideas for improving extract()/parse() in "
              "pdecheck.py):")
        for u in unfair:
            print(f"  - {u}")


if __name__ == "__main__":
    main()
