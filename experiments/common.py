"""
common.py - shared helpers for the checker experiments.

These scripts test pdecheck.py (the checker) directly. No model and no server are needed:
we write candidate answers by hand and see what score and feedback the checker gives.

Where to put this folder
    Copy the whole `experiments/` folder into  projects/01-heat-rod-pde/
    so that pdecheck.py sits one level up. Or set HEATROD_DIR to that folder:
        export HEATROD_DIR=/workspace/projects/01-heat-rod-pde
"""

import os
import sys

import sympy as sp


def _find_repo():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.environ.get("HEATROD_DIR"), here, os.path.dirname(here)]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "pdecheck.py")):
            return c
    sys.exit("Could not find pdecheck.py. Put this folder inside projects/01-heat-rod-pde/ "
             "or set HEATROD_DIR to that folder.")


sys.path.insert(0, _find_repo())

import pdecheck                      # noqa: E402  (the checker under test)
import level0_heatrod                # noqa: E402
import level1_heatrod                # noqa: E402
from pdecheck import x, t            # noqa: E402

LEVELS = {0: level0_heatrod, 1: level1_heatrod}


def problem(level=1, sub=3, seed=0):
    """Build one problem exactly as agent.py would."""
    return LEVELS[level].make(sub, seed)


def describe(p):
    """One-line summary of a problem."""
    return (f"{p['name']} (seed {p['seed']}):  u_t = {sp.sstr(p['k'])}*u_xx on [0, {sp.sstr(p['L'])}],"
            f"  ends {p['left']}/{p['right']},  u(x,0) = {sp.sstr(p['f'])}")


def true_coeff(p, n):
    """The correct coefficient of the n-th allowed wave, computed exactly with sympy.
    This is the integral the model is supposed to do."""
    L = p["L"]
    return sp.simplify(2 / L * sp.integrate(p["f"] * p["basis"](n), (x, 0, L)))


def series(p, coeffs, freq_override=None):
    """Build u(x,t) = sum c_n * exp(-k * lam_n^2 * t) * basis_n(x).

    coeffs          list of coefficients for n = 1, 2, 3, ...
    freq_override   optional function n -> frequency to use IN THE EXPONENT ONLY,
                    to simulate the 'wrong decay rate' mistake.
    """
    k = p["k"]
    u = 0
    for i, c in enumerate(coeffs, start=1):
        if c == 0:
            continue
        lam = freq_override(i) if freq_override else p["lam"](i)
        u += c * sp.exp(-k * lam ** 2 * t) * p["basis"](i)
    return u


def grade(p, answer_text):
    """Run the real checker. answer_text is what the model would have written."""
    return pdecheck.check(p, answer_text)


def show(label, p, answer_text, width=110, feedback=True):
    """Grade one answer and print a readable report."""
    r = grade(p, answer_text)
    parts = r["parts"] or {}
    marks = "  ".join(f"{k}={'Y' if v else 'n'}" for k, v in parts.items()) or "(could not read)"
    err = r["start_error"]
    err_s = f"{err * 100:.3f}%" if isinstance(err, float) else "-"
    print(f"\n[{r['reward']:.1f}] {label}")
    shown = answer_text if len(answer_text) <= width else answer_text[:width - 3] + "..."
    print(f"      answer : {shown}")
    print(f"      parts  : {marks}    start error: {err_s}")
    if feedback and r["reward"] < 1.0:
        fb = r["feedback"]
        print(f"      checker: {fb[:400]}{'...' if len(fb) > 400 else ''}")
    return r
