# Checker experiments for Project 1 (heat rod)

These scripts test `pdecheck.py`, the checker that grades the model's answers. They do not
need the model or the vLLM server: each one writes candidate answers by hand and looks at
the score and feedback the checker gives back.

## Setup

Copy this `experiments/` folder into `projects/01-heat-rod-pde/` (next to `pdecheck.py`):

```bash
cd /workspace/projects/01-heat-rod-pde/experiments
python exp1_grade_answers.py
```

If you keep the folder somewhere else, point it at the project folder first:

```bash
export HEATROD_DIR=/workspace/projects/01-heat-rod-pde
```

Needs only `sympy` and `numpy`, which are already in the pod.

## The experiments

| Script | Question it answers | Run time |
|---|---|---|
| `exp1_grade_answers.py` | What score and feedback does each kind of mistake get? | ~3 s |
| `exp2_terms_and_tolerance.py` | How many series terms does the parabola need, and how does the 0.5% tolerance decide that? | ~5 s |
| `exp3_coefficient_noise.py` | How accurate must the model's numbers be to still pass? | ~35 s |
| `exp4_parser_edge_cases.py` | Does the checker accept every correct answer, however it is written? | ~2 s |

Useful options:

```bash
python exp1_grade_answers.py --sub 1                 # insulated-end problem instead of the parabola
python exp1_grade_answers.py --reveal                # feedback with the target numbers printed
python exp1_grade_answers.py --answer "u(x, t) = 1.03*exp(-pi**2*t/2)*sin(pi*x/2)"
python exp2_terms_and_tolerance.py --tols 0.02 0.005 0.001
python exp3_coefficient_noise.py --terms 5 --trials 200
```

`common.py` holds the shared helpers: building a problem, computing the true coefficients,
building a series answer, and printing a graded result.
