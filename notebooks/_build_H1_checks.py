"""Builds notebooks/H1_checks.ipynb (run once; the notebook is the deliverable)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md(r"""# H1 checks: numbers in the report that the other H1 notebooks do not print

No MLP is trained here. The feature ranking for each seed is read from `results/H1_features.json`
(made by `H1_features.ipynb`), and every split is the same `split(X, y, seed)` used everywhere else.

1. **Rounding.** Does rounding inputs to $\operatorname{sign}(\mathbf{x})$ cause the result? We fit the paper's
   configuration (all inputs, $d=2$) and ours (top 10 features, $d$ chosen on validation) on both raw and rounded
   inputs.
2. **Convergence.** Did the paper's configuration converge, so that its failure is not an optimisation problem?
3. **Correlation screening.** Could low-order correlations alone have found the ten features? We keep every pair
   and triple whose $|\mathrm{corr}|$ with $y$ beats the largest value under shuffled labels (training split only),
   and compare the features they contain with the probe's top 10.
4. **Basis sizes** quoted in the report."""),
code(r"""import itertools
import warnings
from math import comb

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from include.config import NS, SEEDS, TARGET
from include.data import load, to_sign, split
from include.stats import chebyshev_eps, mean_std
from include.tracking import track, save_json, load_json

h1, hf = load_json("H1_hparams.json"), load_json("H1_features.json")
assert h1 is not None and hf is not None, "run H1_learnability.ipynb and H1_features.ipynb first"


def top10(n, seed):
    # the probe's ranking for this seed, saved by H1_features.ipynb
    return np.argsort(hf["results"][str(n)]["influence"][seed])[::-1][:10]


def chosen_d(n):
    return int(hf["summary"][str(n)]["poly"]["choice"].split(",")[1])


raw = lambda A: A
checks = {}"""),
md(r"""## 1. Rounding: raw $\mathbf{x}$ vs $\operatorname{sign}(\mathbf{x})$

The paper's configuration uses full `PolynomialFeatures(2)` with $C$ tuned on validation (from `H1_learnability`).
Ours uses products of distinct features of the top 10, degree as chosen in `H1_features`, $C=1$ (as there)."""),
code(r"""def paper_model(A_tr, y_tr, A_te, y_te, C):
    poly = PolynomialFeatures(2, include_bias=False)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        clf = LogisticRegression(C=C, max_iter=5000).fit(poly.fit_transform(A_tr), y_tr)
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    return clf.score(poly.transform(A_te), y_te), int(clf.n_iter_[0]), converged


def our_model(A_tr, y_tr, A_te, y_te, top, d):
    poly = PolynomialFeatures(d, interaction_only=True, include_bias=False)
    clf = LogisticRegression(C=1.0, max_iter=5000).fit(poly.fit_transform(A_tr[:, top]), y_tr)
    return clf.score(poly.transform(A_te[:, top]), y_te)


with track("H1_checks_rounding"):
    for n in NS:
        X, y = load(n)
        C = h1[str(n)]["LogReg (d=2)"]["C"]
        r = {k: [] for k in ("paper_raw", "paper_sign", "ours_raw", "ours_sign", "paper_n_iter", "paper_converged")}
        for seed in SEEDS:
            X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed)
            for name, f in (("raw", raw), ("sign", to_sign)):
                acc, it, ok = paper_model(f(X_tr), y_tr, f(X_te), y_te, C)
                r[f"paper_{name}"].append(acc)
                if name == "raw":
                    r["paper_n_iter"].append(it)
                    r["paper_converged"].append(ok)
                r[f"ours_{name}"].append(our_model(f(X_tr), y_tr, f(X_te), y_te, top10(n, seed), chosen_d(n)))
        eps = chebyshev_eps(len(y_te))
        checks[n] = {"rounding": r, "eps": eps}
        print(f"n={n} (eps {eps:.3f}, d={chosen_d(n)})")
        for k in ("paper_raw", "paper_sign", "ours_raw", "ours_sign"):
            mu, sd = mean_std(r[k])
            print(f"    {k:<11} {mu:.3f} +- {sd:.3f}   minus eps {mu - eps:.3f}  > 0.90: {mu - eps > TARGET}")"""),
md(r"""## 2. Convergence of the paper's configuration

L-BFGS iterations used (out of `max_iter=5000`) and whether scikit-learn raised a convergence warning. The loss is
convex, so a converged fit is the global optimum."""),
code(r"""for n in NS:
    r = checks[n]["rounding"]
    print(f"n={n}: iterations {r['paper_n_iter']} of 5000, all converged: {all(r['paper_converged'])}")"""),
md(r"""## 3. Correlation screening (training split only)

On $\pm1$ inputs, $\mathrm{corr} = \mathrm{mean}(t\prod_{j\in S}s_j)$ with $t=2y-1$. A subset $S$ is kept if its
$|\mathrm{corr}|$ is larger than the largest $|\mathrm{corr}|$ of any subset of the same size after shuffling $y$."""),
code(r"""with track("H1_checks_screening"):
    for n in NS:
        X, y = load(n)
        screen = []
        for seed in SEEDS:
            X_tr, y_tr, *_ = split(X, y, seed)
            S, t = to_sign(X_tr), 2 * y_tr - 1
            t_shuffled = np.random.default_rng(seed).permutation(t)
            found = set()
            for order in (2, 3):
                subsets = list(itertools.combinations(range(n), order))
                P = np.stack([S[:, list(c)].prod(1) for c in subsets], axis=1)
                corr = np.abs(P.T @ t) / len(t)
                threshold = (np.abs(P.T @ t_shuffled) / len(t)).max()
                found |= {j for c, v in zip(subsets, corr) if v > threshold for j in c}
            probe = set(top10(n, seed).tolist())
            screen.append({"found": sorted(found), "missing": sorted(probe - found), "extra": sorted(found - probe)})
            print(f"n={n} seed={seed}: missing {screen[-1]['missing']}, extra {screen[-1]['extra']}")
        checks[n]["screening"] = screen"""),
md(r"""## 4. Basis sizes quoted in the report

Products of distinct features up to degree $d$ (plus the constant) on $k$ features: $\sum_{i\le d}\binom{k}{i}$.
The paper's full `PolynomialFeatures(d)` on $n$ inputs has $\binom{n+d}{d}$ columns."""),
code(r"""sizes = {
    "ours_k10_d8": sum(comb(10, i) for i in range(9)),
    "ours_k10_d6": sum(comb(10, i) for i in range(7)),
    "paper_n40_d2": comb(42, 2),
    "paper_n40_d6": comb(46, 6),
    "paper_n20_d10": comb(30, 10),
    "paper_n40_d10": comb(50, 10),
}
for k, v in sizes.items():
    print(f"{k:<14} {v:,}")

save_json({"checks": {str(n): c for n, c in checks.items()}, "sizes": sizes}, "H1_checks.json")"""),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("H1_checks.ipynb"))
print("written")
