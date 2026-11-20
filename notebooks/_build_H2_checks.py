"""Builds notebooks/H2_checks.ipynb (run once; the notebook is the deliverable)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md(r"""# H2 checks: numbers in the report that `H2_more_data.ipynb` does not print

No MLP is trained here.

1. **Convergence.** Quadratic LR is refitted on exactly the training subsets of `H2_more_data` (same seeds, same
   nested subsets, same $C$) to record the L-BFGS iterations used and whether scikit-learn warned about convergence.
   Its test accuracies are checked against the saved learning curves.
2. **Correlation by degree** (the numbers behind the `data_corr_by_order` figure). On $\pm1$ inputs, with
   $t = 2y-1$, $c_S = \mathrm{mean}(t\prod_{j\in S}s_j)$ is the correlation of the product with the label, its
   least-squares weight, and $(1+|c_S|)/2$ is the accuracy of predicting $y$ from that product alone. We compare the
   largest $|c_S|$ of each degree with the same statistic after shuffling $y$ (a pure-noise control)."""),
code(r"""import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from include.config import NS, SEEDS
from include.data import load, split
from include.interactions import label_corr
from include.tracking import track, save_json, load_json

h1, h2 = load_json("H1_hparams.json"), load_json("H2_more_data.json")
assert h1 is not None and h2 is not None, "run H1_learnability.ipynb and H2_more_data.ipynb first"
FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)  # as in H2_more_data
checks = {}"""),
md(r"""## 1. Convergence of quadratic LR at every training size"""),
code(r"""with track("H2_checks_convergence"):
    for n in NS:
        X, y = load(n)
        C = h1[str(n)]["LogReg (d=2)"]["C"]
        iters, converged, matches = {}, True, True
        for seed in SEEDS:
            X_tr, y_tr, X_va, y_va, X_te, y_te = split(X, y, seed)
            order = np.random.default_rng(seed).permutation(len(y_tr))
            for f in FRACTIONS:
                keep = order[:int(f * len(y_tr))]
                poly = PolynomialFeatures(2, include_bias=False)
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    clf = LogisticRegression(C=C, max_iter=5000).fit(poly.fit_transform(X_tr[keep]), y_tr[keep])
                converged &= not any(issubclass(x.category, ConvergenceWarning) for x in w)
                iters.setdefault(str(f), []).append(int(clf.n_iter_[0]))
                acc = clf.score(poly.transform(X_te), y_te)
                matches &= acc == h2["curves"][str(n)]["quad. LR"][str(f)][SEEDS.index(seed)]
        all_iters = [i for v in iters.values() for i in v]
        checks[n] = {"iterations": iters, "all_converged": bool(converged), "matches_saved_curves": bool(matches)}
        print(f"n={n}: iterations {min(all_iters)}-{max(all_iters)} of 5000, all converged: {converged}, "
              f"test accuracies identical to H2_more_data: {matches}")"""),
md(r"""## 2. Correlation of each degree with the label, against shuffled labels

Computed on all labelled rows, exactly as in the `data_corr_by_order` figure."""),
code(r"""with track("H2_checks_correlation"):
    for n in NS:
        N = len(load(n)[1])
        checks[n]["corr"] = {}
        print(f"n={n} (N={N}, 2/sqrt(N) = {2 / np.sqrt(N):.3f})")
        for order, name in ((1, "single"), (2, "pairs"), (3, "triples")):
            real = np.abs(np.array(list(label_corr(n, order).values())))
            fake = np.abs(np.array(list(label_corr(n, order, shuffle=True).values())))
            above = int((real > fake.max()).sum())
            checks[n]["corr"][name] = {"max": float(real.max()), "max_shuffled": float(fake.max()),
                                       "n_above_shuffled_max": above, "n_terms": len(real),
                                       "accuracy_of_best_term": float((1 + real.max()) / 2)}
            print(f"    {name:<8} max |corr| {real.max():.3f}  shuffled {fake.max():.3f}  "
                  f"above shuffled max: {above}/{len(real)}  best-term accuracy {(1 + real.max()) / 2:.3f}")

save_json({str(n): c for n, c in checks.items()}, "H2_checks.json")"""),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("H2_checks.ipynb"))
print("written")
