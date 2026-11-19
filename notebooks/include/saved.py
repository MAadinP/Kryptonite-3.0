"""Loading saved results back (with int / float keys again), so figures can be drawn without retraining."""
from .tracking import load_json


def _load(name):
    saved = load_json(name)
    assert saved is not None, f"results/{name} is missing: run the notebook that makes it first"
    return saved


def load_h1():
    # -> summary[n], results[n]  (from H1_learnability.ipynb)
    saved = _load("H1_learnability.json")
    return ({int(n): s for n, s in saved["summary"].items()},
            {int(n): r for n, r in saved["results"].items()})


def load_h1_features():
    # -> res[n], summary[n]  (from H1_features.ipynb)
    saved = _load("H1_features.json")
    res = {}
    for n, r in saved["results"].items():
        res[int(n)] = {**r, "lookup": {int(k): v for k, v in r["lookup"].items()}}
    return res, {int(n): s for n, s in saved["summary"].items()}


def load_h2():
    # -> curves[n][model][fraction], n_train[n]  (from H2_more_data.ipynb)
    saved = _load("H2_more_data.json")
    curves = {int(n): {m: {float(f): acc for f, acc in c.items()} for m, c in by_model.items()}
              for n, by_model in saved["curves"].items()}
    return curves, {int(n): v for n, v in saved["n_train"].items()}
