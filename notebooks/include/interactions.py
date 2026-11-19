"""How strongly single features, pairs and triples of features go with the label.

With signs s_j = +-1 and t = 2y - 1, c = mean(s_i * s_j * ... * t) is (almost exactly) the
correlation between that product and the label, and (1 + |c|) / 2 is the accuracy you would
get by predicting y from that one product.
"""
from functools import lru_cache
from itertools import combinations

import numpy as np

from .data import load, to_sign


@lru_cache(maxsize=None)
def label_corr(name, order, shuffle=False):
    # {(i, j, ...): c} for every set of `order` features. shuffle=True breaks the link to y,
    # which shows how big c gets by chance.
    X, y = load(name)
    if shuffle:
        y = np.random.default_rng(0).permutation(y)
    S = to_sign(X)
    t = 2 * y - 1
    return {idx: float(np.mean(S[:, list(idx)].prod(axis=1) * t))
            for idx in combinations(range(S.shape[1]), order)}


def noise_level(name):
    # two standard deviations of c when there is no relationship at all
    return 2 / np.sqrt(len(load(name)[1]))


def strongest(name, order, k=1):
    c = label_corr(name, order)
    return sorted(c, key=lambda idx: -abs(c[idx]))[:k]


def pair_matrix(name):
    n = load(name)[0].shape[1]
    M = np.zeros((n, n))
    for (i, j), c in label_corr(name, 2).items():
        M[i, j] = M[j, i] = c
    return M
