"""Ranking features by how much an MLP relies on them, and simple models on the top k features."""
from math import comb

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

from .models import predict_logits


def flip_influence(model, S):
    # how often the MLP's prediction changes when feature j is flipped (S holds +-1 signs)
    pred = predict_logits(model, S) > 0
    influence = []
    for j in range(S.shape[1]):
        F = S.copy()
        F[:, j] *= -1
        influence.append(float(((predict_logits(model, F) > 0) != pred).mean()))
    return np.array(influence)


def n_terms(k, degree):
    # number of products of 1..degree distinct features out of k
    return sum(comb(k, i) for i in range(1, degree + 1))


def fit_poly(S_tr, y_tr, features, degree, C=1.0):
    # the paper's polynomial logistic regression, on the chosen features only.
    # with +-1 inputs x_j^2 = 1, so only products of distinct features (multilinear terms) matter
    poly = PolynomialFeatures(degree, interaction_only=True, include_bias=False)
    clf = LogisticRegression(C=C, max_iter=5000).fit(poly.fit_transform(S_tr[:, features]), y_tr)
    return lambda S: clf.predict(poly.transform(S[:, features]))


def fit_lookup(S_tr, y_tr, features):
    # per-pattern Bernoulli MLE: for every sign pattern of the chosen features, predict the
    # label that is more common among training rows with that pattern
    powers = 2 ** np.arange(len(features))
    code = lambda S: (S[:, features] > 0).astype(int) @ powers
    c = code(S_tr)
    ones = np.bincount(c, weights=y_tr, minlength=2 ** len(features))
    total = np.bincount(c, minlength=2 ** len(features))
    p = np.where(total > 0, ones / np.maximum(total, 1), 0.5)
    return lambda S: (p[code(S)] > 0.5).astype(int)
