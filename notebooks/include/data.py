"""Loading and splitting the Kryptonite datasets."""
import numpy as np
from sklearn.model_selection import train_test_split

from .config import DATA


def load(name):
    # e.g. load(20) or load("40-adapt")
    X = np.load(DATA / f"kryptonite-{name}-X.npy").astype(np.float32)
    y = np.load(DATA / f"kryptonite-{name}-y.npy").astype(np.int64)
    return X, y


def load_hidden(name):
    return np.load(DATA / f"hidden-kryptonite-{name}-X.npy").astype(np.float32)


def to_sign(X):
    # each feature is +-1 plus a little Gaussian noise, so taking the sign removes the noise
    return np.sign(X).astype(np.float32)


def split(X, y, seed):
    # stratified 60/20/20 train/val/test
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tr, y_tr, test_size=0.25, random_state=seed, stratify=y_tr)
    return X_tr, y_tr, X_va, y_va, X_te, y_te
