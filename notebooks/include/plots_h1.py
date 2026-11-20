"""Figures for the H1 notebook. They take the `summary` / `results` dicts that notebook builds."""
import numpy as np
import matplotlib.pyplot as plt

from .config import NS, COLORS, MARKERS, TARGET
from .figures import COLUMN, PAGE, label

SHORT_NAMES = {"LogReg (d=1)": "LR\n$d{=}1$", "LogReg (d=2)": "LR\n$d{=}2$",
               "MLP (raw x)": "MLP\nraw", "MLP (sign x)": "MLP\nsign"}


def plot_h1_models(summary, models):
    # seed means with +-eps (Chebyshev, union bound over the models)
    fig, ax = plt.subplots(figsize=(COLUMN, 1.95), layout="constrained")
    x = np.arange(len(models))
    for k, n in enumerate(NS):
        means = [summary[n][m]["mean"] for m in models]
        ax.errorbar(x + (k - 1) * 0.2, means, yerr=summary[n]["eps"], fmt=MARKERS[n], ms=4, capsize=2,
                    color=COLORS[n], label=label(n))
    ax.axhline(TARGET, color="0.4", ls="--", lw=1)
    ax.set_xticks(x, [SHORT_NAMES[m] for m in models])
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.1)
    return fig


def plot_h1_seeds(results, models):
    # every seed's test accuracy, so the spread between seeds is visible
    fig, ax = plt.subplots(figsize=(COLUMN, 2.5), layout="constrained")
    for k, n in enumerate(NS):
        for i, m in enumerate(models):
            acc = results[n][m]["test"]
            ax.plot(np.full(len(acc), i + (k - 1) * 0.2), acc, MARKERS[n], ms=3, alpha=0.7, color=COLORS[n],
                    label=label(n) if i == 0 else None)
    ax.axhline(TARGET, color="0.4", ls="--", lw=1)
    ax.set_xticks(range(len(models)), [SHORT_NAMES[m] for m in models])
    ax.set_ylabel("Test accuracy (each seed)")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.1)
    return fig


# ----------------------------------------------------------------------------- H1 supporting analyses
def plot_influence(influence):
    # influence[n]: how often flipping feature j changes the MLP's prediction (mean over seeds)
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 1.9), layout="constrained", sharey=True)
    for ax, n in zip(axes, NS):
        ax.bar(range(len(influence[n])), influence[n], width=0.7, color=COLORS[n])
        ax.set_title(label(n))
        ax.set_xlabel("Feature $j$")
    axes[0].set_ylabel("P(prediction flips)")
    axes[0].set_ylim(0, 1)
    return fig


def plot_influence_column(influence):
    # one-column version of plot_influence: one row per dataset, the dataset named inside each panel
    fig, axes = plt.subplots(3, 1, figsize=(COLUMN, 2.7), layout="constrained", sharey=True)
    for ax, n in zip(axes, NS):
        ax.bar(range(len(influence[n])), influence[n], width=0.7, color=COLORS[n])
        ax.text(0.99, 0.95, label(n), transform=ax.transAxes, ha="right", va="top")
        ax.set_xlim(-0.8, len(influence[n]) - 0.2)
        ax.set_ylim(0, 1)
        ax.set_yticks([0, 0.5, 1])
    axes[-1].set_xlabel("Feature $j$")
    fig.supylabel("P(prediction flips)", fontsize=10)
    return fig


def plot_topk(res):
    # test accuracy against the number of top features k: the paper's polynomial LR with every
    # product of the k features (d = k), and the lookup table (dotted)
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9), layout="constrained")
    for n in NS:
        ks = sorted(res[n]["lookup"])
        poly_ks = [k for k in ks if f"{k},{k}" in res[n]["poly"]]
        ax.plot(poly_ks, [np.mean(res[n]["poly"][f"{k},{k}"]["test"]) for k in poly_ks],
                marker=MARKERS[n], ms=4, color=COLORS[n], label=label(n))
        ax.plot(ks, [np.mean(res[n]["lookup"][k]["test"]) for k in ks], ls=":", color=COLORS[n])
    ax.plot([], [], ls=":", color="0.4", label="lookup table")
    ax.axhline(TARGET, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Top features used, $k$")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.45, 1.22); ax.set_yticks([0.6, 0.8, 1.0])
    ax.legend(frameon=False, loc="upper left", ncol=2, columnspacing=0.8, handlelength=1.5)
    return fig


def plot_h1_summary(res, k_for, top=14):
    # the three H1 analyses in one page-wide strip (the report's figure):
    # (a) flip influence sorted from largest to smallest (mean over seeds), first `top` ranks
    # (b) the paper's polynomial LR on the top k features with all products (d = k)
    # (c) the paper's polynomial LR on the top k_for[n] features against its degree d
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 1.85), layout="constrained")
    ax = axes[0]
    for n in NS:
        inf = np.sort(np.mean(res[n]["influence"], axis=0))[::-1][:top]
        ax.plot(range(1, top + 1), inf, marker=MARKERS[n], ms=3.5, color=COLORS[n], label=label(n))
    ax.axvline(10.5, color="0.4", ls=":", lw=1)
    ax.set_xlabel("Feature rank")
    ax.set_ylabel("P(prediction flips)")
    ax.set_ylim(0, 1)
    ax.set_xticks([1, 5, 10, top])
    ax.legend(frameon=False, loc="lower left", handletextpad=0.2, borderaxespad=0.1)

    ax = axes[1]
    for n in NS:
        ks = [k for k in sorted(res[n]["lookup"]) if f"{k},{k}" in res[n]["poly"]]
        ax.plot(ks, [np.mean(res[n]["poly"][f"{k},{k}"]["test"]) for k in ks], marker=MARKERS[n], ms=3.5,
                color=COLORS[n])
    ax.set_xlabel("Top features used, $k$")
    ax.set_xticks([2, 4, 6, 8, 10, 12])

    ax = axes[2]
    for n in NS:
        k = k_for[n]
        ds = sorted(int(key.split(",")[1]) for key in res[n]["poly"] if int(key.split(",")[0]) == k)
        ax.plot(ds, [np.mean(res[n]["poly"][f"{k},{d}"]["test"]) for d in ds], marker=MARKERS[n], ms=3.5,
                color=COLORS[n])
    ax.set_xlabel(f"Degree $d$ (top $k={k_for[NS[0]]}$)")
    ax.set_xticks([1, 2, 4, 6, 8, 10])

    for ax in axes[1:]:
        ax.axhline(TARGET, color="0.4", ls="--", lw=1)
        ax.set_ylim(0.45, 1.0)
        ax.set_ylabel("Test accuracy")
    return fig


def plot_poly_degree(res, k_for):
    # test accuracy of the paper's polynomial LR on the top k features, against its degree d
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9), layout="constrained")
    for n in NS:
        k = k_for[n]
        ds = sorted(int(key.split(",")[1]) for key in res[n]["poly"] if int(key.split(",")[0]) == k)
        ax.plot(ds, [np.mean(res[n]["poly"][f"{k},{d}"]["test"]) for d in ds], marker=MARKERS[n], ms=4,
                color=COLORS[n], label=f"{label(n)}, $k={k}$")
    ax.axhline(TARGET, color="0.4", ls="--", lw=1)
    ax.set_xlabel("Polynomial degree $d$")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.legend(frameon=False, loc="lower right")
    return fig
