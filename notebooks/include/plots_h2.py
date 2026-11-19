"""Figures for the H2 notebook (more data)."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .config import NS, COLORS, MARKERS, TARGET
from .figures import COLUMN, label

STYLES = {"MLP": "-", "quad. LR": "--"}


def plot_learning_curves(curves, fractions, n_train):
    # curves[n][model][fraction] -> test accuracies over seeds; n_train[n] = size of the full training split
    fig, ax = plt.subplots(figsize=(COLUMN, 2.5), layout="constrained")
    for n in NS:
        x = [f * n_train[n] for f in fractions]
        for model, ls in STYLES.items():
            acc = [curves[n][model][f] for f in fractions]
            ax.errorbar(x, [np.mean(a) for a in acc], yerr=[np.std(a, ddof=1) for a in acc], ls=ls,
                        marker=MARKERS[n], ms=3, capsize=2, lw=1.2, color=COLORS[n])
    ax.axhline(TARGET, color="0.4", ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("Training points")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.4, 1.0)
    handles = [Line2D([], [], color=COLORS[n], marker=MARKERS[n], ms=3, label=label(n)) for n in NS]
    handles += [Line2D([], [], color="0.3", ls=ls, label=model) for model, ls in STYLES.items()]
    ax.legend(handles=handles, frameon=False, fontsize=7, ncol=2, loc="upper left")
    return fig
