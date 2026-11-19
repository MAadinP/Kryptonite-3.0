"""Figures about the datasets themselves (no models are trained here). Each function returns a figure."""
import numpy as np
import matplotlib.pyplot as plt

from .config import NAMES, COLORS, MARKERS
from .data import load, load_hidden, to_sign
from .figures import COLUMN, label, one_panel, four_panels
from .interactions import label_corr, noise_level, strongest, pair_matrix

ORDER_NAMES = {1: "single features", 2: "pairs", 3: "triples"}
ORDER_COLORS = {1: "0.7", 2: "0.4", 3: "0.0"}  # light to dark
ORDER_MARKERS = {1: "o", 2: "s", 3: "^"}


# ----------------------------------------------------------------------------- basics
def plot_sizes():
    fig, ax = one_panel()
    x = np.arange(len(NAMES))
    labelled = ax.bar(x - 0.2, [len(load(n)[1]) for n in NAMES], width=0.4, color="0.3", label="labelled")
    hidden = ax.bar(x + 0.2, [len(load_hidden(n)) for n in NAMES], width=0.4, color="0.7", label="hidden")
    ax.bar_label(labelled, fontsize=7)
    ax.bar_label(hidden, fontsize=7)
    ax.set_xticks(x, [label(n) for n in NAMES])
    ax.set_ylabel("Rows")
    ax.set_ylim(0, 37000)
    ax.legend(frameon=False, loc="upper left")
    return fig


def plot_class_balance():
    fig, ax = one_panel()
    for k, n in enumerate(NAMES):
        y = load(n)[1]
        ax.errorbar(k, y.mean(), yerr=2 * np.sqrt(0.25 / len(y)), fmt=MARKERS[n], color=COLORS[n], capsize=3)
    ax.axhline(0.5, color="0.5", ls="--", lw=1)
    ax.set_xticks(range(len(NAMES)), [label(n) for n in NAMES])
    ax.set_ylabel("Share with $y = 1$")
    ax.set_ylim(0.46, 0.54)
    return fig


def plot_value_hist():
    fig, ax = one_panel()
    for n in NAMES:
        ax.hist(load(n)[0].ravel(), bins=150, density=True, histtype="step", color=COLORS[n], label=label(n))
    ax.set_xlabel("Feature value $x_j$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, loc="upper center", fontsize=8)
    return fig


def plot_noise_hist():
    e = np.concatenate([(X - to_sign(X)).ravel() for X in (load(n)[0] for n in NAMES)])
    sd = e.std()
    t = np.linspace(-0.5, 0.5, 300)
    fig, ax = one_panel()
    ax.hist(e, bins=150, density=True, color="0.75", label="all datasets")
    ax.plot(t, np.exp(-t**2 / (2 * sd**2)) / (sd * np.sqrt(2 * np.pi)), color="black", lw=1.2,
            label=f"Gaussian, $\\sigma = {sd:.3f}$")
    ax.set_xlabel("$x_j - \\mathrm{sign}(x_j)$")
    ax.set_ylabel("Density")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return fig


def plot_rows(name=20, k=20):
    X, y = load(name)
    rows = np.vstack([X[y == 0][:k], X[y == 1][:k]])
    fig, ax = one_panel(height=2.6)
    im = ax.imshow(rows, cmap="coolwarm", vmin=-1.5, vmax=1.5, aspect="auto", interpolation="none")
    ax.axhline(k - 0.5, color="black", lw=1)
    ax.set_yticks([k / 2 - 0.5, 1.5 * k - 0.5], ["$y = 0$", "$y = 1$"])
    ax.set_xticks(range(0, X.shape[1], 5))
    ax.set_xlabel("Feature $j$")
    fig.colorbar(im, ax=ax, label="$x_j$")
    return fig


# ----------------------------------------------------------------------------- features on their own
def plot_positive_share():
    fig, axes = four_panels(sharey=True)
    for ax, n in zip(axes, NAMES):
        X = load(n)[0]
        band = 2 * np.sqrt(0.25 / len(X))
        ax.axhspan(0.5 - band, 0.5 + band, color="0.9", lw=0)
        ax.plot((X > 0).mean(axis=0), MARKERS[n], ms=3, color=COLORS[n])
        ax.set_xlabel("Feature $j$")
    axes[0].set_ylabel("$P(x_j > 0)$")
    axes[0].set_ylim(0.46, 0.54)
    return fig


def heatmaps(matrices, colorbar_label, lim=0.15):
    fig, axes = four_panels(height=1.95)
    for ax, M in zip(axes, matrices):
        im = ax.imshow(M, cmap="coolwarm", vmin=-lim, vmax=lim, interpolation="none")
        ax.set_xticks(range(0, len(M), 10))
        ax.set_yticks(range(0, len(M), 10))
        ax.set_xlabel("Feature $i$")
    axes[0].set_ylabel("Feature $j$")
    fig.colorbar(im, ax=axes, shrink=0.85, label=colorbar_label)
    return fig


def plot_feature_corr():
    mats = []
    for n in NAMES:
        C = np.corrcoef(to_sign(load(n)[0]).T)
        np.fill_diagonal(C, 0)  # the diagonal is always 1
        mats.append(C)
    return heatmaps(mats, "corr$(s_i, s_j)$")


def plot_train_vs_hidden():
    fig, axes = four_panels(sharey=True)
    for ax, n in zip(axes, NAMES):
        X, H = load(n)[0], load_hidden(n)
        diff = (X > 0).mean(axis=0) - (H > 0).mean(axis=0)
        band = 2 * np.sqrt(0.25 / len(X) + 0.25 / len(H))
        ax.axhspan(-band, band, color="0.9", lw=0)
        ax.bar(range(len(diff)), diff, width=0.7, color=COLORS[n])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("Feature $j$")
    axes[0].set_ylabel("$P(x_j > 0)$\nlabelled $-$ hidden")
    axes[0].set_ylim(-0.04, 0.04)
    return fig


# ----------------------------------------------------------------------------- features vs label
def plot_single_corr():
    fig, axes = four_panels(sharey=True)
    for ax, n in zip(axes, NAMES):
        c = list(label_corr(n, 1).values())
        noise = noise_level(n)
        ax.axhspan(-noise, noise, color="0.9", lw=0)
        ax.bar(range(len(c)), c, width=0.7, color=COLORS[n])
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xlabel("Feature $j$")
    axes[0].set_ylabel("corr$(s_j, y)$")
    axes[0].set_ylim(-0.15, 0.15)
    return fig


def plot_pair_corr():
    return heatmaps([pair_matrix(n) for n in NAMES], "corr$(s_i s_j, y)$")


def plot_corr_by_order():
    fig, ax = one_panel()
    orders = [1, 2, 3]
    for n in NAMES:
        real = [max(map(abs, label_corr(n, k).values())) for k in orders]
        fake = [max(map(abs, label_corr(n, k, shuffle=True).values())) for k in orders]
        ax.plot(orders, real, marker=MARKERS[n], ms=4, color=COLORS[n], label=label(n))
        ax.plot(orders, fake, ls=":", lw=1, color=COLORS[n])
    ax.plot([], [], ls=":", lw=1, color="0.4", label="shuffled $y$")
    ax.set_xticks(orders, ["single", "pairs", "triples"])
    ax.set_ylabel("Largest |corr| with $y$")
    ax.set_ylim(0, 0.2)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    return fig


def plot_corr_ranked(k=15):
    fig, axes = four_panels(sharey=True)
    ranks = range(1, k + 1)
    for ax, n in zip(axes, NAMES):
        for order in (1, 2, 3):
            top = sorted(map(abs, label_corr(n, order).values()), reverse=True)[:k]
            ax.plot(ranks, top, marker=ORDER_MARKERS[order], ms=3, color=ORDER_COLORS[order],
                    label=ORDER_NAMES[order])
        fake = sorted(map(abs, label_corr(n, 3, shuffle=True).values()), reverse=True)[:k]
        ax.plot(ranks, fake, ls=":", color="0.4", label="triples, shuffled $y$")
        ax.set_xlabel("Rank")
    axes[0].set_ylabel("|corr| with $y$")
    axes[0].set_ylim(0, 0.2)
    axes[-1].legend(frameon=False, fontsize=7, loc="upper right")
    return fig


def plot_corr_hist(name=40):
    fig, ax = one_panel()
    bins = np.linspace(-0.15, 0.15, 121)
    ax.hist(list(label_corr(name, 3, shuffle=True).values()), bins=bins, color="0.88",
            label="triples, shuffled $y$")
    for order in (2, 3):
        ax.hist(list(label_corr(name, order).values()), bins=bins, histtype="step",
                color=ORDER_COLORS[order], label=ORDER_NAMES[order])
    ax.set_yscale("log")
    ax.set_xlabel(f"corr with $y$ ({label(name)})")
    ax.set_ylabel("Count")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    return fig


def plot_top_pair_quadrants():
    fig, axes = four_panels(height=1.9)
    for ax, n in zip(axes, NAMES):
        X, y = load(n)
        S = to_sign(X)
        i, j = strongest(n, 2)[0]
        P = np.array([[y[(S[:, i] == a) & (S[:, j] == b)].mean() for b in (-1, 1)] for a in (-1, 1)])
        im = ax.imshow(P, cmap="coolwarm", vmin=0.3, vmax=0.7)
        for a in range(2):
            for b in range(2):
                ax.text(b, a, f"{P[a, b]:.2f}", ha="center", va="center", fontsize=9)
        ax.set_xticks([0, 1], ["$-1$", "$+1$"])
        ax.set_yticks([0, 1], ["$-1$", "$+1$"])
        ax.set_xlabel(f"sign $x_{{{j}}}$")
        ax.set_ylabel(f"sign $x_{{{i}}}$")
    fig.colorbar(im, ax=axes, shrink=0.85, label="$P(y = 1)$")
    return fig


# ----------------------------------------------------------------------------- Kryptonite-40 vs 40-adapt
def plot_40_vs_adapt():
    fig, ax = plt.subplots(figsize=(COLUMN, 3.0), layout="constrained")
    for order in (2, 3):
        a, b = label_corr(40, order), label_corr("40-adapt", order)
        ax.scatter(list(a.values()), [b[k] for k in a], s=4, alpha=0.5, edgecolors="none",
                   color=ORDER_COLORS[order], label=ORDER_NAMES[order])
    for name in (40, "40-adapt"):
        top = strongest(name, 2) + strongest(name, 3, k=2)
        xs = [label_corr(40, len(idx))[idx] for idx in top]
        ys = [label_corr("40-adapt", len(idx))[idx] for idx in top]
        ax.scatter(xs, ys, s=30, marker=MARKERS[name], color=COLORS[name], zorder=3,
                   label=f"strongest in {label(name)}")
        i, j = top[0]
        ax.annotate(f"$s_{{{i}}} s_{{{j}}}$", (xs[0], ys[0]), xytext=(6, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="0.8", lw=0.6, zorder=0)
    ax.axvline(0, color="0.8", lw=0.6, zorder=0)
    ax.set_xlim(-0.16, 0.16)
    ax.set_ylim(-0.16, 0.16)
    ax.set_aspect("equal")
    ax.set_xlabel("corr with $y$ in Kryptonite-40")
    ax.set_ylabel("corr with $y$ in 40-adapt")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    return fig


def plot_40_vs_adapt_pairs():
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 1.9), layout="constrained")
    for ax, name in zip(axes, (40, "40-adapt")):
        im = ax.imshow(pair_matrix(name), cmap="coolwarm", vmin=-0.15, vmax=0.15, interpolation="none")
        ax.set_title(label(name))
        ax.set_xticks(range(0, 40, 10))
        ax.set_yticks(range(0, 40, 10))
    fig.colorbar(im, ax=axes, shrink=0.85, label="corr$(s_i s_j, y)$")
    return fig
