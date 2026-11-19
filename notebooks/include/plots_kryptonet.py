"""Figures about the pretrained KryptoNet-40, and what LoRA would cost on it."""
import numpy as np
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Patch

from .config import COLORS, MARKERS, TARGET
from .data import load
from .figures import COLUMN, label, one_panel
from .interactions import strongest
from .models import load_kryptonet, accuracy, predict_logits
from .stats import chebyshev_eps

SETS = (40, "40-adapt")
SET_NAMES = {40: "Kryptonite-40", "40-adapt": "40-adapt"}
LAYER_COLORS = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]  # hidden layers 1-4, light to dark


def linear_layers(model):
    return [m for m in model if isinstance(m, torch.nn.Linear)]


def plot_zero_shot():
    model = load_kryptonet()
    fig, ax = one_panel()
    for k, name in enumerate(SETS):
        X, y = load(name)
        acc, eps = accuracy(model, X, y), chebyshev_eps(len(y))
        ax.errorbar(k, acc, yerr=eps, fmt=MARKERS[name], color=COLORS[name], capsize=3)
        ax.annotate(f"{acc:.3f}", (k, acc + eps), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    ax.axhline(0.5, color="0.5", lw=1)
    ax.axhline(TARGET, color="0.5", ls="--", lw=1)
    ax.text(1.55, 0.5, "coin flip", ha="right", va="bottom", fontsize=8, color="0.3")
    ax.text(1.55, TARGET, "target", ha="right", va="bottom", fontsize=8, color="0.3")
    ax.set_xticks([0, 1], [SET_NAMES[n] for n in SETS])
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("Zero-shot accuracy")
    return fig


def plot_logits():
    model = load_kryptonet()
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.0), layout="constrained", sharey=True)
    for ax, name in zip(axes, SETS):
        X, y = load(name)
        z = predict_logits(model, X)
        bins = np.linspace(z.min(), z.max(), 50)
        ax.hist(z[y == 1], bins=bins, density=True, color=COLORS[name], alpha=0.35, label="$y = 1$")
        ax.hist(z[y == 0], bins=bins, density=True, histtype="step", color=COLORS[name], label="$y = 0$")
        ax.axvline(0, color="black", lw=0.6)
        ax.set_title(SET_NAMES[name])
        ax.set_xlabel("Logit")
    axes[0].set_ylabel("Density")
    axes[0].set_ylim(0, 0.38)
    axes[0].legend(frameon=False, fontsize=7, ncol=2, loc="upper left")
    return fig


def plot_input_weights():
    # how much each input feature feeds into the first layer
    W1 = linear_layers(load_kryptonet())[0].weight.detach().numpy()  # 512 x 40
    colors = ["0.65"] * W1.shape[1]
    for name in SETS:
        for j in strongest(name, 2)[0]:
            colors[j] = COLORS[name]
    fig, ax = one_panel()
    ax.bar(range(W1.shape[1]), np.linalg.norm(W1, axis=0), width=0.7, color=colors)
    ax.set_xlabel("Input feature $j$")
    ax.set_ylabel("$\\|W_1[:, j]\\|_2$")
    ax.set_ylim(0, 4.9)
    handles = [Patch(color=COLORS[n], label=f"strongest pair in {label(n)}") for n in SETS]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="upper center", ncol=2)
    return fig


def plot_weight_spectra():
    fig, ax = one_panel()
    for k, layer in enumerate(linear_layers(load_kryptonet())[:-1]):
        W = layer.weight.detach().numpy()
        s = np.linalg.svd(W, compute_uv=False)
        ax.plot(np.arange(1, len(s) + 1), s / s[0], color=LAYER_COLORS[k],
                label=f"layer {k + 1} ({W.shape[1]}$\\to${W.shape[0]})")
    ax.set_yscale("log")
    ax.set_xlabel("Index $i$")
    ax.set_ylabel("$\\sigma_i / \\sigma_1$")
    ax.legend(frameon=False, fontsize=7)
    return fig


def plot_weight_energy():
    fig, ax = one_panel()
    for k, layer in enumerate(linear_layers(load_kryptonet())[:-1]):
        s = np.linalg.svd(layer.weight.detach().numpy(), compute_uv=False)
        ax.plot(np.arange(1, len(s) + 1), np.cumsum(s**2) / np.sum(s**2), color=LAYER_COLORS[k],
                label=f"layer {k + 1}")
    ax.set_xscale("log")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Rank $r$")
    ax.set_ylabel("Share of $\\|W\\|_F^2$\nin the top $r$ directions")
    ax.legend(frameon=False, fontsize=7)
    return fig


def plot_lora_params():
    # LoRA replaces the update to a d_out x d_in matrix with B A, B: d_out x r, A: r x d_in,
    # so each layer trains r * (d_in + d_out) numbers instead of d_in * d_out
    model = load_kryptonet()
    layers = linear_layers(model)
    ranks = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256])
    lora = ranks * sum(l.in_features + l.out_features for l in layers)
    full = sum(p.numel() for p in model.parameters())
    linear_probe = 512 + 1
    mlp_probe = 512 * 512 + 512 + 512 + 1

    fig, ax = one_panel(height=2.4)
    ax.plot(ranks, lora, marker="o", ms=4, color="black", label="LoRA on every layer")
    for value, text in ((full, "full fine-tuning"), (mlp_probe, "MLP probe"), (linear_probe, "linear probe")):
        ax.axhline(value, color="0.5", ls="--", lw=1)
        ax.text(1, value * 1.2, f"{text} ({value:,})", fontsize=7, color="0.3")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Trainable parameters")
    ax.legend(frameon=False, fontsize=7, loc="center right")
    return fig
