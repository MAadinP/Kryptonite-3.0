"""Figure for the H3 (LoRA) notebook. Reads the dict saved by run_lora.py (results_lora.json)."""
import numpy as np
import matplotlib.pyplot as plt

from .config import TARGET
from .figures import COLUMN

FROZEN = ("zero_shot", "linear_probe", "mlp_probe", "scratch")  # the paper's methods + same arch. from scratch


def lora_ranks(res):
    return sorted(int(m[len("lora_r"):]) for m in res["test"] if m.startswith("lora_r"))


def plot_lora_rank(res):
    # LoRA test accuracy against rank (mean +- sd over seeds, np.std as in run_lora.py's table),
    # with full fine-tuning and the frozen / from-scratch baselines as reference lines
    test = res["test"]
    ranks = lora_ranks(res)
    means = [np.mean(test[f"lora_r{r}"]) for r in ranks]
    sds = [np.std(test[f"lora_r{r}"]) for r in ranks]
    full = np.mean(test["full_finetune"])
    frozen = np.mean([np.mean(test[m]) for m in FROZEN])

    fig, ax = plt.subplots(figsize=(COLUMN, 2.5), layout="constrained")
    ax.errorbar(ranks, means, yerr=sds, marker="o", ms=4, capsize=2, color="#2a78d6", label="LoRA")
    ax.axhline(TARGET, ls="--", color="0.3", lw=1, label="90% target")
    ax.axhline(full, ls="-.", color="#eb6834", lw=1.2, label=f"Full fine-tune")
    ax.axhline(frozen, ls=":", color="0.4", lw=1.2, label="Frozen / scratch")
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [str(r) for r in ranks])
    ax.minorticks_off()
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Test accuracy")
    ax.set_ylim(0.45, 1.0)
    fig.legend(frameon=False, ncol=2, loc="outside upper center", columnspacing=1.0, handlelength=1.8)
    return fig
