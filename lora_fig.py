import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 10, "font.family": "serif"})

ranks = np.array([1, 2, 4, 8, 16, 32])
means = np.array([0.942, 0.918, 0.946, 0.948, 0.955, 0.953])
stds = np.array([0.019, 0.023, 0.013, 0.012, 0.005, 0.004])

# Wider canvas gives the 2-column legend and plot area room to breathe
fig, ax = plt.subplots(figsize=(4.8, 3.4))

ax.errorbar(
    ranks,
    means,
    yerr=stds,
    marker="o",
    capsize=3,
    color="tab:blue",
    label="LoRA (mean $\\pm$ std, 5 seeds)",
)

ax.axhline(0.90, ls="--", color="black", lw=1, label="90% target")
ax.axhline(0.853, ls=":", color="tab:red", lw=1, label="Full fine-tuning (0.853)")
ax.axhline(
    0.50, ls=":", color="grey", lw=1, label="Frozen probes / scratch ($\\approx$0.50)"
)

ax.set_xscale("log", base=2)
ax.set_xticks(ranks)
ax.set_xticklabels([str(r) for r in ranks])
ax.set_xlabel("LoRA rank $r$")
ax.set_ylabel("Test accuracy")
ax.set_ylim(0.45, 1.00)
ax.grid(alpha=0.3, lw=0.5)

# Push anchor slightly lower to clear the x-axis label cleanly
ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.24),
    ncol=2,
    fontsize=8.5,
    frameon=False,
)

fig.tight_layout()
fig.savefig("fig_rank.pdf", bbox_inches="tight")