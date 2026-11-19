"""Figure for the sustainability notebook."""
import matplotlib.pyplot as plt

from .figures import COLUMN


def plot_energy(log):
    # estimated energy of each logged experiment, with its CO2e written at the end of the bar
    names = list(log)
    wh = [log[k]["energy_kWh"] * 1000 for k in names]
    fig, ax = plt.subplots(figsize=(COLUMN, 0.28 * len(names) + 0.8), layout="constrained")
    bars = ax.barh(names, wh, color="0.45", height=0.6)
    ax.bar_label(bars, labels=[f"{log[k]['kgCO2e'] * 1000:.1f} g" for k in names], padding=3, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated energy (Wh)")
    ax.set_xlim(0, max(wh) * 1.25)
    ax.tick_params(axis="y", labelsize=7)
    return fig
