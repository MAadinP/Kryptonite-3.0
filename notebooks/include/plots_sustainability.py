"""Figure for the sustainability notebook."""
import matplotlib.pyplot as plt

from .figures import COLUMN
from .tracking import energy_kwh, GRID_INTENSITY


def plot_energy(log):
    # estimated energy of each logged experiment (from its CPU time), with its CO2e at the end of the bar
    names = list(log)
    wh = [energy_kwh(log[k]["cpu_s"]) * 1000 for k in names]
    fig, ax = plt.subplots(figsize=(COLUMN, 0.32 * len(names) + 0.9), layout="constrained")
    bars = ax.barh(names, wh, color="0.45", height=0.6)
    ax.bar_label(bars, labels=[f"{w * GRID_INTENSITY:.1f} g" for w in wh], padding=3)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated energy (Wh)")
    ax.set_xlim(0, max(wh) * 1.35)
    return fig
