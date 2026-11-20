"""Plot style for the report, saving figures to figures/, and a few layout shortcuts."""
import matplotlib.pyplot as plt

from .config import FIGURES, NAMES

COLUMN = 3.25  # width of one column in the ICML template (inches)
PAGE = 6.75    # width of the whole page


def use_paper_style():
    # 10pt to match the paper, and fonttype 42 so the PDFs don't use Type-3 fonts
    plt.rcParams.update({
        "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10, "legend.fontsize": 10,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
    })


def savefig(fig, name):
    FIGURES.mkdir(exist_ok=True)
    # save at exactly the drawn size (no "tight" cropping): a COLUMN-wide figure then fills
    # \linewidth at 100% scale, so its 10pt text stays 10pt in the report
    fig.savefig(FIGURES / f"{name}.pdf")
    fig.savefig(FIGURES / f"{name}.png", dpi=200)


def label(name):
    return "40-adapt" if name == "40-adapt" else f"$n={name}$"


def one_panel(height=2.2):
    return plt.subplots(figsize=(COLUMN, height), layout="constrained")


def four_panels(height=1.9, **kwargs):
    # one panel per dataset, across the whole page
    fig, axes = plt.subplots(1, 4, figsize=(PAGE, height), layout="constrained", **kwargs)
    for ax, name in zip(axes, NAMES):
        ax.set_title(label(name))
    return fig, axes
