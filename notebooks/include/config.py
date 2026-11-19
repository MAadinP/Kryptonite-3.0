"""Paths, dataset names and colours used everywhere."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "Datasets"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
HIDDENLABELS = ROOT / "hiddenlabels"
KRYPTONET = ROOT / "kryptonet40.pt"

NS = (20, 30, 40)
NAMES = (20, 30, 40, "40-adapt")  # every labelled dataset
SEEDS = (0, 1, 2, 3, 4)
TARGET = 0.90  # accuracy the paper says can't be reached
DELTA = 0.05

# one colour and marker per dataset (checked for colour-blind readers)
COLORS = {20: "#2a78d6", 30: "#eb6834", 40: "#1baf7a", "40-adapt": "#c03f8a"}
MARKERS = {20: "o", 30: "s", 40: "^", "40-adapt": "D"}
