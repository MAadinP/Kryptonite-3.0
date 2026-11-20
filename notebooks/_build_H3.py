"""Builds notebooks/H3_lora.ipynb (run once; the notebook is the deliverable)."""
from pathlib import Path

import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
md(r"""# H3: Can KryptoNet-40 be adapted to Kryptonite-40-Adapt?

**Claim refuted.** The mock paper finds that KryptoNet-40 used zero-shot, or with a linear or MLP probe on its frozen
representation, stays at chance on Kryptonite-40-Adapt, and concludes that the model cannot be adapted.

**Hypothesis H3.** LoRA on the four hidden layers reaches 90% mean test accuracy, while the frozen-representation
methods (and the same architecture trained from scratch) stay near chance.

The experiment itself is `run_lora.py` in the repository root (LoRA ranks 1--32, full fine-tuning, zero-shot,
linear probe, MLP probe and from-scratch, on 5 seeded 60/20/20 splits; Adam, lr $10^{-3}$, batch 128, at most 100
epochs, early stopping on validation accuracy with patience 15). This notebook runs it unchanged, logs its CPU time
for the sustainability section, and draws the report's figure from the numbers it saves (`results_lora.json`)."""),
code(r"""import json
import os
import runpy
import shutil

import numpy as np
import matplotlib.pyplot as plt

from include.config import ROOT
from include.tracking import track
from include.figures import use_paper_style, savefig
from include.plots_h3 import plot_lora_rank, lora_ranks

use_paper_style()
RESULTS_FILE = ROOT / "results_lora.json"
PREDICTIONS = ROOT / "hiddenlabels" / "y_predicted_40_adapt.npy"
RERUN = not RESULTS_FILE.exists()  # set to True to rerun the experiment (about 15 CPU-minutes)"""),
md(r"""## 1. Run the experiment

`run_lora.py` uses paths relative to the repository root, so it runs from there. It also rewrites the submitted
predictions; we keep a copy first and check whether the rerun reproduces them exactly."""),
code(r"""if RERUN:
    backup = PREDICTIONS.read_bytes() if PREDICTIONS.exists() else None
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        with track("H3_lora"):
            runpy.run_path("run_lora.py", run_name="__main__")
    finally:
        os.chdir(cwd)
    if backup is not None:
        print("\nrerun reproduces the submitted predictions exactly:", PREDICTIONS.read_bytes() == backup)
else:
    print(f"using saved results in {RESULTS_FILE.name} (set RERUN = True to retrain)")"""),
md(r"""## 2. Results table (Table in the report)

Mean $\pm$ standard deviation of test accuracy over the 5 seeds, and the number of trainable parameters."""),
code(r"""res = json.loads(RESULTS_FILE.read_text())
names = {"zero_shot": "Zero-shot", "linear_probe": "Linear probe", "mlp_probe": "MLP probe",
         "scratch": "Scratch (same arch.)", "full_finetune": "Full fine-tuning",
         **{f"lora_r{r}": f"LoRA r={r}" for r in lora_ranks(res)}}
print(f"{'method':<22}{'trainable':>10}   test accuracy      val accuracy")
for key, name in names.items():
    t, v = np.array(res["test"][key]), np.array(res["val"][key])
    print(f"{name:<22}{res['params'][key]:>10,}   {t.mean():.3f} +- {t.std():.3f}    {v.mean():.3f}")

lora = [k for k in names if k.startswith("lora")]
print("\nLoRA mean test accuracy above 90% at every rank:", all(np.mean(res["test"][k]) > 0.90 for k in lora))
print("rank with the highest mean validation accuracy:", max(lora, key=lambda k: np.mean(res["val"][k])))"""),
md(r"""## 3. Figure

LoRA test accuracy against rank (mean $\pm$ sd over 5 seeds), with full fine-tuning and the mean of the frozen /
from-scratch baselines as reference lines. Drawn one column wide with 10 pt text, as the report requires."""),
code(r"""fig = plot_lora_rank(res)
savefig(fig, "H3_lora_rank")
plt.show()"""),
]
nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(__file__).with_name("H3_lora.ipynb"))
print("written")
