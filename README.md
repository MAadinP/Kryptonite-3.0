# Refuting Kryptonite-3.0

Code, results and hidden-set predictions for our ImpCML 2026 response paper
(Mathematics for Machine Learning, 60037/70015, Imperial College London).

The mock paper in `PaperAndInstructions/` makes three claims, and we test each with one hypothesis:

| Claim in the mock paper | Our hypothesis | Where |
|---|---|---|
| Polynomial LR stays below 59%, so Kryptonite-*n* is a counterexample to universal approximation | **H1**: the paper's own polynomial LR exceeds 90% once its basis (inputs and degree) is chosen from the data | `notebooks/H1_*` |
| More data does not help | **H2**: more data helps a model that can represent the signal; the quadratic model stays flat | `notebooks/H2_*` |
| KryptoNet-40 cannot be adapted to Kryptonite-40-Adapt | **H3**: LoRA on the hidden layers exceeds 90% while frozen-representation methods stay at chance | `run_lora.py` |

Every notebook is committed **with its outputs**, so every number and figure can be checked without running anything.

## Figures in the report and the code that makes them

All report figures are saved to `figures/` as PDF (used in the report) and PNG.
`notebooks/report_figures.ipynb` redraws **every** figure from the saved results in `results/` in a few minutes, without
training anything; it is the quickest way to check a figure.

| Report figure | File | Data produced by | Also drawn in |
|---|---|---|---|
| Figure (file in `figures/`) | Notebook | What it shows |
|---|---|---|
| `H1_models.pdf` | `notebooks/H1_learnability.ipynb` | Test accuracy of logistic regression (degree 1 and 2, all inputs) and an MLP (raw and sign inputs) on the same 5 splits, with Chebyshev error bars. Polynomial LR stays near chance; the MLP reaches 0.93--0.95, so the data are learnable. |
| `H1_influence_col.pdf` | `notebooks/H1_features.ipynb` | For each feature, how often flipping its sign changes the MLP probe's prediction (training split, mean over seeds). Exactly ten features matter for every *n*; the rest are noise. |
| `H1_topk.pdf` | `notebooks/H1_features.ipynb` | The paper's polynomial LR refitted on the top *k* probe-ranked features (all products). Accuracy jumps from about 0.68 to 0.96 at *k* = 10 and drops when irrelevant features are added; the lookup table (noise ceiling) matches it. |
| `H1_poly_degree.pdf` | `notebooks/H1_features.ipynb` | The same model on the top 10 features against degree *d*. The paper's *d* = 2 gives about 0.56; the 90% target needs *d* &ge; 6. |
| `H2_learning_curves.pdf` | `notebooks/H2_more_data.ipynb` | Test accuracy against training-set size (5--100%) for the paper's quadratic LR and the MLP. Quadratic LR stays flat (bias-limited); the MLP gains 37--43 points. |
| `data_corr_by_order.pdf` | `notebooks/understand_data.ipynb` | Strongest correlation of single features, pairs and triples with the label, against shuffled labels. Pairs carry only weak signal (about 57% accuracy alone), which explains the flat quadratic curve. |
| `H3_lora_rank.pdf` | `notebooks/H3_lora.ipynb` | LoRA test accuracy against rank (mean &plusmn; sd over 5 seeds), with full fine-tuning and the frozen / from-scratch baselines. LoRA exceeds 90% at every rank; frozen methods stay at chance. |

Every one of these figures is also redrawn, from the saved results only, by `notebooks/report_figures.ipynb`.
`H3_lora.ipynb` runs the experiment script `run_lora.py` unchanged and draws the figure from its output
`results_lora.json`; it replaces the earlier `lora_fig.py`, which had the numbers typed in by hand.

The report's tables are printed in `H1_learnability.ipynb` and `H1_features.ipynb` (H1), `H2_more_data.ipynb` (H2)
and `H3_lora.ipynb` (H3). Numbers quoted in the text that those notebooks do not print are in:

* `notebooks/H1_checks.ipynb`: raw vs sign inputs, convergence of the paper's model, correlation screening, basis sizes.
* `notebooks/H2_checks.ipynb`: convergence at every training size, correlations by degree.
* `notebooks/sustainability.ipynb`: CPU time, energy and CO2e for the environmental-impact section.

## Repository layout

```
notebooks/            experiments (each saves its numbers to results/ and its figures to figures/)
notebooks/include/    shared code: data loading and splits, models, training, statistics, plotting
results/              saved numbers (*.json), every trained H1 model and split (results/H1/),
                      final MLPs (results/final/), runtime log for the sustainability section
figures/              every figure as PDF and PNG
hiddenlabels/         y_predicted_{20,30,40,40_adapt}.npy  (our hidden-set predictions)
run_lora.py           H3 experiment: LoRA, full fine-tuning and frozen baselines (run by H3_lora.ipynb)
results_lora.json     H3 results saved by run_lora.py
requirements.txt      exact package versions used
run_notebook.py       runs notebooks in place and saves their outputs
```

## Reproducing the results

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python run_notebook.py notebooks/<name>.ipynb      # or open it in Jupyter
```

Run the notebooks in this order (later ones read the results of earlier ones). Times are CPU time on a laptop with
4 torch threads.

| Order | Notebook | What it does | CPU time |
|---|---|---|---|
| 1 | `H1_learnability.ipynb` | Picks hyper-parameters on the seed-0 validation split; fits LR (*d* = 1, 2) and the MLP on 5 seeds x 3 datasets; saves every model and split | ~6 h |
| 2 | `H1_features.ipynb` | MLP probe, flip-influence ranking, polynomial LR and lookup table on the top *k* features | ~1.5 h |
| 3 | `H1_checks.ipynb` | Robustness checks behind numbers quoted in the report (no MLP training) | ~5 min |
| 4 | `H2_more_data.ipynb` | Learning curves for quadratic LR and the MLP | ~5 h |
| 5 | `H2_checks.ipynb` | Convergence and correlation numbers for H2 (no MLP training) | ~2 min |
| 6 | `final_predictions.ipynb` | Writes `hiddenlabels/y_predicted_{20,30,40}.npy` and runs `check_submission.py` | ~20 min |
| 7 | `H3_lora.ipynb` | Runs `run_lora.py` (LoRA, full fine-tuning, frozen baselines); writes `results_lora.json` and `hiddenlabels/y_predicted_40_adapt.npy`; draws the H3 figure | ~15 min |
| 8 | `sustainability.ipynb` | Energy and CO2e from `results/runtime_log.json` | seconds |
| 9 | `report_figures.ipynb` | Redraws every figure from the saved results | ~5 min |

`understand_data.ipynb` and `understand_kryptonet.ipynb` are exploratory analyses of the datasets and of KryptoNet-40;
nothing is trained.

### Seeding

Every source of randomness is seeded: the stratified 60/20/20 splits (`random_state=seed`), MLP initialisation
(`torch.manual_seed(seed)`), mini-batch order (a seeded `torch.Generator`), the nested training subsets in H2 and the
shuffled-label controls. With the versions in `requirements.txt` and 4 torch threads, retraining reproduces the saved
models bit for bit; `H1_learnability.ipynb` (section 5) reloads every saved model and checks that it reproduces its
saved test predictions.

## Hidden-set predictions

| File | Model | Validation accuracy |
|---|---|---|
| `y_predicted_20.npy` | Polynomial LR on the top 10 probe-ranked features, degree 8 | 0.958 |
| `y_predicted_30.npy` | Polynomial LR on the top 10 probe-ranked features, degree 8 | 0.959 |
| `y_predicted_40.npy` | Polynomial LR on the top 10 probe-ranked features, degree 6 | 0.959 |
| `y_predicted_40_adapt.npy` | KryptoNet-40 with LoRA (rank 16) on the hidden layers | see `H3_lora.ipynb` |

All four pass `python check_submission.py .`

## Coursework materials (from the original handout)

| Path | What it is |
|---|---|
| `PaperAndInstructions/` | Mock paper, coursework instructions, marking scheme |
| `Datasets/` | Labelled Kryptonite-{20,30,40} and 40-Adapt data, and the unlabelled hidden sets |
| `kryptonet40.pt`, `kryptonet.py` | KryptoNet-40 weights and model definition |
| `MockPaperExperiments.ipynb` | Reproduces the mock paper's figures |
| `check_submission.py` | Checks the format of the hidden-set predictions |
