"""Code shared by the notebooks in this folder.

    config.py            paths, dataset names, colours
    data.py              loading and splitting the datasets
    interactions.py      how single features, pairs and triples go with the label
    models.py            the MLP, its training loop, KryptoNet-40
    features.py          ranking features by MLP flip-influence; polynomial LR and lookup table on the top k
    stats.py             Chebyshev confidence interval, mean and std
    tracking.py          runtime / CO2 log and saving results
    figures.py           paper plot style and saving figures
    plots_data.py        figures about the datasets
    plots_kryptonet.py   figures about KryptoNet-40 and LoRA
    plots_h1.py          figures for H1_learnability and H1_features
    plots_h2.py          figures for H2_more_data
    plots_sustainability.py   figure for the sustainability notebook

From a notebook in this folder:  from include.data import load
"""
