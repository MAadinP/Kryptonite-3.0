"""The MLP, its training loop, and the pretrained KryptoNet-40."""
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import KRYPTONET


def mlp(n_inputs, width=128, depth=2):
    layers, d = [], n_inputs
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]
        d = width
    return nn.Sequential(*layers, nn.Linear(d, 1))


def load_kryptonet():
    # KryptoNet-40 is a plain ReLU MLP with four hidden layers of width 512
    model = mlp(40, width=512, depth=4)
    model.load_state_dict(torch.load(KRYPTONET, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def predict_logits(model, X):
    model.eval()
    return model(torch.as_tensor(X, dtype=torch.float32)).squeeze(1).numpy()


def accuracy(model, X, y):
    return float(((predict_logits(model, X) > 0) == y).mean())


def fit(model, X_tr, y_tr, X_va, y_va, lr=1e-3, weight_decay=1e-2, epochs=300, patience=40,
        batch_size=128, seed=0):
    # minimise binary cross-entropy (the Bernoulli NLL) with mini-batch AdamW.
    # stop when val accuracy hasn't improved for `patience` epochs and keep the best weights.
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    X_tr = torch.as_tensor(X_tr, dtype=torch.float32)
    y_tr = torch.as_tensor(y_tr, dtype=torch.float32)
    g = torch.Generator().manual_seed(seed)

    best_acc, best_state, bad_epochs = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(len(y_tr), generator=g)
        for i in range(0, len(perm), batch_size):
            b = perm[i:i + batch_size]
            loss = F.binary_cross_entropy_with_logits(model(X_tr[b]).squeeze(1), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()

        acc = accuracy(model, X_va, y_va)
        if acc > best_acc:
            best_acc, best_state, bad_epochs = acc, copy.deepcopy(model.state_dict()), 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_acc
