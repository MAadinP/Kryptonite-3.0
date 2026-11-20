"""
LoRA adaptation of KryptoNet-40 on Kryptonite-40-Adapt (CPU only).

Put this file in the repo root (next to kryptonet.py) and run:
    python run_lora_final.py

Pure LoRA: every pretrained weight/bias (including the output head) is FROZEN.
Only A and B in the four hidden layers are trained.

Produces:
    results_lora.json                           (all numbers for tables/figures)
    hiddenlabels/y_predicted_40_adapt.npy       (10,000 binary labels to submit)
"""
import copy, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
from kryptonet import kryptonet, load_kryptonet

# ------------------------------ config -------------------------------------
D = "Datasets"
X_PATH = os.path.join(D, "kryptonite-40-adapt-X.npy")         # (4000, 40)  labelled
Y_PATH = os.path.join(D, "kryptonite-40-adapt-y.npy")         # (4000,)
HIDDEN_PATH = os.path.join(D, "hidden-kryptonite-40-adapt-X.npy")  # (10000, 40) unlabelled
WEIGHTS = "kryptonet40.pt"
OUT_DIR = "hiddenlabels"

HIDDEN_IDX = (0, 2, 4, 6)      # the four hidden nn.Linear layers in the nn.Sequential
HEAD_IDX = 8                   # final Linear(512, 1)

RANKS = [1, 2, 4, 8, 16, 32]
SEEDS = [0, 1, 2, 3, 4]
LR, EPOCHS, BS, PATIENCE = 1e-3, 100, 128, 15


# ------------------------------ LoRA ---------------------------------------
class LoRALinear(nn.Module):
    """y = W x + b + (alpha/r) * B (A x).
    W, b frozen. A: r x d_in (Kaiming-uniform), B: d_out x r (zeros), so the
    adapted model is identical to the pretrained one at step 0."""
    def __init__(self, base: nn.Linear, r: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        # (x A^T) B^T costs O(N r (d_in + d_out)); forming B @ A first would not.
        return self.base(x) + self.scale * ((x @ self.A.T) @ self.B.T)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def add_lora(model, r, alpha=None):
    freeze_all(model)
    for i in HIDDEN_IDX:
        model[i] = LoRALinear(model[i], r, alpha if alpha else 2 * r)
    return model


def n_trainable(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------ training / eval ----------------------------
def logits(model, x):
    return model(x).squeeze(1)


def acc(model, X, y):
    model.eval()
    with torch.no_grad():
        return ((logits(model, X) > 0).long() == y).float().mean().item()


def fit(model, Xtr, ytr, Xva, yva, seed):
    """Adam + BCE; early stopping on validation accuracy; returns best val acc."""
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    best, bad, state = -1.0, 0, None
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        for i in range(0, len(Xtr), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            lossf(logits(model, Xtr[idx]), ytr[idx].float()).backward()
            opt.step()
        va = acc(model, Xva, yva)
        if va > best:
            best, bad, state = va, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(state)
    return best


def split(n, seed, fracs=(0.6, 0.2)):
    idx = np.random.RandomState(seed).permutation(n)
    a, b = int(fracs[0] * n), int((fracs[0] + fracs[1]) * n)
    return idx[:a], idx[a:b], idx[b:]


def stats(v):
    return f"{np.mean(v):.3f} +/- {np.std(v):.3f}"


# ------------------------------ main ---------------------------------------
if __name__ == "__main__":
    _t0 = time.process_time()
    torch.set_num_threads(os.cpu_count())

    torch.set_num_threads(os.cpu_count())
    X = torch.from_numpy(np.load(X_PATH)).float()
    y = torch.from_numpy(np.load(Y_PATH)).long()
    Xh = torch.from_numpy(np.load(HIDDEN_PATH)).float()
    print("labelled:", tuple(X.shape), "label mean:", round(y.float().mean().item(), 3),
          "| hidden:", tuple(Xh.shape))

    # ---- sanity checks -----------------------------------------------------
    pre = load_kryptonet(WEIGHTS)
    print("full model params:", sum(p.numel() for p in pre.parameters()), "(expect 809473)")
    print("zero-shot acc on all 4000:", round(acc(pre, X, y), 3), "(paper reports ~0.48)")
    chk = add_lora(load_kryptonet(WEIGHTS), r=8).eval()
    with torch.no_grad():
        assert torch.allclose(logits(chk, X), logits(pre, X), atol=1e-5), "LoRA init changed the model!"
    assert n_trainable(chk) == 3624 * 8, n_trainable(chk)
    print("checks passed: LoRA starts identical to pretrained; trainable params =", n_trainable(chk))

    # ---- experiments -------------------------------------------------------
    methods = ["zero_shot", "linear_probe", "mlp_probe", "scratch", "full_finetune"] \
              + [f"lora_r{r}" for r in RANKS]
    val = {m: [] for m in methods}
    test = {m: [] for m in methods}
    params = {}

    for s in SEEDS:
        tr, va, te = split(len(X), s)
        Xtr, ytr, Xva, yva, Xte, yte = X[tr], y[tr], X[va], y[va], X[te], y[te]

        def record(name, model, train=True):
            v = fit(model, Xtr, ytr, Xva, yva, s) if train else acc(model, Xva, yva)
            val[name].append(v)
            test[name].append(acc(model, Xte, yte))
            params[name] = n_trainable(model)
            print(f"seed {s} {name:14s} trainable {params[name]:7d}  val {v:.3f}  test {test[name][-1]:.3f}")

        torch.manual_seed(s)
        record("zero_shot", freeze_all(load_kryptonet(WEIGHTS)), train=False)

        m = freeze_all(load_kryptonet(WEIGHTS))               # paper's linear probe: head only
        for p in m[HEAD_IDX].parameters():
            p.requires_grad_(True)
        record("linear_probe", m)

        record("scratch", kryptonet())                        # same architecture, random init

        m = load_kryptonet(WEIGHTS)                           # all 809,473 weights trainable
        for p in m.parameters():
            p.requires_grad_(True)
        record("full_finetune", m)

        for r in RANKS:
            torch.manual_seed(s)
            record(f"lora_r{r}", add_lora(load_kryptonet(WEIGHTS), r))

        # MLP probe (mock paper, Sec. 3.3): hidden layers frozen, output layer replaced by a
        # one-hidden-layer MLP (512 units) trained on h^(4). Run last so earlier results are unchanged.
        torch.manual_seed(s)
        m = freeze_all(load_kryptonet(WEIGHTS))
        m[HEAD_IDX] = nn.Sequential(nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, 1))  # new, trainable
        record("mlp_probe", m)

    print("\n=== test accuracy, mean +/- std over", len(SEEDS), "seeds ===")
    for m in methods:
        print(f"{m:14s} trainable {params[m]:7d}  {stats(test[m])}")
    json.dump({"val": val, "test": test, "params": params, "seeds": SEEDS},
              open("results_lora.json", "w"), indent=1)

    # ---- final model -> hidden-set predictions -----------------------------
    best_r = 16
    lora_names = [m for m in methods if m.startswith("lora")]
    val_choice = max(lora_names, key=lambda m: np.mean(val[m]))
    print(f"\nusing r = {best_r} (a priori); validation-argmax would be {val_choice}")

    tr, va, _ = split(len(X), 123, fracs=(0.9, 0.1))          # use almost all labelled data
    torch.manual_seed(123)
    final = add_lora(load_kryptonet(WEIGHTS), best_r)
    fit(final, X[tr], y[tr], X[va], y[va], 123)
    final.eval()
    with torch.no_grad():
        preds = (logits(final, Xh) > 0).long().numpy()

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "y_predicted_40_adapt.npy"), preds)
    print("saved", os.path.join(OUT_DIR, "y_predicted_40_adapt.npy"),
          preds.shape, "fraction predicted 1:", round(float(preds.mean()), 3))
    print("Now run: python check_submission.py")

    _cpu_seconds = time.process_time() - _t0
    print(f"\nCPU time: {_cpu_seconds:.1f} s "
    f"({_cpu_seconds/3600:.3f} CPU-hours)")
