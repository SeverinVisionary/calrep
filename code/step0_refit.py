"""CALREP Step 0 (PREREGISTRATION.md): reproduce Lovelace 2605.01640 Table 15
(additive_1p P = 0.002857, n = 158, <=64 epochs) from datablations' published
final losses, under the four preregistered run-set definitions.

Fitter: a reimplementation of justinlovelace/data-constrained-scaling
dcscaling/fit.py + laws.py (2026-10-07) read as a spec. Run-name parsing: the
datablations notebook's own get_params / get_unique_tokens / get_tokens.

usage: python -I step0_refit.py parametric_fit.ipynb out.json
"""
import ast, json, re, sys
import numpy as np
import torch
import torch.nn as nn


def literal_dict(src, name):
    i = src.index(name + " = {")
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        depth += src[k] == "{"
        depth -= src[k] == "}"
        if depth == 0:
            return ast.literal_eval(src[j:k + 1])

def load_notebook(path):
    global LOSS, TOKENS_MAP, PARAMS_MAP
    nb = json.load(open(path))
    c1 = "".join(nb["cells"][1]["source"])
    c25 = "".join(nb["cells"][25]["source"])
    LOSS = {"cell1": literal_dict(c1, "NAMES_TO_VAL_LOSSES"), "cell25": literal_dict(c25, "NAMES_TO_VAL_LOSSES")}
    TOKENS_MAP = literal_dict(c1, "TOKENS_MAP")
    PARAMS_MAP = literal_dict(c1, "PARAMS_MAP")

# --- datablations notebook parsing, verbatim logic ---
def get_params(name):
    idx = 2
    while name[:idx] not in PARAMS_MAP:
        idx += 1
    return PARAMS_MAP[name[:idx]], idx

def get_unique_tokens(name):
    idx = -4 if (name[-1] in ("b", "m")) or ((name[-1] not in ("b", "m")) and (name[-2] not in ("b", "m"))) else -3
    ut = name[idx:]
    while (ut.count("b") > 1) or (ut.count("m") > 1):
        idx += 1
        ut = name[idx:]
    ut = TOKENS_MAP[ut] if ut in TOKENS_MAP else float(ut.replace("b", ".").strip("."))
    return ut * 1e9, idx

def get_tokens(t):
    t = TOKENS_MAP[t] if t in TOKENS_MAP else float(t.replace("b", ".").strip("."))
    return t * 1e9

def clean(name):
    for s in ["op2", "seed1", "seed2", "seed3", "seed4", "opt1", "opt", "c4", "v2"]:
        name = name.replace(s, "")
    return name

def rows(d, drop_seeds=False, drop_variants=False):
    out, skipped = [], []
    for raw, loss in d.items():
        if loss == 0:
            continue
        if drop_seeds and "seed" in raw:
            continue
        if drop_variants and any(v in raw for v in ("dedup", "opt", "subopt")):
            continue
        name = clean(raw.replace("dedup", "")) if "dedup" in raw else clean(raw)
        try:
            N, a = get_params(name)
            U, b = get_unique_tokens(name)
            D = get_tokens(name[a:b])
        except Exception as e:  # noqa: BLE001 - logged, not hidden
            skipped.append((raw, repr(e)))
            continue
        out.append({"name": raw, "N": N, "D": D, "U_D": U, "epochs": D / U, "val_loss": float(loss)})
    return out, skipped

# --- Lovelace fitter (spec: dcscaling/fit.py, laws.py) ---
def huber(yp, yt):
    return nn.functional.huber_loss(torch.log(yp + 1e-8), torch.log(yt + 1e-8), reduction="sum", delta=1e-3)

def lbfgs(params, closure_fn, max_iter):
    opt = torch.optim.LBFGS(params, lr=1, max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        opt.zero_grad(); l = closure_fn(); l.backward(); return l
    try:
        opt.step(closure)
    except RuntimeError as e:
        if "cannot be converted to type float without overflow" in str(e):
            return float("inf")
        raise
    with torch.no_grad():
        return closure_fn().item()

def fit_chinchilla(N, D, y):
    X = [torch.tensor(v, dtype=torch.float32) for v in (N, D)]
    yt = torch.tensor(y, dtype=torch.float32)
    best = (float("inf"), None)
    for E in [1.5, 2.0, 2.5, 3.0]:
        for A in [1.0, 10.0, 100.0]:
            for al in [0.3, 0.5, 0.7]:
                for B in [1.0, 10.0, 100.0]:
                    for be in [0.3, 0.5, 0.7]:
                        p = [nn.Parameter(torch.tensor(np.log(v), dtype=torch.float32)) for v in (E, A, al, B, be)]
                        f = lambda: huber(torch.exp(p[0]) + torch.exp(p[1]) * X[0] ** (-torch.exp(p[2]))
                                          + torch.exp(p[3]) * X[1] ** (-torch.exp(p[4])), yt)
                        l = lbfgs(p, f, 200)
                        if np.isfinite(l) and l < best[0]:
                            best = (l, [float(torch.exp(q)) for q in p])
    return best

def fit_p1(base, N, U, R, y):
    E, A, al, B, be = [torch.tensor(v, dtype=torch.float32) for v in base]
    Nt, Ut, Rt = [torch.tensor(v, dtype=torch.float32) for v in (N, U, R)]
    yt = torch.tensor(y, dtype=torch.float32)
    fresh = E + A * Nt ** (-al) + B * (Ut * (1 + Rt)) ** (-be)
    best = (float("inf"), None)
    for p0 in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0, 100.0]:
        lp = nn.Parameter(torch.tensor(np.log(p0), dtype=torch.float32))
        f = lambda: huber(fresh + torch.exp(lp) * Rt * (Nt / Ut), yt)
        l = lbfgs([lp], f, 500)
        if np.isfinite(l) and l < best[0]:
            best = (l, float(torch.exp(lp)))
    return best

def r2(y, p):
    return float(1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))

def run(rs):
    N = np.array([r["N"] for r in rs]); U = np.array([r["U_D"] for r in rs])
    ep = np.array([r["epochs"] for r in rs]); y = np.array([r["val_loss"] for r in rs])
    R = ep - 1
    single = np.isclose(ep, 1.0, atol=1e-6) | (ep < 1.0 + 1e-6)
    h1, base = fit_chinchilla(N[single], (U * ep)[single], y[single])
    h2, P = fit_p1(base, N, U, R, y)
    E, A, al, B, be = base
    pred = E + A * N ** (-al) + B * (U * (1 + R)) ** (-be) + P * R * N / U
    multi = R > 1e-9
    return {"n": int(len(rs)), "n_single": int(single.sum()), "n_multi": int(multi.sum()),
            "base": dict(zip(["E", "A", "alpha", "B", "beta"], base)), "P": P,
            "r2": r2(y, pred), "r2_multi": r2(y[multi], pred[multi]), "huber_p2": h2}

if __name__ == "__main__":
    load_notebook(sys.argv[1])
    torch.manual_seed(0)
    torch.set_num_threads(4)
    defs = {
        "a_cell1_le64": rows(LOSS["cell1"]),
        "b_cell1_le64_noseed": rows(LOSS["cell1"], drop_seeds=True),
        "c_cell1_le64_noseed_novariant": rows(LOSS["cell1"], drop_seeds=True, drop_variants=True),
        "d_cell25_le64": rows(LOSS["cell25"]),
    }
    out = {"target": {"P": 0.002857, "n": 158, "phase1": {"E": 1.9031, "A": 432.63, "alpha": 0.3362, "B": 5360.24, "beta": 0.3868}}}
    for k, (rs, skipped) in defs.items():
        rs = [r for r in rs if r["epochs"] <= 64 + 1e-6]
        res = run(rs)
        res["skipped_unparsed"] = skipped
        res["pass_n"] = abs(res["n"] - 158) <= 5
        res["pass_P"] = abs(res["P"] - 0.002857) / 0.002857 <= 0.15
        res["runs"] = [r["name"] for r in rs]
        out[k] = res
        print(k, {x: res[x] for x in ("n", "n_single", "n_multi", "P", "r2", "r2_multi", "pass_n", "pass_P")},
              {x: round(v, 4) for x, v in res["base"].items()}, "skipped", len(skipped), flush=True)
    json.dump(out, open(sys.argv[2], "w"), indent=1)
