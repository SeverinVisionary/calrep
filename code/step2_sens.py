"""CALREP Step 2 sensitivities S1-S4 (PREREGISTRATION.md; reported, not decisive).

Definitions fixed here before the script was first run (2026-10-09):
  S1  additive_1p rho on the Step 1 set minus diverged runs. Diverged: published loss
      exceeds the minimum published loss in its (N, U_D) cell by more than 1.0 nat. Cell
      minima are taken over the accepted Step 0 set (158 runs), before the checkpoint
      filter (corrected 2026-10-09 after PR #1152 review; first run used the 125-run set).
  S2  additive_2p and additive_4p (vendored Lovelace fitter), each on our own Phase 1
      base per loss. Reported: P_TS / P_raw, and the ratio of the mean fitted penalty
      over multi-epoch runs (TS / raw), which is comparable across forms.
  S3  muennighoff_d and muennighoff_dn (vendored fitter, our Phase 1 base): R* per loss.
      Larger R* means repeated data stays useful longer. Reported raw and TS.
  S4  per cell (N, U_D) with >= 2 runs in the set: rise_X = L_X(max-epoch run) -
      min over the cell of L_X, for X in {raw, TS}. For cells with rise_raw > 0:
      phi = 1 - rise_TS / rise_raw (share of the post-minimum rise removed). Per cell
      and median. Added after PR #1152 review: cells containing an S1-diverged run are
      marked, and the median is also reported without them (CONTEXT.md: diverged runs
      "must be flagged, not pooled").

usage: python -I step2_sens.py NOTEBOOK RUNSET final.json PUBLISHED_TSV out.json
"""
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "vendor"))
import step0_refit as s0  # noqa: E402
from step2_fit import fit  # noqa: E402
from dcscaling.fit import _fit_params  # noqa: E402
from dcscaling.laws import Chinchilla, predict_loss  # noqa: E402

def phase1(N, U, ep, y):
    single = ep < 1.0 + 1e-6
    _, base = s0.fit_chinchilla(N[single], (U * ep)[single], y[single])
    return Chinchilla(*base)

def main():
    nb, runset, final_p, pub_p, out_p = sys.argv[1:6]
    import torch
    torch.manual_seed(0); torch.set_num_threads(4)
    s0.load_notebook(nb)
    rows, _ = s0.rows(s0.LOSS["cell1"], drop_seeds=True, drop_variants=True)
    keep = {r["name"] for r in csv.DictReader(open(runset), delimiter="\t") if r["in_step1"] == "yes"}
    final = json.load(open(final_p))
    pub = {r[0]: float(r[1]) for r in csv.reader(open(pub_p), delimiter="\t")}
    use = [r for r in rows if r["epochs"] <= 64 + 1e-6 and r["name"] in keep]
    names = [r["name"] for r in use]
    N = np.array([r["N"] for r in use]); U = np.array([r["U_D"] for r in use]); ep = np.array([r["epochs"] for r in use])
    L = {"raw": np.array([final[n]["raw_nll"] for n in names]), "ts": np.array([final[n]["ts_nll"] for n in names])}
    P = np.array([pub[n] for n in names]); R = ep - 1
    out = {}

    # S1
    cell = list(zip(N, U))
    acc = [r for r in rows if r["epochs"] <= 64 + 1e-6]  # accepted Step 0 set, before the checkpoint filter
    cmin = {}
    for r in acc:
        c = (r["N"], r["U_D"]); cmin[c] = min(cmin.get(c, np.inf), r["val_loss"])
    div = np.array([P[i] > cmin[cell[i]] + 1.0 for i in range(len(names))])
    m = ~div
    f = {k: fit(N[m], U[m], ep[m], L[k][m]) for k in L}
    out["S1"] = {"diverged": [n for n, d in zip(names, div) if d], "n": int(m.sum()),
                 "P_raw": f["raw"]["P"], "P_ts": f["ts"]["P"], "rho": f["ts"]["P"] / f["raw"]["P"]}
    print("S1", {k: v for k, v in out["S1"].items() if k != "diverged"}, "diverged", out["S1"]["diverged"], flush=True)

    # S2, S3
    base = {k: phase1(N, U, ep, L[k]) for k in L}
    multi = R > 0
    for form in ("additive_2p", "additive_4p", "muennighoff_d", "muennighoff_dn"):
        res = {}
        for k in L:
            fp = _fit_params(form, base[k], N, U, R, L[k])
            if fp is None:
                res[k] = None; continue
            params, hub = fp
            fresh = predict_loss("chinchilla", {}, base[k], N, U, R)
            pen = predict_loss(form, params, base[k], N, U, R) - fresh
            res[k] = {"params": params, "huber": hub, "mean_penalty_multi": float(np.mean(pen[multi]))}
        if res["raw"] and res["ts"]:
            if "P" in res["raw"]["params"]:
                res["rho_P"] = res["ts"]["params"]["P"] / res["raw"]["params"]["P"]
            res["rho_mean_penalty"] = res["ts"]["mean_penalty_multi"] / res["raw"]["mean_penalty_multi"]
        out[form] = res
        print(form, json.dumps(res, default=float)[:400], flush=True)

    # S4
    cells = []
    for c in sorted(set(cell)):
        idx = [i for i in range(len(names)) if cell[i] == c]
        if len(idx) < 2:
            continue
        last = max(idx, key=lambda i: ep[i])
        rise = {k: float(L[k][last] - min(L[k][i] for i in idx)) for k in L}
        if rise["raw"] > 0:
            cells.append({"N": float(c[0]), "U_D": float(c[1]), "runs": [names[i] for i in idx],
                          "has_diverged": bool(any(div[i] for i in idx)),
                          "max_epochs": float(ep[last]), "rise_raw": rise["raw"], "rise_ts": rise["ts"],
                          "phi": 1 - rise["ts"] / rise["raw"]})
    clean = [c["phi"] for c in cells if not c["has_diverged"]]
    out["S4"] = {"cells": cells, "median_phi": float(np.median([c["phi"] for c in cells])) if cells else None,
                 "median_phi_excl_diverged_cells": float(np.median(clean)) if clean else None}
    for c in cells:
        print(f"S4 N={c['N']:.3g} U={c['U_D']:.3g} ep={c['max_epochs']:.0f} rise_raw={c['rise_raw']:.4f} rise_ts={c['rise_ts']:.4f} phi={c['phi']:.3f}")
    print("S4 median phi", out["S4"]["median_phi"], "excluding cells with diverged runs", out["S4"]["median_phi_excl_diverged_cells"])
    json.dump(out, open(out_p, "w"), indent=1, default=float)

if __name__ == "__main__":
    main()
