"""Merge sequence-range parts of each run and fit T (PREREGISTRATION.md Step 1, DEVIATIONS D1).
usage: python -I finalize.py results/parts results/runs.json"""
import glob, json, os, sys
from collections import defaultdict
import numpy as np


def fit_from_sums(raw, grid, cnt, LOGT=None):
    from scipy.interpolate import CubicSpline
    from scipy.optimize import minimize_scalar
    raw, grid, cnt = np.asarray(raw), np.asarray(grid), np.asarray(cnt)
    LOGT = np.round(np.arange(-0.6, 0.6001, 0.15), 4) if LOGT is None else np.asarray(LOGT)
    curves = grid / cnt[:, None]
    res, cs, edge = {}, [], False
    for h in (0, 1):
        c = CubicSpline(LOGT, curves[h]); k = int(np.argmin(curves[h]))
        edge |= k in (0, len(LOGT) - 1)
        r = minimize_scalar(c, bounds=(LOGT[max(k - 1, 0)], LOGT[min(k + 1, len(LOGT) - 1)]),
                            method="bounded", options={"xatol": 1e-5})
        cs.append(c); res["T_" + "AB"[h]] = float(np.exp(r.x))
    res["nll_B_at_T_A"] = float(cs[1](np.log(res["T_A"])))
    res["nll_A_at_T_B"] = float(cs[0](np.log(res["T_B"])))
    res["ts_nll"] = float((res["nll_B_at_T_A"] * cnt[1] + res["nll_A_at_T_B"] * cnt[0]) / cnt.sum())
    res["raw_nll"] = float(raw.sum() / cnt.sum())
    res["T_at_grid_edge"] = bool(edge)
    if edge:  # preregistered: an edge minimum is redone on a wider grid, never used as-is
        res["ts_nll"] = None; res["needs_wider_grid"] = True
    res["tokens_scored"] = int(cnt.sum())
    return res


if __name__ == "__main__":
    parts = defaultdict(list)
    for f in glob.glob(os.path.join(sys.argv[1], "*.json")):
        d = json.load(open(f))
        key = (os.path.basename(f).split("__")[0], round(max(d["logT_grid"]), 4), len(d["logT_grid"]))
        parts[key].append(d)
    out = {}
    for (name, gmax, glen), ps in sorted(parts.items(), key=lambda kv: (kv[0][0], -kv[0][1], -kv[0][2])):
        ps.sort(key=lambda d: d["seq_start"])
        cover = [(d["seq_start"], d["seq_end"]) for d in ps]
        complete = cover[0][0] == 0 and cover[-1][1] == 512 and all(a[1] == b[0] for a, b in zip(cover, cover[1:]))
        if name in out and out[name]["complete"]:
            continue  # a wider complete grid was already taken
        shas = {d.get("code_sha") for d in ps}  # parts before 2026-10-09 carry none (provenance: shard branch bases)
        if len(shas) > 1:
            raise SystemExit(f"{name}: parts from different evaluator commits {sorted(map(str, shas))}; refusing to combine")
        raw = np.sum([d["sum_raw"] for d in ps], 0); grid = np.sum([d["sum_grid"] for d in ps], 0)
        cnt = np.sum([d["count"] for d in ps], 0)
        r = fit_from_sums(raw, grid, cnt, ps[0]["logT_grid"])
        r.update({"complete": complete, "parts": cover, "logt_max": gmax, "arch": ps[0]["arch"], "code_sha": shas.pop(),
                  "exact_check": [d["exact_check"] for d in ps if "exact_check" in d]})
        out[name] = r
    json.dump(out, open(sys.argv[2], "w"), indent=1)
    edge = [k for k, v in out.items() if v.get("needs_wider_grid")]
    print(len(out), "runs;", sum(v["complete"] for v in out.values()), "complete;", len(edge), "need a wider grid:", edge)
