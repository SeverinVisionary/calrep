"""CALREP Step 2 (PREREGISTRATION.md; D7): refit Lovelace's additive_1p on our raw
and TS-NLL for the Step 1 run set, and bootstrap rho = P_TS / P_raw.

Run set: Step 0 definition (c) restricted to step1_runset.tsv in_step1 == yes.
Fit per loss: Phase 1 Chinchilla on single-epoch runs, Phase 2 additive_1p with the
base locked (step0_refit's fitter, unchanged). Bootstrap: Lovelace's
stratified_resamples (within model size), 1000 resamples, seed 42; each resample
refits both phases for both losses on the same rows (preregistered).

Harness gate (PREREGISTRATION Step 1; D7): computed here from final.json and the
published losses before any fit; the script stops unless Spearman >= 0.99 and >= 95% of
runs have |residual of (raw - published) from its OLS line on published| <= 0.05.

usage: python -I step2_fit.py NOTEBOOK RUNSET final.json out.json --published TSV
       [--boot 1000] [--shard i k] [--workers W] [--point-only]
A shard computes resamples i, i+k, ... of the full seeded list, so shards merge.
"""
import argparse, csv, json, os, sys
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step0_refit as s0  # noqa: E402

def stratified_resamples(N, n_bootstrap, seed):
    # Lovelace dcscaling/fit.py stratified_resamples, verbatim logic
    rng = np.random.RandomState(seed)
    strata = [np.where(N == n)[0] for n in np.unique(N)]
    return [np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
            for _ in range(n_bootstrap)]

def harness(raw, pub):
    from scipy.stats import spearmanr
    D = raw - pub
    b = np.polyfit(pub, D, 1)
    share = float(np.mean(np.abs(D - np.polyval(b, pub)) <= 0.05))
    rho = float(spearmanr(raw, pub).correlation)
    return {"spearman": rho, "d7_share_within_0.05": share, "pass": rho >= 0.99 and share >= 0.95}

def fit(N, U, ep, y):
    R = ep - 1
    single = np.isclose(ep, 1.0, atol=1e-6) | (ep < 1.0 + 1e-6)
    if single.sum() < 5:
        return None
    _, base = s0.fit_chinchilla(N[single], (U * ep)[single], y[single])
    if base is None:
        return None
    _, P = s0.fit_p1(base, N, U, R, y)
    return {"P": P, "base": dict(zip(["E", "A", "alpha", "B", "beta"], base))}

_S = {}

def _init(state):
    import torch
    torch.set_num_threads(1)
    _S.update(state)

def _boot(task):
    b, idx = task
    s = _S
    out = {"b": b}
    for k in ("raw", "ts"):
        f = fit(s["N"][idx], s["U"][idx], s["ep"][idx], s[k][idx])
        out[k] = None if f is None else f["P"]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook"); ap.add_argument("runset"); ap.add_argument("final"); ap.add_argument("out")
    ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--shard", type=int, nargs=2, default=[0, 1]); ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--point-only", action="store_true")
    ap.add_argument("--published", required=True, help="datablations_published_val_losses.tsv")
    a = ap.parse_args()

    s0.load_notebook(a.notebook)
    rows, skipped = s0.rows(s0.LOSS["cell1"], drop_seeds=True, drop_variants=True)
    rows = [r for r in rows if r["epochs"] <= 64 + 1e-6]
    keep = {r["name"] for r in csv.DictReader(open(a.runset), delimiter="\t") if r["in_step1"] == "yes"}
    final = json.load(open(a.final))
    use, missing = [], []
    for r in rows:
        if r["name"] not in keep:
            continue
        f = final.get(r["name"])
        if not f or not f.get("complete") or f.get("ts_nll") is None:
            missing.append(r["name"]); continue
        use.append({**r, "raw": f["raw_nll"], "ts": f["ts_nll"]})
    if missing:
        raise SystemExit(f"runs without a complete result: {missing}")
    N = np.array([r["N"] for r in use]); U = np.array([r["U_D"] for r in use]); ep = np.array([r["epochs"] for r in use])
    raw = np.array([r["raw"] for r in use]); ts = np.array([r["ts"] for r in use])
    pubd = {r[0]: float(r[1]) for r in csv.reader(open(a.published), delimiter="\t")}
    hc = harness(raw, np.array([pubd[r["name"]] for r in use]))
    if not hc["pass"]:
        raise SystemExit(f"harness gate failed, no fit: {hc}")
    import hashlib
    here = os.path.dirname(os.path.abspath(__file__))
    # identity of the inputs and of the fitter source itself (not the repo HEAD, which result commits advance)
    fitter = hashlib.sha256(b"".join(open(os.path.join(here, f), "rb").read() for f in ("step2_fit.py", "step0_refit.py")))
    out = {"n": len(use), "n_single": int((ep < 1.0 + 1e-6).sum()), "harness": hc, "runs": [r["name"] for r in use],
           "final_sha256": hashlib.sha256(open(a.final, "rb").read()).hexdigest(), "fitter_sha256": fitter.hexdigest()}

    if a.shard == [0, 1]:
        import torch
        torch.manual_seed(0); torch.set_num_threads(4)
        fr, ft = fit(N, U, ep, raw), fit(N, U, ep, ts)
        out.update({"raw": fr, "ts": ft, "rho": ft["P"] / fr["P"]})
        print(f"n={out['n']} single={out['n_single']} P_raw={fr['P']:.6g} P_TS={ft['P']:.6g} rho={out['rho']:.4f}", flush=True)
    if not a.point_only:
        res = stratified_resamples(N, a.boot, a.seed)
        i, k = a.shard
        tasks = [(b, res[b]) for b in range(i, a.boot, k)]
        state = {"N": N, "U": U, "ep": ep, "raw": raw, "ts": ts}
        if a.workers > 1:
            with mp.get_context("spawn").Pool(a.workers, initializer=_init, initargs=(state,)) as pool:
                draws = pool.map(_boot, tasks, chunksize=1)
        else:
            _init(state); draws = [_boot(t) for t in tasks]
        out["boot"] = {"n_bootstrap": a.boot, "seed": a.seed, "shard": a.shard, "draws": draws}
    json.dump(out, open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
