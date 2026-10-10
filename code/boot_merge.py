"""Merge CALREP Step 2 bootstrap shards into results/step2_boot.json.

Upper bound per D7 addendum: 97.5th percentile of rho_b (95th also reported). A resample
with a failed fit on either loss is dropped and counted.

usage: python -I boot_merge.py step2_point.json out.json boot_*.json [--allow-partial]
Refuses to write bounds unless every resample index 0..n_bootstrap-1 is present.
"""
import json, sys
import numpy as np

allow_partial = "--allow-partial" in sys.argv
args = [a for a in sys.argv[1:] if a != "--allow-partial"]
point = json.load(open(args[0]))
draws, meta = {}, set()
for p in args[2:]:
    d = json.load(open(p))
    # same experiment: run list, input checksum and fitter commit (the last two absent before 2026-10-10)
    meta.add((d["boot"]["n_bootstrap"], d["boot"]["seed"], d["boot"]["shard"][1],
              json.dumps(d["runs"]), d.get("final_sha256"), d.get("fitter_sha256")))
    for x in d["boot"]["draws"]:
        if x["b"] in draws:
            raise SystemExit(f"duplicate resample {x['b']} in {p}")
        draws[x["b"]] = x
if len(meta) != 1:
    raise SystemExit(f"shards disagree on n_bootstrap/seed/k/run list/input checksum/fitter source ({len(meta)} variants)")
nb, seed, k, runs, fsha, csha = meta.pop()
# the point estimate must come from the same experiment as the draws
if json.dumps(point["runs"]) != runs:
    raise SystemExit("point estimate and bootstrap shards have different run lists")
for key, v in (("final_sha256", fsha), ("fitter_sha256", csha)):
    if (point.get(key) or v) and point.get(key) != v:
        raise SystemExit(f"point estimate and bootstrap shards disagree on {key}")
missing = sorted(set(range(nb)) - set(draws))
if missing and not allow_partial:
    raise SystemExit(f"{len(missing)} of {nb} resamples missing (first: {missing[:5]}); rerun those shards")
ok = [x for x in draws.values() if x["raw"] and x["ts"] and np.isfinite(x["raw"]) and np.isfinite(x["ts"])]
rho = np.array([x["ts"] / x["raw"] for x in ok])
out = {"n_bootstrap": nb, "seed": seed, "n_resamples_present": len(draws), "n_success": len(ok),
       "missing": sorted(set(range(nb)) - set(draws)),
       "rho_point": point["rho"], "rho_median": float(np.median(rho)),
       "rho_p2.5": float(np.percentile(rho, 2.5)), "rho_p5": float(np.percentile(rho, 5)),
       "rho_p95": float(np.percentile(rho, 95)), "rho_p97.5": float(np.percentile(rho, 97.5)),
       "share_rho_ge_0.7": float(np.mean(rho >= 0.7)), "share_rho_le_0.5": float(np.mean(rho <= 0.5)),
       "P_raw_p2.5_p97.5": [float(np.percentile([x["raw"] for x in ok], q)) for q in (2.5, 97.5)],
       "P_ts_p2.5_p97.5": [float(np.percentile([x["ts"] for x in ok], q)) for q in (2.5, 97.5)]}
json.dump(out, open(args[1], "w"), indent=1)
print({k: v for k, v in out.items() if k != "missing"}, "missing", len(out["missing"]))
