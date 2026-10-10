"""Recompute results/harness_check.json from final.json and the published losses.
Preregistered rule: Spearman(raw, published) >= 0.99 and |delta - median delta| <= 0.03 for >= 95% of runs.
D7 (post hoc): Spearman >= 0.99 and |residual of delta from its OLS line on published| <= 0.05 for >= 95% of runs.
delta = raw NLL - published final validation loss. Shares are rounded to 3 decimals, as in the shipped file.
usage: python3 -I harness_check.py results/final.json inventory/datablations_published_val_losses.tsv out.json"""
import csv, json, sys
import numpy as np
from scipy.stats import spearmanr

final = json.load(open(sys.argv[1]))
pub = {r[0]: float(r[1]) for r in csv.reader(open(sys.argv[2]), delimiter="\t")}
names = sorted(final)
raw = np.array([final[n]["raw_nll"] for n in names]); P = np.array([pub[n] for n in names])
D = raw - P; med = float(np.median(D))
pre_ok = np.abs(D - med) <= 0.03
b = np.polyfit(P, D, 1); d7_ok = np.abs(D - np.polyval(b, P)) <= 0.05
rho = float(spearmanr(raw, P).correlation)
out = {"n": len(names), "spearman": rho, "median_delta": med,
       "prereg_within_0.03": round(float(pre_ok.mean()), 3), "prereg_pass": bool(rho >= 0.99 and pre_ok.mean() >= 0.95),
       "d7_slope": float(b[0]), "d7_intercept": float(b[1]),
       "d7_within_0.05": round(float(d7_ok.mean()), 3), "d7_pass": bool(rho >= 0.99 and d7_ok.mean() >= 0.95),
       "prereg_fail_runs": [n for n, ok in zip(names, pre_ok) if not ok],
       "d7_fail_runs": [n for n, ok in zip(names, d7_ok) if not ok]}
json.dump(out, open(sys.argv[3], "w"), indent=1)
print(out["prereg_within_0.03"], out["d7_within_0.05"], out["d7_fail_runs"])
