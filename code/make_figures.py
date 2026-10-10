"""Draw the release figures from the shipped result files (matplotlib, Agg backend).
Epoch counts come from the datablations notebook, parsed exactly as in the fits (step0_refit.rows).
usage: python3 -I code/make_figures.py inputs/parametric_fit.ipynb figures/"""
import csv, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import step0_refit as s0  # noqa: E402

nb, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
J = lambda p: json.load(open(os.path.join(ROOT, p)))
final, point, sens, hc = J("results/final.json"), J("results/step2_point.json"), J("results/step2_sens.json"), J("results/harness_check.json")
boot = J("results/step2_boot.json")
draws = [d for i in range(12) for d in J(f"results/boot/boot_{i}.json")["boot"]["draws"]]
rho_b = np.array([d["ts"] / d["raw"] for d in draws if d.get("raw") and d.get("ts")])
pub = {r[0]: float(r[1]) for r in csv.reader(open(os.path.join(ROOT, "inventory/datablations_published_val_losses.tsv")), delimiter="\t")}
s0.load_notebook(nb)
rows, _ = s0.rows(s0.LOSS["cell1"], drop_seeds=True, drop_variants=True)
use = [dict(r, **final[r["name"]]) for r in rows if r["name"] in point["runs"]]
assert len(use) == point["n"] == 125, len(use)
diverged = set(sens["S1"]["diverged"]); d7fail = set(hc["d7_fail_runs"])
BLUE, ORANGE, GREEN, RED, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#888888"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})


def save(fig, name):
    fig.savefig(os.path.join(out, name), bbox_inches="tight"); plt.close(fig); print("wrote", name)


# 1. pipeline
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#333333", lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.6)
def arrow(x0, y0, x1, y1):
    ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.1))
box(0.1, 3.2, 2.3, 1.1, "125 public checkpoints\nMuennighoff et al., C4\nN 7M–2.8B, 1–64 epochs", "#DCEBF7")
box(0.1, 1.6, 2.3, 1.1, "Evaluation slice\nC4 validation shard 00000\n1,048,576 tokens, halves A / B", "#DCEBF7")
box(3.0, 2.3, 2.4, 1.3, "Evaluate each run\nraw NLL\nT_A on A → score B\nT_B on B → score A\n= TS-NLL", "#FFF1D6")
box(3.0, 0.4, 2.4, 1.2, "Harness check vs\npublished losses\nprereg: FAIL (86.4%)\nD7 post hoc: pass (98.4%)", "#F9D9D0")
box(6.0, 2.3, 2.3, 1.3, "Lovelace two-phase fit\n1: Chinchilla, 1 epoch\n2: penalty P·R·N/U\nraw → P_raw\nTS → P_TS", "#DFF2EA")
box(6.0, 0.4, 2.3, 1.2, "Sensitivities S1–S4\ndrop diverged runs,\nother penalty forms,\nper-cell rise", "#EEEEEE")
box(8.9, 2.3, 2.0, 1.3, "ρ = P_TS / P_raw\n= 0.682\n95% CI 0.580–0.858\n(1000 bootstrap)", "#DFF2EA")
box(8.9, 0.4, 2.0, 1.2, "Decision table\nρ≤0.5, upper<0.7: continue\nρ≥0.7: kill\nelse: report (here)", "#EEEEEE")
arrow(2.4, 3.75, 3.0, 3.2); arrow(2.4, 2.15, 3.0, 2.7); arrow(5.4, 2.95, 6.0, 2.95); arrow(4.2, 2.3, 4.2, 1.6)
arrow(5.4, 1.0, 6.0, 2.4); arrow(8.3, 2.95, 8.9, 2.95); arrow(9.9, 2.3, 9.9, 1.6); arrow(7.15, 2.3, 7.15, 1.6)
ax.text(5.5, 4.45, "CALREP pipeline", ha="center", fontsize=12, weight="bold")
save(fig, "fig1_pipeline.png")

# 2. temperature and TS gain vs epochs
ep = np.array([r["epochs"] for r in use]); N = np.array([r["N"] for r in use])
T = np.array([(r["T_A"] + r["T_B"]) / 2 for r in use]); gain = np.array([r["raw_nll"] - r["ts_nll"] for r in use])
fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
for a, y, lab in ((axs[0], T, "fitted temperature T (mean of T_A, T_B)"), (axs[1], np.maximum(gain, 1e-5), "TS gain: raw NLL − TS-NLL (nats)")):
    sc = a.scatter(ep, y, c=np.log10(N), cmap="viridis", s=26, edgecolor="k", lw=0.3)
    for i, r in enumerate(use):
        if r["name"] in d7fail:
            a.annotate(r["name"] + " (fails D7)", (ep[i], y[i]), xytext=(-95, -14 if r["name"] == "1b13b9100m" else 2),
                       textcoords="offset points", fontsize=7, arrowprops=dict(arrowstyle="-", color=GREY, lw=0.6))
    a.set_xscale("log"); a.set_xlabel("epochs (log scale)"); a.set_ylabel(lab)
axs[0].axhline(1, color=GREY, ls="--", lw=1)
axs[0].text(0.98, 0.17, "T > 1: softening lowers loss\nT = 1 (dashed): scaling changes nothing\nT < 1: sharpening lowers loss", transform=axs[0].transAxes,
            ha="right", fontsize=7.5, color="#555555")
axs[1].set_yscale("log"); axs[1].text(0.02, 0.02, "gains ≤ 1e-5 drawn at 1e-5", transform=axs[1].transAxes, fontsize=7, color=GREY)
cb = fig.colorbar(sc, ax=axs, shrink=0.85); cb.set_label("model size N"); cb.set_ticks([7, 8, 9]); cb.set_ticklabels(["10M", "100M", "1B"])
fig.suptitle("Temperature grows with repetition (125 runs)", weight="bold")
save(fig, "fig2_temperature_vs_epochs.png")

# 3. bootstrap distribution of rho with the decision zones
fig, ax = plt.subplots(figsize=(8, 3.8))
ax.axvspan(0.3, 0.5, color=GREEN, alpha=0.12); ax.axvspan(0.5, 0.7, color=ORANGE, alpha=0.12); ax.axvspan(0.7, 1.0, color=RED, alpha=0.12)
ax.hist(rho_b, bins=40, color=BLUE, alpha=0.85)
lo, hi = boot["rho_p2.5"], boot["rho_p97.5"]
ax.axvline(point["rho"], color="k", lw=2); ax.axvline(lo, color="k", ls=":"); ax.axvline(hi, color="k", ls=":")
top = ax.get_ylim()[1]
for x, t in ((0.4, "continue\nρ ≤ 0.5 and\nupper bound < 0.7"), (0.535, "report\n(neither)"), (0.94, "kill\nρ ≥ 0.7")):
    ax.text(x, top * 0.92, t, ha="center", va="top", fontsize=8)
ax.text(0.32, top * 0.55, f"point estimate ρ = {point['rho']:.3f} (solid)\n95% interval {lo:.3f}–{hi:.3f} (dotted)", fontsize=8)
ax.set_xlim(0.3, 1.0); ax.set_xlabel("ρ = P_TS / P_raw"); ax.set_ylabel("bootstrap draws")
ax.set_title(f"Bootstrap of ρ ({len(rho_b)} resamples): {np.mean(rho_b >= 0.7):.1%} ≥ 0.7, {np.mean(rho_b <= 0.5):.0%} ≤ 0.5", weight="bold")
save(fig, "fig3_rho_bootstrap.png")

# 4. sensitivities as a forest plot
items = [("Primary: additive_1p, P ratio", point["rho"], (lo, hi)),
         ("S1: without 11 diverged runs", sens["S1"]["rho"], None),
         ("S2: additive_2p, mean-penalty ratio", sens["additive_2p"]["rho_mean_penalty"], None),
         ("S2: additive_4p, mean-penalty ratio", sens["additive_4p"]["rho_mean_penalty"], None),
         ("S3: Muennighoff R*_D, penalty ratio", sens["muennighoff_d"]["rho_mean_penalty"], None),
         ("S3: Muennighoff R*_D,N, penalty ratio", sens["muennighoff_dn"]["rho_mean_penalty"], None)]
fig, ax = plt.subplots(figsize=(8, 3.4))
for i, (lab, v, ci) in enumerate(items[::-1]):
    if ci: ax.plot(ci, [i, i], color="k", lw=1.5)
    ax.plot(v, i, "o", color=BLUE if ci else GREY, ms=8); ax.text(v + 0.012, i + 0.12, f"{v:.3f}", fontsize=8)
ax.set_yticks(range(len(items))); ax.set_yticklabels([x[0] for x in items[::-1]])
ax.axvline(0.5, color=GREEN, ls="--", lw=1); ax.axvline(0.7, color=RED, ls="--", lw=1); ax.axvline(1.0, color=GREY, lw=0.8)
ax.text(0.505, len(items) - 0.55, "continue line", color=GREEN, fontsize=7.5); ax.text(0.705, len(items) - 0.55, "kill line", color=RED, fontsize=7.5)
ax.set_ylim(-0.6, len(items) - 0.3)
ax.set_xlim(0.45, 1.02); ax.set_xlabel("TS / raw ratio (1 = temperature removes nothing)")
ax.set_title("How much of the penalty survives temperature scaling", weight="bold")
save(fig, "fig4_sensitivities.png")

# 5. S4 cells: post-minimum rise, raw vs TS
cells = sorted(sens["S4"]["cells"], key=lambda c: (c["U_D"], c["N"]))
fmt = lambda x: f"{x / 1e9:.1f}B" if x >= 1e9 else f"{x / 1e6:.0f}M"
lab = [f"N {fmt(c['N'])}\nU {fmt(c['U_D'])}" + ("\n(diverged)" if c["has_diverged"] else "") for c in cells]
x = np.arange(len(cells)); w = 0.38
fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(x - w / 2, [c["rise_raw"] for c in cells], w, label="raw NLL", color=GREY)
ax.bar(x + w / 2, [c["rise_ts"] for c in cells], w, label="TS-NLL", color=BLUE)
for i, c in enumerate(cells):
    ax.text(i, c["rise_raw"] * 1.12, f"φ={c['phi']:.2f}", ha="center", fontsize=8)
ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
ax.set_ylabel("loss rise after the cell minimum (nats, log)"); ax.legend(frameon=False)
ax.set_title("Cells whose loss rises with repetition\nTS removes a share φ of the rise (median 0.45 over the 7 cells without diverged runs)", weight="bold", fontsize=10)
save(fig, "fig5_post_minimum_rise.png")

# 6. harness check
P = np.array([pub[r["name"] if r["name"] in pub else r["name"].replace("21m", "20m")] for r in use]); R = np.array([r["raw_nll"] for r in use])
D = R - P; med = np.median(D); b = np.polyfit(P, D, 1); xs = np.linspace(P.min(), P.max(), 50)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axhspan(med - 0.03, med + 0.03, color=ORANGE, alpha=0.15, label="preregistered: |Δ − median| ≤ 0.03 (86.4% inside)")
ax.fill_between(xs, np.polyval(b, xs) - 0.05, np.polyval(b, xs) + 0.05, color=BLUE, alpha=0.12, label="D7 post hoc: |residual from line| ≤ 0.05 (98.4%)")
ax.plot(xs, np.polyval(b, xs), color=BLUE, lw=1)
ok = np.array([r["name"] not in d7fail for r in use])
ax.scatter(P[ok], D[ok], s=18, color="k"); ax.scatter(P[~ok], D[~ok], s=40, color=RED, marker="x", label="fail D7: " + ", ".join(sorted(d7fail)))
ax.set_xlabel("published final validation loss (Muennighoff et al.)"); ax.set_ylabel("Δ = our raw NLL − published")
ax.legend(fontsize=7.5, frameon=False, loc="upper left"); ax.set_title(f"Harness check: ranks agree (Spearman {hc['spearman']:.5f}), but the offset depends on loss level", weight="bold")
save(fig, "fig6_harness_check.png")
