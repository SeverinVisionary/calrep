# CALREP-P results

**Verdict: C note, exploratory.** Cross-fitted temperature scaling removes about a third of Lovelace's additive overfitting
coefficient on the Muennighoff C4 sweep: ρ = P_TS / P_raw = **0.682** (bootstrap 95% interval 0.580–0.858). That lies
between the continue line (ρ ≤ 0.5) and the kill line (ρ ≥ 0.7), so the preregistered decision table gives "report as a
C note and close". The interval lies wholly above the continue line and straddles the kill line: 48% of draws
are ≥ 0.7, and none is ≤ 0.5. The sensitivities lean towards the kill side: excluding diverged runs gives ρ = 0.81, and other
penalty forms remove 13–23% of the fitted penalty. In the cells where loss actually rises after its minimum, TS removes
about half of the rise (median φ = 0.45 over the 7 cells without diverged runs; 0.47 over all 8).

**Status of the verdict.** The preregistered harness check failed, so under the preregistration Step 2 should not have
run. The owner chose to proceed under D7, a post-hoc harness criterion fixed before any Step 2 statistic was computed.
Every Step 2 number here is therefore exploratory, not a preregistered confirmation or kill. See DEVIATIONS.md D7.

## Step 0: target identified

Lovelace Table 15 was reproduced exactly from datablations' published losses. Definition (c), cell 1 with no seeds and
no variants and ≤ 64 epochs: n = 158, P = 0.0028571 against the published 0.002857. `step0/step0_results.json`.

## Step 1: 125 checkpoints evaluated

- 125 of the 158 runs: N ≤ 2.8B with a checkpoint on the Hub (D4).
- Each run was scored on 1,048,576 tokens of C4 validation (shard 00000, GPT-2 tokens, 512 × 2048), with cross-fitted
  T over document halves. `results/final.json`.
- All 125 complete; no T at a grid edge. `1b15b9100m` needed the wider grid (T = 1.81).
- Compute: 67 Claude Code cloud routines on CPU, plus local fits. No paid GPU.
- Three Hub layout cases needed evaluator fixes before scoring: nested checkpoint folder (21 runs), nested sbatch
  (1 run), and inferred 25M architecture (D5). They are infrastructure only; the harness check covers correctness.

**Harness check** (`results/harness_check.json`):

| Rule | Spearman | Share within tolerance | Pass |
|---|---|---|---|
| Preregistered: \|Δ − median\| ≤ 0.03 | 1.0000 | 86.4% (needs 95%) | **no** |
| D7 (post hoc): \|residual from Δ ~ published\| ≤ 0.05 | 1.0000 | 98.4% | yes |

Δ = raw − published has median −0.046. It is close to linear in published loss (slope +0.014). For the 100 runs with
published loss < 4.5, its sd is 0.006. The 17 runs that fail the preregistered rule are undertrained, high-loss runs.
This pattern fits a slice-difficulty difference that depends on loss level, not a model error. The D6 slice diagnostic
had kept the preregistered tolerance: slice-to-slice change was ≤ 0.016. D7 fails: `1b13b9100m`, `1b15b9100m`.

## Step 2: refit and decision (exploratory)

| | P (additive_1p) | Phase 1 base (E, A, α, B, β) |
|---|---|---|
| Raw NLL | 0.003633 | 1.987, 565.0, 0.347, 18185, 0.454 |
| TS-NLL | 0.002478 | 1.987, 578.9, 0.349, 18179, 0.454 |

- n = 125, of which 27 are single-epoch.
- **ρ = 0.682.** Stratified pairs bootstrap, 1000 resamples, seed 42, both phases refit per resample, all 1000 succeeded:
  - median 0.697;
  - 95% interval 0.580–0.858 (2.5th–97.5th percentile; 95th percentile 0.832);
  - P_raw interval 0.00277–0.00445, P_TS interval 0.00215–0.00284;
  - share of draws ≥ 0.7: 48.4%; share ≤ 0.5: 0%.
- The continue condition fails on both counts: ρ > 0.5, and the upper bound 0.858 ≥ 0.7. Kill does not trigger on the point
  estimate either: 0.682 < 0.7.
- Decision table: neither kill (ρ ≥ 0.7) nor continue (ρ ≤ 0.5) → C note.
- The Phase 1 base barely moves, so the change in P comes from the multi-epoch runs.
  - Single-epoch runs have T = 0.69–1.03. The smallest models are underconfident; the rest are near calibrated.
  - 23 of the 98 multi-epoch runs have T > 1.1, with a maximum of 1.81.
  - The median TS gain over all runs is 0.004 nats.

`results/step2_point.json`, `results/step2_boot.json`.

## Sensitivities (reported, not decisive)

| | Result |
|---|---|
| S1: exclude 11 diverged runs (> 1 nat above the cell minimum over the 158 accepted runs) | ρ = **0.811**, n = 114 |
| S2: additive_2p, ratio of mean fitted penalty over multi-epoch runs (TS / raw) | 0.797 (P ratio 0.870) |
| S2: additive_4p, same | 0.771 (P alone is not comparable: it trades off against δ, κ, γ) |
| S3: muennighoff_d, R*_D raw → TS | 23.8 → 27.6 (penalty ratio 0.866) |
| S3: muennighoff_dn, R*_D raw → TS | 46.1 → 50.6 (penalty ratio 0.852) |
| S4: φ in the 8 cells with a post-minimum rise ≤ 64 epochs | median **0.45** over the 7 cells without diverged runs (0.47 over all 8); range 0.39–0.64 |

S4 by cell, all at 51–61 epochs. Each line is N, U_D, rise raw → rise TS, φ:
- 220M, 100M: 0.057 → 0.026, 0.54
- 281M, 100M: 0.146 → 0.074, 0.50
- 421M, 100M: 0.410 → 0.227, 0.45
- 619M, 100M: 0.851 → 0.394, 0.54
- 1.1B, 100M: 2.133 → 0.774, 0.64 (contains S1-diverged runs; flagged, excluded from the 0.45 median)
- 1.5B, 1.5B: 0.038 → 0.023, 0.41
- 2.2B, 1.5B: 0.147 → 0.090, 0.39
- 2.8B, 1.5B: 0.228 → 0.131, 0.43

`results/step2_sens.json`. Definitions are in the `code/step2_sens.py` docstring, fixed before the first run. One
correction after PR #1152 review: S1 cell minima now span the 158 accepted runs, not the 125 evaluated ones. This adds
`2b8100m100m`, whose cell-mate `2b84b8100m` has no checkpoint; ρ_S1 was 0.796 before the correction. A second
addition: S4 cells containing a diverged run are flagged, and the median is also reported without them, following
CONTEXT.md's "flagged, not pooled".

## Reading

1. **What it says.** Under repetition, part of the held-out loss rise is overconfidence that one scalar temperature
   removes. In rising cells that is about half of the post-minimum rise (φ 0.39–0.64, no clear trend with N). The
   absolute rise removed grows with N. Fitted multi-epoch temperatures reach 1.81.
2. **What it does not say.** The fitted overfitting coefficient is mostly not a calibration term:
   - 68% of P survives TS on the primary form (bootstrap 58–86%), 81% without the diverged runs, and 77–87% across
     other forms.
   - The primary ρ is pulled down by a few heavily overfit, badly miscalibrated runs. `1b15b9100m` drops 1.37 nats
     under TS alone.
3. **Limits.**
   - Exploratory status (D7).
   - Muennighoff's sweep only. Lovelace's FineWeb sweep, which carries their headline P = 0.023, has no released
     checkpoints.
   - A 1M-token eval slice.
   - N ≤ 2.8B.
   - TS is one global scalar per run. Richer recalibration could remove more, which bounds ρ from above only.

## Spend

No paid GPU. 67 cloud routines (owner-approved, ≤ 10 concurrent) and local CPU fits. Step 2 bootstrap: 4 routines (b3 reran 2 shards lost to cloud worker restarts, exit 137).
