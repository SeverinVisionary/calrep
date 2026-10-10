# CALREP deviations from PREREGISTRATION.md

Each entry was logged before any Step 1 result entered the Step 2 refit. The smoke runs, which are used only for the
harness check, are noted where they informed an entry.

**D1. Temperature fit by grid and spline, not bounded scalar minimization.**
- *Why:* exact minimization needs repeated full-vocabulary passes per run. With logits too large to store
  (1M × 50,304), each pass needs either a second forward or stored hidden states.
- *What changed:* NLL(T) is computed per half on a fixed grid of 25 values of log T in [−0.6, 0.6] (step 0.05,
  T = 0.55–1.82), in the same forward pass. T_A and T_B are the minimizers of a cubic spline through that grid.
  NLL_B(T_A) and NLL_A(T_B) are read off the spline.
- *Edge cases:* a run whose minimum falls on a grid edge is flagged (`T_at_grid_edge`) and redone on a wider grid.
- *Verification:* the spline-read cross-fitted values are checked against an exact recompute (`--exact-check`) on at
  least one smoke run. The error is reported in RESULTS.
- *Effect on the decision:* none expected. At the minimum the curve is flat to second order.

**D1 addendum (2026-10-08, Codex review on PR #1152).** A run whose minimum falls on a grid edge gets `ts_nll = null`
and `needs_wider_grid` from `finalize.py`. It is rerun with a `logt_max` column in its units file, for example 1.0
(T = 0.37–2.72). `finalize.py` uses the widest complete grid for each run. No edge-clipped value can enter Step 2.

**D1b (2026-10-08, after the smoke run, before any Step 2 input): grid step 0.15, not 0.05.**
- *Why:* on the 4-vCPU cloud containers the 25-point grid cost about 40 min per run, which dominated every run under
  150M.
- *What changed:* the grid is now 9 points over the same range [−0.6, 0.6], with the spline unchanged.
- *Check:* at step 0.05 the smoke exact-check (7M, full 1M tokens) gave spline − exact = 6e-7 nats. Exact checks are
  scheduled on several production units at step 0.15, and their errors go into RESULTS.
- The smoke parts at step 0.05 stay valid: `finalize.py` groups parts by their own grid.

**D5 (2026-10-08). Inferred architecture for the 25M runs.** `PARAM_25M`, which the 25M sbatch files reference, is in no
version of datablations `utils/model_params.sh`. For any missing entry, d, ffw and the layer count come from the shard
shapes, and heads × kv from the QKV rows. kv follows the table's pattern: 32 for d < 512. For 25M this gives d = 448,
7 heads × 32 (attention width 224 = QKV rows / 3). The same inference applied to the 21M run (`PARAM_21M`, d = 288, 7 × 32); noted 2026-10-09. The harness check validates it, and `code/diag_kv.py` scores every head split consistent with the QKV shape (PR #1152 review).

**D2. The model is run directly from the DeepSpeed pipeline shards, without `deepspeed_to_transformers.py`.**
- *Why:* it avoids a Megatron-DeepSpeed install and a conversion step on every routine.
- *Layout assumed (bigscience Megatron-DeepSpeed GPT):*
  - learned position embeddings;
  - pre-LayerNorm with eps 1e-5;
  - fused QKV laid out per head as [q; k; v];
  - tanh-GELU;
  - logits = h · Eᵀ over the padded 50,304 vocabulary;
  - fp32 inference.
- *Check:* the preregistered harness check against published losses is unchanged and catches any layout error.

**D3. Name alias.** Step 0 run `20m400m400m` (the notebook's `PARAMS_MAP` key) is stored on the Hub as
`lm1-misc/21m400m400m`. It is mapped by alias.

**D4. Step 1 run set: 125 of the 158.**
- 28 runs are over 2.8B: the 3.9B, 4.2B and 8.7B runs.
- Five runs have no checkpoint on the Hub: `14m1b5100m`, `14m2b7100m`, `44m1b5100m`, `2b84b8100m` and `2b84b84b8`.
- The list is in `step0/step1_runset.tsv`.
- The 4.2B and 8.7B seed repositories are not used, because definition (c) excludes seeds.
- Step 2 refits raw and TS on these same 125 runs, as preregistered.

**D6 (2026-10-08, owner decision, before any Step 2 input). Harness-check slice diagnostic.**
- *Trigger:* in the smoke run, Δ = raw − published was +0.023 / +0.028 / +0.008 / −0.048 for 7M / 14M / 44M / 146M. The
  146M run is 0.063 from the median, beyond the 0.03 tolerance. Δ falls with model quality, which suggests our 1M-token
  slice and the published 210M-token set differ in difficulty in a way that depends on the model.
- *Diagnostic:* the same four runs are also scored on a second, disjoint slice (`eval_data_s2/`, built the same way
  from `c4-validation.00001-of-00008.json.gz`).
- *Rule, fixed now:*
  - **(i)** If the slice-to-slice difference in Δ for these runs is at least as large as the 146M deviation (≥ 0.03
    nats for some run), the 0.03 per-run tolerance is mis-specified for a 1M-token slice. The harness is then judged
    on two conditions:
    - Spearman(raw, published) ≥ 0.99, unchanged;
    - for ≥ 95% of runs, the absolute residual of Δ from a straight-line fit of Δ on published loss is ≤ 0.05 nats.
      ("absolute" added 2026-10-09 after PR #1152 review; D7 and its code use |residual|.)
  - **(ii)** Otherwise the preregistered tolerance stands as written.
- *Scope:* the decision statistic ρ is unaffected, because raw and TS are both computed and refit on the same slice.

**D7 (2026-10-09 ~15:40 UTC, owner decision, POST HOC: after the harness data, before any Step 2 statistic). Harness
judged on residuals from a linear fit in published loss; Step 2 is reported as exploratory.**
- *Trigger:* the preregistered harness check failed. Over 119 runs, |Δ − median Δ| ≤ 0.03 for 85.7% (the rule needs
  95%). Spearman was 1.0000. D6 had resolved to branch (ii), so the per-run tolerance stood. See STATUS 2026-10-09
  15:35.
- *Diagnosis:* Δ = raw − published is close to linear in published loss (slope +0.014). The failing runs are the
  undertrained, high-loss runs. For the 100 runs with published loss < 4.5, Δ has sd 0.006. This is consistent with an
  eval-slice difference that depends on loss level, not a model or layout error.
- *Rule now:* the harness passes if Spearman ≥ 0.99 and, for ≥ 95% of runs, |residual of Δ from the OLS line of Δ on
  published loss| ≤ 0.05. This is D6 branch (i)'s criterion, adopted without its trigger. Runs failing it are listed.
  On the 119-run interim the share is 98.3%; the final check uses all runs.
- *Status of the result:* this is a post-hoc change to a preregistered gate. The Step 2 ρ verdict is reported as
  exploratory, not as a preregistered confirmation or kill. Its decision thresholds are applied unchanged and labelled
  as such.
- *Unaffected:* ρ is computed from raw and TS NLL on the same tokens. The level offset against published losses does
  not enter it.

**D7 addendum (2026-10-09 ~16:00 UTC, before any Step 2 statistic). Definitions the preregistration left open.**
- "Bootstrap 95% upper bound" is the 97.5th percentile of the bootstrap ρ draws: the upper end of the two-sided 95%
  percentile interval, which is the more conservative reading. The 95th percentile is also reported.
- A resample whose Phase 1 has fewer than 5 single-epoch runs, or whose fit fails, is dropped and counted.
- Raw and TS fits in a resample use the same rows. ρ_b = P_TS,b / P_raw,b.
