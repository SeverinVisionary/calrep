# CALREP-P preregistration

Frozen 2026-10-07, after `GATE_PASSED` (owner decision; see `candidate.json`) and before any Step 0 fit was run. Any
change after this commit goes in `DEVIATIONS.md` with its reason. Context: `CONTEXT.md`.

## Question and hypothesis

Lovelace et al. 2605.01640 fit an additive overfitting coefficient P (Eq. 6, one-parameter form) on Muennighoff's public
C4 sweep (Table 15: P = 0.002857, n = 158, up to 64 epochs). The question is how much of P is removed when each run's
held-out NLL is replaced by its cross-fitted temperature-scaled NLL (TS-NLL).

**H1:** ρ = P_TS / P_raw ≤ 0.5.

Both P values come from our own evaluation tokens, so the comparison is like for like. The published P is used only in
Step 0.

## Step 0: identify the run set (CPU-seconds, local, published numbers only)

**Inputs:**
- The datablations published final-loss dictionaries and `PARAMS_MAP` from `utils/parametric_fit.ipynb`, cells 1 and 25.
- Token counts and unique budgets parsed with that notebook's own `get_unique_tokens` and `TOKENS_MAP` logic, not a new
  parser.

**Procedure:** Lovelace's released `dcscaling/fit.py` (repo `justinlovelace/data-constrained-scaling`, MIT, commit of
2026-10-07), reimplemented from the file as a spec rather than imported from the cloned repo:
- Phase 1 Chinchilla on single-epoch runs. Huber δ = 1e-3 on log loss, L-BFGS with strong Wolfe line search,
  4×3×3×3×3 init grid, max_iter 200.
- Phase 2 `additive_1p` with the base locked. Init grid {1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1, 100}, max_iter 500.

**Candidate run-set definitions.** All are tried and every result is reported:
- **(a)** the cell-1 dictionary, loss > 0, epochs ≤ 64;
- **(b)** (a) minus seed replicates;
- **(c)** (b) minus `dedup`, `opt` and `subopt` variants;
- **(d)** the cell-25 dictionary (outliers removed), epochs ≤ 64.

**Acceptance:** n within ±5 of 158, and P within ±15% of 0.002857. If more than one definition passes, use the one with
the smallest |P − 0.002857|. Ties are broken by |n − 158|.

**Stop rule:** if none passes, stop and report. The target is then unidentified, and contacting the authors needs owner
approval.

## Step 1: evaluate the checkpoints

**Run set:** the accepted Step 0 set, restricted to N ≤ 2.8B. Runs whose checkpoint is missing on the Hub are listed and
dropped. Step 2 refits the raw loss on this same restricted set, so the comparison stays like for like.

**Evaluation tokens:**
- Source: `allenai/c4` `en` validation, in file order from `c4-validation.00000-of-00008.json.gz`.
- Tokenizer: GPT-2, with EOD appended to each document.
- Documents are concatenated until 1,048,576 tokens and packed into 512 sequences of 2,048 tokens.
- Documents with an even index go to half A and odd to half B. Each token belongs to the half of its document.

**Model:** final DeepSpeed checkpoint, model states only, converted with Megatron-DeepSpeed `deepspeed_to_transformers.py`
(tp = pp = 1). Inference in fp32.

**Per run:**
1. Raw NLL is the mean over all tokens.
2. T_A minimizes the mean NLL of softmax(z / T) on half A. Search log T on [−1.5, 1.5] by bounded scalar minimization to
   1e-4.
3. T_B is fitted the same way on half B.
4. TS-NLL is the token-weighted mean of NLL_B(T_A) and NLL_A(T_B), as in Ashukha 2002.06470 §3.5.

T_A and T_B are both stored.

**Harness check (must pass before Step 2):**
- Let Δ_i = raw_i − published_i.
- Require Spearman(raw, published) ≥ 0.99.
- Require |Δ_i − median Δ| ≤ 0.03 nats for ≥ 95% of runs.
- Failing runs are listed. If the aggregate check fails, stop: the harness is wrong.

**Divergence flag**, fixed now: a run is flagged when its published loss exceeds the minimum published loss in its
(N, U_D) cell by more than 1.0 nat. Flagged runs stay in the primary analysis, as in Lovelace's fit, and are excluded in
sensitivity S1.

## Step 2: the refit and the decision

Two parallel fits:
- **Raw fit:** Phase 1 on the single-epoch runs' raw NLL, then Phase 2 `additive_1p` on all runs' raw NLL, giving P_raw.
- **TS fit:** the same two phases on TS-NLL, giving P_TS.

**ρ = P_TS / P_raw.** Its uncertainty comes from a stratified pairs bootstrap by model size (Lovelace's
`stratified_resamples`): 1,000 resamples, seed 42. Each resample refits both phases for both losses.

**Decision (primary analysis only):**

| Outcome | Condition | Verdict |
|---|---|---|
| Kill | ρ ≥ 0.7 | Temperature removes less than 30% of the overfitting coefficient. CALREP closes at C. |
| Continue | ρ ≤ 0.5 and bootstrap 95% upper bound < 0.7 | — |
| Otherwise | — | Report as a C note and close. |

**Sensitivity analyses** (reported, not decisive):
- **S1:** excluding diverged runs.
- **S2:** `additive_2p` and `additive_4p`.
- **S3:** Muennighoff App. F alpha-beta-decay law, raw vs TS.
- **S4:** per-cell φ (share of the post-minimum rise removed) for cells that rise within 64 epochs.

## Compute and spend

- Step 0 runs locally (seconds).
- Step 1 starts with one smoke routine: three checkpoints (14M, 146M and 1.1B), with the harness check on them. Then up
  to 10 Claude Code cloud routines, sharded by model-size band, on CPU, streaming one checkpoint at a time.
- No paid GPU without a separate owner go-ahead (fallback: L4, ≈$6–9).
- No upstream contact or public posting without owner approval.
