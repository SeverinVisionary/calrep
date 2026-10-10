#!/bin/bash
# Recompute the shipped analysis outputs from the shipped evaluator outputs and compare (coverage: README).
# Needs inputs/parametric_fit.ipynb (run fetch_inputs.sh first). CPU only.
#   bash reproduce.sh            Step 0, run set, final.json from parts, harness check, Step 2 point, sensitivities,
#                                and the merge of the shipped bootstrap shards
#   bash reproduce.sh --boot     also rerun the 1000-resample bootstrap (many CPU-hours; uses all cores)
# Re-running Step 1 itself (125 checkpoints from the Hugging Face Hub) is described in README.md.
set -euo pipefail
cd "$(dirname "$0")"
NB=inputs/parametric_fit.ipynb; PUB=inventory/datablations_published_val_losses.tsv
[ -f "$NB" ] || { echo "missing $NB: run fetch_inputs.sh" >&2; exit 1; }
mkdir -p reproduced
cmpj() { python3 -I code/compare_json.py "$@"; }
# keys step2_fit.py gained after the shipped point and bootstrap files were computed (README, errata)
LATER="harness final_sha256 fitter_sha256"

python3 -I code/test_compare_json.py
python3 -I code/step0_refit.py "$NB" reproduced/step0_results.json
cmpj reproduced/step0_results.json step0/step0_results.json
python3 -I code/step1_runset.py step0/step0_results.json inventory reproduced/step1_runset.tsv
cmp reproduced/step1_runset.tsv step0/step1_runset.tsv
echo "step1_runset.tsv: identical"
python3 -I code/finalize.py results/parts reproduced/final.json
cmp reproduced/final.json results/final.json
echo "final.json: identical"
python3 -I code/harness_check.py results/final.json "$PUB" reproduced/harness_check.json
cmpj reproduced/harness_check.json results/harness_check.json
python3 -I code/step2_fit.py "$NB" step0/step1_runset.tsv results/final.json reproduced/step2_point.json \
  --point-only --published "$PUB"
cmpj reproduced/step2_point.json results/step2_point.json --ignore $LATER
python3 -I code/step2_sens.py "$NB" step0/step1_runset.tsv results/final.json "$PUB" reproduced/step2_sens.json
cmpj reproduced/step2_sens.json results/step2_sens.json
python3 -I code/boot_merge.py results/step2_point.json reproduced/step2_boot.json results/boot/boot_*.json
cmpj reproduced/step2_boot.json results/step2_boot.json

if [ "${1:-}" = "--boot" ]; then
  mkdir -p reproduced/boot
  for i in $(seq 0 11); do
    python3 -I code/step2_fit.py "$NB" step0/step1_runset.tsv results/final.json reproduced/boot/boot_$i.json \
      --shard "$i" 12 --workers "$(python3 -c 'import os;print(os.cpu_count())')" --published "$PUB"
    cmpj reproduced/boot/boot_$i.json results/boot/boot_$i.json --ignore $LATER
  done
fi
echo "done"
