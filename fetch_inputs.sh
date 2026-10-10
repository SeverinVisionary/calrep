#!/bin/bash
# Fetch the two external inputs at pinned revisions and check their sha256.
#   inputs/parametric_fit.ipynb        huggingface/datablations utils/, commit 13701315 (Apache-2.0)
#   eval_data/{tokens,half}.npy        built from allenai/c4 en validation shard 00000, revision 1588ec45 (ODC-BY)
# usage: bash fetch_inputs.sh            (about 40 MB download; needs python with numpy and tiktoken)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p inputs eval_data
NB=https://raw.githubusercontent.com/huggingface/datablations/13701315b163f1c54642b94ad237b2699187c3ee/utils/parametric_fit.ipynb
C4=https://huggingface.co/datasets/allenai/c4/resolve/1588ec454efa1a09f29cd18ddd04fe05fc8653a2/en/c4-validation.00000-of-00008.json.gz
curl -sSfL "$NB" -o inputs/parametric_fit.ipynb
curl -sSfL "$C4" -o inputs/c4-validation.00000-of-00008.json.gz
shasum -a 256 -c <<'EOF'
6c65d270e70c944a9ac90f5f1213a4584e1ff9d6d2781cd86900663f4ab17db9  inputs/parametric_fit.ipynb
1f25b6af12da84115301d4ee93ea5246c8fea5bb4a2008472794d95b917cc97f  inputs/c4-validation.00000-of-00008.json.gz
EOF
python3 -I code/prep_eval_tokens.py inputs/c4-validation.00000-of-00008.json.gz eval_data
shasum -a 256 -c <<'EOF'
247819314140fbdc80569c52813668288424bdf38d6f6456f718659a65ab199d  eval_data/tokens.npy
dbf2d03ff599dafce920119618923f3cbbee5ab62d1d73aad6c9f5df8816f65e  eval_data/half.npy
EOF
