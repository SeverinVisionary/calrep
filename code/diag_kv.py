"""Diagnostic for D5 (PR #1152 review): is the inferred head grouping right for the
inferred-architecture families? Scores one run's first NSEQ sequences with every head
split (heads x kv) consistent with the QKV shape and prints raw NLL for each.

usage: python -I diag_kv.py REPO DIR STEP EVAL_DATA_DIR [NSEQ]
"""
import os, sys, tempfile
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calrep_eval as ce  # noqa: E402

repo, d, step, data = sys.argv[1:5]
nseq = int(sys.argv[5]) if len(sys.argv) > 5 else 16
torch.set_num_threads(4)
ce.LOGT = ce.make_grid(0.6)
from huggingface_hub import HfApi  # noqa: E402
work = tempfile.mkdtemp(prefix="diagkv_")
ck = f"{d}/global_step{step}"
for f in HfApi().list_repo_tree(repo, path_in_repo=ck):
    if f.path.split("/")[-1].startswith("layer_") and "model_states" in f.path:
        ce.hf_get(repo, f.path, work)
emb, blocks, final = ce.load_shards(os.path.join(work, ck))
inner = blocks[0]["self_attention.query_key_value.weight"].shape[0] // 3
tokens = np.load(os.path.join(data, "tokens.npy"))[:nseq]
half = np.load(os.path.join(data, "half.npy"))[:nseq]
print(f"{d}: d_model={emb['word_embeddings.weight'].shape[1]} attention inner={inner}")
for kv in [k for k in range(8, inner + 1) if inner % k == 0]:
    raw, _, cnt, _ = ce.score(tokens, half, emb, blocks, final, {"heads": inner // kv, "kv": kv}, 4)
    print(f"  heads={inner // kv:3d} kv={kv:3d}  raw NLL={raw.sum() / cnt.sum():.4f}", flush=True)
