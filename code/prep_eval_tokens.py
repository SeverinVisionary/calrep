"""Build the fixed CALREP evaluation slice (PREREGISTRATION.md Step 1).
allenai/c4 en validation shard 00000, file order, GPT-2 BPE (tiktoken 'gpt2'),
EOD (50256) appended per document, concatenated to 512 * 2048 tokens.
Writes tokens.npy (int32, [512, 2048]) and half.npy (uint8, [512, 2048]):
half of the document each token belongs to (0 = even doc index -> A, 1 = odd -> B).
usage: python -I prep_eval_tokens.py c4-validation.00000-of-00008.json.gz outdir"""
import gzip, json, sys, hashlib
import numpy as np
import tiktoken
enc = tiktoken.get_encoding("gpt2")
NSEQ, L = 512, 2048
need = NSEQ * L
toks, half = [], []
with gzip.open(sys.argv[1], "rt") as f:
    for i, line in enumerate(f):
        ids = enc.encode_ordinary(json.loads(line)["text"]) + [50256]
        toks.extend(ids); half.extend([i % 2] * len(ids))
        if len(toks) >= need:
            break
t = np.asarray(toks[:need], dtype=np.int32).reshape(NSEQ, L)
h = np.asarray(half[:need], dtype=np.uint8).reshape(NSEQ, L)
np.save(sys.argv[2] + "/tokens.npy", t); np.save(sys.argv[2] + "/half.npy", h)
print("docs used", i + 1, "tokens", t.size, "half-B share", round(float(h.mean()), 4),
      "sha256", hashlib.sha256(t.tobytes()).hexdigest()[:16])
