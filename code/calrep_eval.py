"""CALREP Step 1 evaluator (PREREGISTRATION.md): raw and cross-fitted
temperature-scaled NLL of one datablations checkpoint on the fixed C4 slice.

The model is run directly from the Megatron-DeepSpeed pipeline shards
(layer_*-model_00-model_states.pt, bf16 -> fp32), with the bigscience
Megatron-DeepSpeed GPT layout: learned position embeddings, pre-LayerNorm
(eps 1e-5), fused QKV laid out per head as [q_h; k_h; v_h], tanh-GELU,
untied-free output (logits = h @ word_embeddings.T over the padded vocab),
dropout off. Correctness is not assumed: the harness check in Step 1 compares
raw NLL against each run's published final loss.

Each invocation scores a sequence range and writes sums; finalize.py merges
the parts of a run and fits the temperatures. Temperature fit: NLL(T) of each half is evaluated on a fixed grid of 25
values of log T in [-0.6, 0.6] in the same forward pass (a minimum on the grid
edge is flagged and the run is redone on a wider grid); T_A, T_B are the
minimizers of a cubic spline in log T through that grid, and the cross-fitted
values NLL_B(T_A), NLL_A(T_B) are read off the spline. With --exact-check the
cross-fitted values are recomputed exactly at T_A / T_B in a second pass.

usage:
  python -I calrep_eval.py --repo datablations/lm1-misc --dir 14m100m100m \
      --step 190 --data eval_data --out results/14m100m100m.json [--nseq 512]
"""
import argparse, json, os, re, shutil, sys, tempfile, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn.functional as F

# datablations utils/model_params.sh: (d_model, ffw, kv_size, n_heads, n_layers)
# pinned to the last commit touching the file (2023-05-25); unchanged through the Step 1 campaign
PARAMS_SH = "https://raw.githubusercontent.com/huggingface/datablations/74a2610a7ddf285cd828331e805baf4a64b370a0/utils/model_params.sh"
# Hub revisions in force for the whole Step 1 campaign (repos last modified May 2023); inventory/hub_revisions.txt
REVISION = {"datablations/lm1-misc": "ddd985fd2b46aba0b83b0e6a2c6f2af1511c66f0",
            "datablations/lm1-2b8-55b-c4-repetitions": "e4c25f9f9ffb88c4ca9750167078b23f62708e64"}


def make_grid(logt_max=0.6):
    # both endpoints included, step <= 0.15; logt_max 0.6 gives the 9-point D1b grid exactly
    n = int(np.ceil(round(2 * logt_max / 0.15, 6))) + 1
    return np.round(np.linspace(-logt_max, logt_max, n), 4)


LOGT = make_grid()  # 9 points (step 0.15), T in [0.55, 1.82]; --logt-max widens it (DEVIATIONS.md D1)


def hf_get(repo, path, dest):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=path, local_dir=dest, revision=REVISION.get(repo))


def arch_from_sbatch(sbatch_text, params_sh_text):
    m = re.search(r'MODEL_PARAM=\("\$\{(PARAM_\w+)\[@\]\}"\)', sbatch_text)
    if not m:
        raise ValueError("no MODEL_PARAM in sbatch")
    tbl = dict(re.findall(r"^(PARAM_\w+)=\(([^)]*)\)", params_sh_text, re.M))
    if m.group(1) not in tbl:  # e.g. PARAM_25M is in no version of model_params.sh: infer from shards
        return {"param_name": m.group(1), "inferred": True}
    d, ffw, kv, nh, nl = (int(x) for x in tbl[m.group(1)].split())
    return {"param_name": m.group(1), "d": d, "ffw": ffw, "kv": kv, "heads": nh, "layers": nl}


def load_shards(ckdir):
    files = sorted(f for f in os.listdir(ckdir) if f.startswith("layer_") and "model_states" in f)
    emb, blocks, final = None, [], None
    for f in files:
        sd = torch.load(os.path.join(ckdir, f), map_location="cpu", weights_only=True)
        sd = {k: v.float() for k, v in sd.items()}
        if "word_embeddings.weight" in sd:
            emb = sd
        elif "input_layernorm.weight" in sd:
            blocks.append(sd)
        elif set(sd) == {"weight", "bias"}:
            final = sd
        else:
            raise ValueError(f"unrecognised shard {f}: {sorted(sd)}")
    return emb, blocks, final


@torch.inference_mode()
def hidden(ids, emb, blocks, final, heads, kv):
    """ids [B, S] -> final hidden states [B, S, d]."""
    B, S = ids.shape
    x = emb["word_embeddings.weight"][ids] + emb["position_embeddings.weight"][:S][None]
    for p in blocks:
        h = F.layer_norm(x, x.shape[-1:], p["input_layernorm.weight"], p["input_layernorm.bias"], 1e-5)
        qkv = F.linear(h, p["self_attention.query_key_value.weight"], p["self_attention.query_key_value.bias"])
        qkv = qkv.view(B, S, heads, 3 * kv)
        q, k, v = qkv.split(kv, dim=-1)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).reshape(B, S, heads * kv)
        x = x + F.linear(a, p["self_attention.dense.weight"], p["self_attention.dense.bias"])
        h = F.layer_norm(x, x.shape[-1:], p["post_attention_layernorm.weight"], p["post_attention_layernorm.bias"], 1e-5)
        h = F.gelu(F.linear(h, p["mlp.dense_h_to_4h.weight"], p["mlp.dense_h_to_4h.bias"]), approximate="tanh")
        x = x + F.linear(h, p["mlp.dense_4h_to_h.weight"], p["mlp.dense_4h_to_h.bias"])
    return F.layer_norm(x, x.shape[-1:], final["weight"], final["bias"], 1e-5)


@torch.inference_mode()
def score(tokens, half, emb, blocks, final, arch, bs, exact_T=None):
    """Sums over next-token predictions, by half: raw NLL, NLL on the log-T grid,
    and (optionally) NLL at exact temperatures {half: T}."""
    W = emb["word_embeddings.weight"]
    betas = torch.tensor(np.exp(-LOGT), dtype=torch.float32)  # 1/T
    grid = torch.zeros(2, len(LOGT), dtype=torch.float64)
    raw = torch.zeros(2, dtype=torch.float64)
    cnt = torch.zeros(2, dtype=torch.float64)
    exact = torch.zeros(2, dtype=torch.float64)
    for i in range(0, tokens.shape[0], bs):
        ids = torch.from_numpy(tokens[i:i + bs]).long()
        if os.environ.get('CALREP_VERBOSE'):
            print(f'batch {i} t={time.time():.1f}', file=sys.stderr, flush=True)
        hs = hidden(ids, emb, blocks, final, arch["heads"], arch["kv"])
        hs, tgt = hs[:, :-1].reshape(-1, hs.shape[-1]), ids[:, 1:].reshape(-1)
        hh = torch.from_numpy(half[i:i + bs, 1:].reshape(-1).astype(np.int64))
        for j in range(0, hs.shape[0], 2048):
            z = hs[j:j + 2048] @ W.T                      # [n, V]
            y = tgt[j:j + 2048]; g = hh[j:j + 2048]
            zy = z.gather(1, y[:, None]).squeeze(1)
            nll = torch.logsumexp(z, 1) - zy
            raw.index_add_(0, g, nll.double()); cnt.index_add_(0, g, torch.ones_like(nll, dtype=torch.float64))
            for b_i, b in enumerate(betas):
                nb = torch.logsumexp(z * b, 1) - zy * b
                grid[:, b_i].index_add_(0, g, nb.double())
            if exact_T is not None:
                for hf in (0, 1):
                    m = g == hf
                    if m.any():
                        b = 1.0 / exact_T[hf]
                        exact[hf] += (torch.logsumexp(z[m] * b, 1) - zy[m] * b).double().sum()
    return raw.numpy(), grid.numpy(), cnt.numpy(), exact.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True); ap.add_argument("--dir", required=True)
    ap.add_argument("--step", required=True, type=int); ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--nseq", type=int, default=512)
    ap.add_argument("--logt-max", type=float, default=0.6)
    ap.add_argument("--seq-start", type=int, default=0); ap.add_argument("--seq-end", type=int, default=None)
    ap.add_argument("--bs", type=int, default=4); ap.add_argument("--exact-check", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    global LOGT
    LOGT = make_grid(a.logt_max)
    t0 = time.time()
    torch.set_num_threads(os.cpu_count() or 4)
    end = a.seq_end if a.seq_end is not None else a.nseq
    tokens = np.load(os.path.join(a.data, "tokens.npy"))[a.seq_start:end]
    half = np.load(os.path.join(a.data, "half.npy"))[a.seq_start:end]
    work = tempfile.mkdtemp(prefix="calrep_")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files, sb, ckrel = [], [], None
        # some lm1-misc runs nest the checkpoint one level deeper: <dir>/<dir>/global_step<N>
        cands = [f"{a.dir}/global_step{a.step}", f"{a.dir}/{a.dir}/global_step{a.step}"]
        for attempt in range(5):  # targeted listings; a whole-repo listing came back incomplete under load
            try:
                sb = []
                for sd in (a.dir, f"{a.dir}/{a.dir}"):  # 1b566b1b5 keeps its sbatch next to the nested checkpoint
                    sb = [f.path for f in api.list_repo_tree(a.repo, path_in_repo=sd, revision=REVISION.get(a.repo))
                          if re.fullmatch(rf"{re.escape(sd)}/sbatch_[^/]*\.sh", f.path)]
                    if sb:
                        break
                for c in cands:
                    try:
                        files = [f.path for f in api.list_repo_tree(a.repo, path_in_repo=c, revision=REVISION.get(a.repo))
                                 if f.path.split("/")[-1].startswith("layer_") and "model_states" in f.path]
                    except Exception as e:  # noqa: BLE001 - absent layout; try the next
                        print(f"listing {c} attempt {attempt}: {e!r}", file=sys.stderr, flush=True)
                        files = []
                    if files:
                        ckrel = c
                        break
            except Exception as e:  # noqa: BLE001 - retried, then fatal below
                print(f"listing attempt {attempt}: {e!r}", file=sys.stderr, flush=True)
            if files and sb:
                break
            time.sleep(30 * (attempt + 1))
        if not files or not sb:
            raise RuntimeError(f"no layer shards or sbatch found for {a.repo} under {cands}")
        for f in files:
            hf_get(a.repo, f, work)
        sbatch_text = open(hf_get(a.repo, sorted(sb)[0], work)).read()
        import urllib.request
        params_sh = urllib.request.urlopen(PARAMS_SH, timeout=60).read().decode()
        arch = arch_from_sbatch(sbatch_text, params_sh)
        t_dl = time.time() - t0
        print(f'downloaded in {t_dl:.1f}s arch={arch}', file=sys.stderr, flush=True)
        emb, blocks, final = load_shards(os.path.join(work, ckrel))
        if arch.get("inferred"):
            d = emb["word_embeddings.weight"].shape[1]
            kv = 32 if d < 512 else (64 if d < 1280 else 128)  # model_params.sh pattern; heads*kv from QKV rows; harness check validates
            arch.update({"d": d, "ffw": blocks[0]["mlp.dense_h_to_4h.weight"].shape[0], "kv": kv,
                         "heads": blocks[0]["self_attention.query_key_value.weight"].shape[0] // 3 // kv,
                         "layers": len(blocks)})
        if len(blocks) != arch["layers"] or emb is None or final is None:
            raise ValueError(f"shards {len(blocks)} blocks vs arch {arch}")
        n_params = sum(v.numel() for sd in [emb, final, *blocks] for v in sd.values())
        raw, grid, cnt, _ = score(tokens, half, emb, blocks, final, arch, a.bs)
        out = {"repo": a.repo, "dir": a.dir, "step": a.step, "arch": arch, "n_params_loaded": int(n_params),
               "seq_start": a.seq_start, "seq_end": end, "logT_grid": LOGT.tolist(),
               "sum_raw": raw.tolist(), "sum_grid": grid.tolist(), "count": cnt.tolist(),
               "seconds_download": round(t_dl, 1), "threads": torch.get_num_threads()}
        if a.exact_check:
            from finalize import fit_from_sums
            f = fit_from_sums(raw, grid, cnt, LOGT)
            _, _, _, ex = score(tokens, half, emb, blocks, final, arch, a.bs, exact_T={0: f["T_B"], 1: f["T_A"]})
            out["exact_check"] = {"T_A": f["T_A"], "T_B": f["T_B"],
                                  "spline_nll_B_at_T_A": f["nll_B_at_T_A"], "exact_nll_B_at_T_A": float(ex[1] / cnt[1]),
                                  "spline_nll_A_at_T_B": f["nll_A_at_T_B"], "exact_nll_A_at_T_B": float(ex[0] / cnt[0])}
        out["seconds_total"] = round(time.time() - t0, 1)
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print(json.dumps({"dir": a.dir, "seqs": [a.seq_start, end], "raw_nll": float(raw.sum() / cnt.sum()), "seconds_total": out["seconds_total"]}))
    finally:
        if not a.keep:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
