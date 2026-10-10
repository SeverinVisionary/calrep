"""Run CALREP work units (run, sequence range) and checkpoint each to git.
units file: TSV with columns name, seq_start, seq_end [, exact, logt_max, data]; run metadata from step0/step1_runset.tsv.
Each unit writes results/parts/<name>__<start>-<end>.json; with --push BRANCH the part is committed
and pushed to that branch right after it finishes. Units whose part already exists are skipped.
usage: python -I run_units.py units/smoke.tsv [--push research/calrep-shard-smoke]"""
import argparse, csv, json, os, subprocess, sys, time
here = os.path.dirname(os.path.abspath(__file__)); root = os.path.dirname(here)
ap = argparse.ArgumentParser(); ap.add_argument("units"); ap.add_argument("--push"); a = ap.parse_args()
meta = {r["name"]: r for r in csv.DictReader(open(os.path.join(root, "step0/step1_runset.tsv")), delimiter="\t")}
failed = []
# the evaluator commit, read once: result commits made with --push advance HEAD during the loop
code_sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
os.makedirs(os.path.join(root, "results"), exist_ok=True)
log = open(os.path.join(root, "results", "unitlog_" + os.path.basename(a.units).replace(".tsv", ".txt")), "a")
for u in csv.DictReader(open(a.units), delimiter="\t"):
    r = meta[u["name"]]; s, e = int(u["seq_start"]), int(u["seq_end"])
    data = u.get("data") or "eval_data"
    parts = os.path.join(root, "results", "parts" if data == "eval_data" else "parts_" + data)
    os.makedirs(parts, exist_ok=True)
    gm = u.get("logt_max") or "0.6"
    tag = "" if gm == "0.6" else f"__w{gm}"
    out = os.path.join(parts, f'{u["name"]}__{s:03d}-{e:03d}{tag}.json')
    if os.path.exists(out):
        print("skip", out, flush=True); continue
    cmd = [sys.executable, "-I", os.path.join(here, "calrep_eval.py"), "--repo", r["repo"], "--dir", r["dir"],
           "--step", r["step"], "--data", os.path.join(root, data), "--out", out,
           "--seq-start", str(s), "--seq-end", str(e), "--bs", "1"]
    cmd += ["--logt-max", gm]
    if u.get("exact", "") == "1":
        cmd.append("--exact-check")
    t = time.time(); p = subprocess.run(cmd); dt = time.time() - t
    msg = f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {u["name"]} {s}-{e} rc={p.returncode} {dt:.0f}s'
    print(msg, flush=True); log.write(msg + "\n"); log.flush()
    if p.returncode != 0:
        failed.append(f'{u["name"]} {s}-{e}')
    if p.returncode == 0:  # provenance: the evaluator commit this part was computed with
        j = json.load(open(out)); j["code_sha"] = code_sha; json.dump(j, open(out, "w"))
    if a.push and p.returncode == 0:
        subprocess.run(["git", "-C", root, "add", out, log.name], check=True)
        subprocess.run(["git", "-C", root, "commit", "-q", "--no-verify", "-m", f"calrep part {u['name']} {s}-{e} (cloud)"], check=True)
        for _ in range(5):
            if subprocess.run(["git", "-C", root, "push", "-q", "origin", f"HEAD:refs/heads/{a.push}"]).returncode == 0:
                break
            time.sleep(20)
        else:
            failed.append(f'{u["name"]} {s}-{e} (push)')

if failed:
    print("FAILED UNITS:", "; ".join(failed), flush=True)
    sys.exit(1)
