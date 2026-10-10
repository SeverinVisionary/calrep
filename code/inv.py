import sys, json, re, collections
from huggingface_hub import HfApi
api = HfApi()
repo = sys.argv[1]; out = sys.argv[2]
files = []
for f in api.list_repo_tree(repo, recursive=True):
    files.append({"path": f.path, "size": getattr(f, "size", None)})
json.dump(files, open(out, "w"))
runs = collections.defaultdict(lambda: {"model": 0, "optim": 0, "steps": set(), "tb_val": False, "evals": 0})
for f in files:
    p = f["path"].split("/")
    if len(p) < 2: continue
    r = runs[p[0]]; s = f["size"] or 0
    for q in p:
        m = re.match(r"global_step(\d+)", q)
        if m: r["steps"].add(int(m.group(1)))
    if "model_states" in f["path"] or "model_optim_rng" in f["path"]: r["model"] += s
    elif "optim_states" in f["path"]: r["optim"] += s
    if "val" in f["path"] and "tensorboard" in f["path"]: r["tb_val"] = True
    if "eval" in f["path"].lower() and f["path"].endswith(".json"): r["evals"] += 1
tot = 0
for k in sorted(runs):
    r = runs[k]; tot += r["model"]
    print(f'{k}\tsteps={sorted(r["steps"])}\tmodelGB={r["model"]/1e9:.2f}\toptimGB={r["optim"]/1e9:.2f}\ttbval={r["tb_val"]}\tevaljson={r["evals"]}')
print("TOTAL model-state GB", round(tot/1e9,1), "runs", len(runs))
