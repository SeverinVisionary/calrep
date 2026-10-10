"""Map the Step 0 accepted run set (definition c) to Hub checkpoints, N <= 2.81e9.
usage: python -I step1_runset.py step0/step0_results.json inventory/ out.tsv"""
import json, sys, re
res = json.load(open(sys.argv[1]))["c_cell1_le64_noseed_novariant"]
inv = {}
for fn, repo in [("datablations_lm1-misc.tsv", "datablations/lm1-misc"),
                 ("datablations_c4_isoflop_repos.tsv", None)]:
    cur = repo
    for line in open(sys.argv[2] + "/" + fn):
        f = line.rstrip("\n").split("\t")
        if f[0].startswith("TOTAL"):
            continue
        m = re.search(r"steps=\[([^\]]*)\]", line)
        steps = [int(s) for s in m.group(1).replace(" ", "").split(",") if s] if m else []
        gb = float(re.search(r"modelGB=([\d.]+)", line).group(1))
        inv.setdefault(f[0], []).append((cur, steps, gb))
# the isoflop tsv concatenates five repos; attribute by name
def isoflop_repo(name):
    if name.startswith("2b855b"): return "datablations/lm1-2b8-55b-c4-repetitions"
    if name.startswith("4b284b"): return "datablations/lm1-4b2-84b-c4-repetitions"
    if name.startswith("8b7178b"): return "datablations/lm1-8b7-178b-c4-repetitions"
N = dict(zip(res["runs"], [None]*len(res["runs"])))
step0 = {}
import ast
out = []
for name in sorted(res["runs"]):
    cands = [name, name + "c4", name.replace("20m", "21m", 1)]
    hit = None
    for c in cands:
        for repo, steps, gb in inv.get(c, []):
            repo = repo or isoflop_repo(c)
            if steps:
                hit = (repo, c, max(steps), gb)
                break
        if hit: break
    out.append((name, hit))
json_N = {}
nb_params = {"7m":7.098752e6,"14m":1.41e7,"20m":19.703712e6,"25m":35.5e6,"44m":4.4e7,"83m":8.27e7,"146m":1.465e8,"196m":201.236224e6,"220m":2.205e8,"221m":2.205e8,"280m":2.81e8,"421m":4.212e8,"574m":573.7e6,"619m":6.187e8,"1b1":1.0963e9,"1b5":1.5173e9,"2b2":2.160013824e9,"2b8":2.81e9,"3b8":3.581186048e9,"3b9":3.89971072e9,"4b2":4.2465e9,"8b7":8.67e9}
def params(n):
    i = 2
    while n[:i] not in nb_params: i += 1
    return nb_params[n[:i]]
with open(sys.argv[3], "w") as fo:
    fo.write("name\tN\tin_step1\trepo\tdir\tstep\tmodelGB\n")
    kept = miss = big = 0; gbs = 0
    for name, hit in out:
        n = params(name)
        if n > 2.81e9 + 1:
            big += 1; fo.write(f"{name}\t{n:.4g}\tno:N>2.8B\t\t\t\t\n"); continue
        if not hit:
            miss += 1; fo.write(f"{name}\t{n:.4g}\tno:no_checkpoint\t\t\t\t\n"); continue
        kept += 1; gbs += hit[3]
        fo.write(f"{name}\t{n:.4g}\tyes\t{hit[0]}\t{hit[1]}\t{hit[2]}\t{hit[3]}\n")
print(f"step0 set 158: N>2.8B {big}, no checkpoint {miss}, kept {kept}, model-state GB {gbs:.1f}")
