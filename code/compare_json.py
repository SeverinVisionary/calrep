"""Compare two JSON result files: relative tolerance on finite floats, exact elsewhere.
Booleans never match numbers, and NaN or infinity never matches anything.
Keys listed with --ignore are skipped at any depth (e.g. timing fields).
usage: python -I compare_json.py reproduced.json shipped.json [--rtol 1e-6] [--ignore key ...]"""
import argparse, json, math, sys

ap = argparse.ArgumentParser()
ap.add_argument("a"); ap.add_argument("b")
ap.add_argument("--rtol", type=float, default=1e-6); ap.add_argument("--ignore", nargs="*", default=[])
args = ap.parse_args()
bad = []


def walk(x, y, path):
    if isinstance(x, dict) and isinstance(y, dict):
        for k in sorted(set(x) | set(y)):
            if k in args.ignore:
                continue
            if k not in x or k not in y:
                bad.append(f"{path}/{k}: present in one file only")
            else:
                walk(x[k], y[k], f"{path}/{k}")
    elif isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            bad.append(f"{path}: length {len(x)} vs {len(y)}")
        for i, (u, v) in enumerate(zip(x, y)):
            walk(u, v, f"{path}[{i}]")
    elif isinstance(x, bool) or isinstance(y, bool):
        if not (isinstance(x, bool) and isinstance(y, bool)) or x != y:
            bad.append(f"{path}: {x!r} vs {y!r}")
    elif isinstance(x, float) or isinstance(y, float):
        if any(isinstance(v, float) and not math.isfinite(v) for v in (x, y)):
            bad.append(f"{path}: non-finite value {x!r} vs {y!r}")
            return
        if not (isinstance(x, (int, float)) and isinstance(y, (int, float))) or \
                not math.isclose(x, y, rel_tol=args.rtol, abs_tol=1e-12):
            bad.append(f"{path}: {x} vs {y}")
    elif x != y:
        bad.append(f"{path}: {x!r} vs {y!r}")


walk(json.load(open(args.a)), json.load(open(args.b)), "")
print(f"{args.a} vs {args.b}: " + ("match" if not bad else f"{len(bad)} differences"))
for line in bad[:20]:
    print("  " + line)
sys.exit(1 if bad else 0)
