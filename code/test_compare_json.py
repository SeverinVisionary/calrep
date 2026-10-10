"""Negative controls for compare_json.py: each case must give the expected exit code (run by reproduce.sh).
usage: python3 -I test_compare_json.py"""
import json, os, subprocess, sys, tempfile

CMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "compare_json.py")
CASES = [  # (a, b, expected exit code)
    ('{"p": true}', '{"p": 1}', 1), ('{"p": false}', '{"p": 0.0}', 1), ('{"p": Infinity}', '{"p": Infinity}', 1),
    ('{"p": NaN}', '{"p": NaN}', 1), ('{"p": 1.0}', '{"p": 1.01}', 1), ('{"p": [1]}', '{"p": [1, 2]}', 1),
    ('{"p": 1}', '{"q": 1}', 1), ('{"p": "x"}', '{"p": "y"}', 1),
    ('{"p": 1.0}', '{"p": 1.0000000001}', 0), ('{"p": true, "q": [1, "x"]}', '{"p": true, "q": [1, "x"]}', 0),
]
bad = 0
with tempfile.TemporaryDirectory() as d:
    a, b = os.path.join(d, "a.json"), os.path.join(d, "b.json")
    for x, y, want in CASES:
        open(a, "w").write(x); open(b, "w").write(y)
        got = subprocess.run([sys.executable, "-I", CMP, a, b], capture_output=True).returncode
        if got != want:
            bad += 1; print(f"FAIL {x} vs {y}: exit {got}, want {want}")
print(f"compare_json self-test: {len(CASES) - bad}/{len(CASES)} cases as expected")
sys.exit(1 if bad else 0)
