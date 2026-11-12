"""Run every reproducer under silentbug and check the result.

r*.py  real bugs: a BUG-level finding of the `# expect:` kind must be reported, and for
       files with a `# BUG` marker its line (or the user line that called it) must be that line.
f*.py  fixed versions / controls: no BUG-level finding may be reported.

Usage: python score.py [path/to/silentbug executable]
"""
import json, os, re, subprocess, sys, glob

HERE = os.path.dirname(os.path.abspath(__file__))
exe = sys.argv[1] if len(sys.argv) > 1 else "silentbug"


def run(path):
    p = subprocess.run([exe, "run", "--keep-going", "--json", "-", path], capture_output=True, text=True, cwd=HERE, timeout=300)
    try:
        return json.loads(p.stdout)
    except json.JSONDecodeError:
        return {"error": (p.stderr or p.stdout)[-400:], "findings": []}


def lines_of(f):
    out = set()
    for k in ("where", "caller"):
        if f.get(k):
            out.add((os.path.basename(f[k]["file"]), f[k]["line"]))
    return out


rows, ok_all = [], True
for path in sorted(glob.glob(os.path.join(HERE, "[rf][0-9][0-9]_*.py"))):
    name = os.path.basename(path)
    src = open(path).read().splitlines()
    rep = run(path)
    bugs = [f for f in rep["findings"] if f["level"] == "bug"]
    if name.startswith("r"):
        m = re.search(r"# expect: (\S+)(?: (\S+))?", "\n".join(src))
        kind, sub = m.group(1), m.group(2)
        marks = {(name, i + 1) for i, l in enumerate(src) if l.rstrip().endswith("# BUG")}
        hit = [f for f in bugs if f["kind"] == kind and (not sub or any(sub in p for p in f["params"]))]
        good = bool(hit) and (not marks or any(lines_of(f) & marks for f in hit))
        got = ", ".join(f"{f['kind']}:{f['title'][:60]}" for f in bugs) or "nothing"
    else:
        good = not bugs and "error" not in rep
        got = ", ".join(f"{f['kind']}:{f['title'][:60]}" for f in bugs) or ("ERROR " + rep["error"] if "error" in rep else "nothing")
    ok_all &= good
    rows.append((name, "yes" if good else "NO", got))

w = max(len(r[0]) for r in rows)
for n, g, got in rows:
    print(f"{n:<{w}}  {g:<3}  {got}")
r = [x for x in rows if x[0].startswith("r")]; f = [x for x in rows if x[0].startswith("f")]
print(f"\nbugs found at the right line: {sum(x[1] == 'yes' for x in r)}/{len(r)}   "
      f"controls clean: {sum(x[1] == 'yes' for x in f)}/{len(f)}")
sys.exit(0 if ok_all else 1)
