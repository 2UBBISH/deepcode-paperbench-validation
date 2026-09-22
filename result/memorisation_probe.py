"""Memorisation probe (09-22; usage: python3 memorisation_probe.py <authors-repo> <submission-tree> <paper.md>): how much of a generated tree looks like the authors' repo that the run was never shown.
Signals: (a) identifier overlap — function/class/argument names defined in the author repo that reappear in the tree
(excluding names that also appear in paper.md); (b) exact shared code lines ≥ 40 chars (normalised whitespace) excluding
imports/boilerplate; (c) file-name overlap; (d) literal constants shared and absent from the paper."""
import ast, re, sys, glob, os, collections
author, tree, paper = sys.argv[1], sys.argv[2], sys.argv[3]
ptxt = open(paper, encoding="utf-8", errors="replace").read().lower()
def py_files(root): return [f for f in glob.glob(f"{root}/**/*.py", recursive=True) if "/.git/" not in f]
def names(root):
    defs, args = set(), set()
    for f in py_files(root):
        try: t = ast.parse(open(f, encoding="utf-8", errors="replace").read())
        except Exception: continue
        for n in ast.walk(t):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): defs.add(n.name)
            if isinstance(n, ast.arg): args.add(n.arg)
    return defs, args
def lines(root):
    out = collections.Counter()
    for f in py_files(root):
        for l in open(f, encoding="utf-8", errors="replace"):
            s = re.sub(r"\s+", " ", l.strip())
            if len(s) >= 40 and not s.startswith(("import ", "from ", "#", '"""', "'''", "return ", "self.", "def __init__", "@", "print(")): out[s] += 1
    return out
ad, aa = names(author); td, ta = names(tree)
generic = {"forward", "__init__", "main", "train", "step", "loss", "evaluate", "fit", "sample", "update", "run", "reset", "get", "set", "x", "y", "self", "args", "kwargs", "device", "seed", "config", "model", "params", "lr", "epochs", "batch_size", "n", "i", "k", "key", "rng", "data", "path", "name", "shape", "dtype", "mean", "cov", "mu", "sigma", "grad", "loss_fn", "closure", "log", "verbose"}
shared_defs = sorted(d for d in ad & td if d not in generic and d.lower() not in ptxt)
shared_args = sorted(a for a in aa & ta if a not in generic and len(a) > 3 and a.lower() not in ptxt)
al, tl = lines(author), lines(tree); shared_lines = [l for l in al if l in tl]
afiles = {os.path.basename(f) for f in py_files(author)}; tfiles = {os.path.basename(f) for f in py_files(tree)}
shared_files = sorted((afiles & tfiles) - {"__init__.py", "main.py", "utils.py", "train.py", "models.py", "setup.py"})
consts = lambda root: set(re.findall(r"(?<![\w.])(\d+\.\d+e-?\d+|\d+e-?\d+|0\.\d{2,})(?![\w.])", " ".join(open(f, errors="replace").read() for f in py_files(root))))
shared_consts = sorted(c for c in consts(author) & consts(tree) if c not in ptxt and c not in {"0.5", "0.9", "0.99", "0.999", "1e-8", "1e-5", "1e-4", "1e-3", "1e-6", "0.01", "0.001", "0.95", "0.25", "0.75"})
print(f"author: {len(ad)} defs/{len(aa)} args/{len(al)} lines | tree: {len(td)} defs/{len(ta)} args/{len(tl)} lines")
print(f"shared def/class names NOT in paper ({len(shared_defs)}): {shared_defs[:25]}")
print(f"shared arg names NOT in paper ({len(shared_args)}): {shared_args[:25]}")
print(f"shared file basenames ({len(shared_files)}): {shared_files}")
print(f"shared code lines ≥40 chars ({len(shared_lines)}):"); [print("   ", l[:140]) for l in shared_lines[:12]]
print(f"shared numeric constants NOT in paper ({len(shared_consts)}): {shared_consts[:20]}")
