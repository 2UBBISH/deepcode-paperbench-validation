#!/usr/bin/env python3
"""Rebuild the per-paper score table in README.md from grades/*.grade.json (name: <paper>_<arm>_<runid>_… ; the 09-21
fre/rice files carry line3/line2/line0 labels and a .reparsed.json with the re-parsed score)."""
import json, re, glob, os
D = os.path.dirname(os.path.abspath(__file__)); G = os.path.join(D, "grades")  # result/grades
papers = "fre rice adaptive-pruning all-in-one bam bbox bridging-data-gaps ftrl mechanistic-understanding pinn lbcs lca-on-the-line sapg sequential-neural-score-estimation robust-clip sample-specific-masks stay-on-topic-with-classifier-free-guidance stochastic-interpolants test-time-model-adaptation what-will-my-model-forget".split()
cells = {}
for f in sorted(glob.glob(os.path.join(G, "*.grade.json"))):
    n = os.path.basename(f)
    m = re.match(r"(?P<paper>[a-z0-9-]+?)_(?P<arm>deepcode|codex)(?:-(?P<var>hydrated))?(?:_(?P<sub>(?:line\d|nofid)(?:_nofid)?))?_(?P<rid>[0-9a-f]{8})", n)
    if not m: continue
    d = json.load(open(f)); jo = d["paperbench_result"]["judge_output"]; score, inv, leaves = jo["score"], jo["num_invalid_leaf_nodes"], jo["num_leaf_nodes"]
    rp = f.replace(".grade.json", ".reparsed.json"); note = ""
    if os.path.exists(rp): r = json.load(open(rp)); score, inv = r["reparsed_score"], r["reparsed_invalid"]; note = "（补解析）"
    if "thinkon" in n: continue  # the 09-21 thinking-on fre grade stays out of the table
    arm = m["arm"] + ("_nofid" if m["sub"] and "nofid" in m["sub"] else "")
    key = m["paper"] + ("†" if m["var"] == "hydrated" else "")
    cells[(key, arm)] = f"**{score:.3f}**{note}" if inv <= 2 else f"{score:.3f}（{inv} 无效叶，作废）"
rows = ["| 论文 | deepcode（保真开） | codex | claude | deepcode 保真关 |", "| --- | --- | --- | --- | --- |"]
for p in [q for q in papers for q in ([q, q+"†"] if (q+"†","deepcode") in cells or (q+"†","codex") in cells else [q])]:
    rows.append(f"| {p} | {cells.get((p,'deepcode'),'—')} | {cells.get((p,'codex'),'—')} | — | {cells.get((p,'deepcode_nofid'),'')} |")
have = [p for p in papers if (p,'deepcode') in cells and (p,'codex') in cells]
footnote = "\n\n† = 补全版 `paper.md`（`materials/papers/robust-clip/hydration/`）：两臂都在补全后的输入上重跑（09-22）；未加 † 的 robust-clip 行是截断版输入，留作对照。"
table = "\n".join(rows) + f"\n\n有效对比（同一裁判口径下两臂都有分）：**{len(have)} 篇**" + ("：" + "、".join(have) if have else "") + "。" + footnote
s = open(os.path.join(D, "README.md"), encoding="utf-8").read()
start, end = "<!-- scores:start -->", "<!-- scores:end -->"
if start not in s:
    i = s.index("| 论文 | deepcode（保真开）"); j = s.index("\n\n", s.index("| adaptive-pruning", i))
    s = s[:i] + start + "\n" + table + "\n" + end + s[j:]
else:
    s = s[:s.index(start)] + start + "\n" + table + "\n" + s[s.index(end):]
open(os.path.join(D, "README.md"), "w", encoding="utf-8").write(s); print(f"README table rebuilt: {len(cells)} cells, {len(have)} full pairs")
