# robust-clip · paper.md 补全记录（2026-09-22）

官方 `paper.md`（现存为 `paper.md.official`，72,550 字符）在 §1 引言第 4 段句中截断（"…it is foreseeable that they"），
接着直接是 Table 1 和 `\subsection*{4.1…}`：§1 后半、§2 Related Work、§3（3.1–3.3，含 TeCoA / FARE 损失定义）、§4 开头、
§4.2 正文、§4.3 正文缺失（对应 PDF 第 1 页末至第 3 页、第 6–7 页正文）。附录 A–C 完整。

补全来源：arXiv 2402.12336v2 的 LaTeX 源（`arxiv.org/e-print/2402.12336`，2024-06-05 版 = PaperBench 用的 ICML 版；
`paper.pdf` 的 LFS 指针 sha256 与 HF 镜像 `josancamon/paperbench` 的 PDF 一致）。`tex2md.py` 把 `main_icml.tex` 的对应段落转成
PaperBench 的 md 方言（`\section*{N. Title}`、`$$` 公式、作者-年份引用、`\ref` → 表/图/节号）；三段产物 `part_*.md`，
在三处接缝插入：§1 截断句处（→ §1 余下 + §2 + §3 + §4 开头）、Table 3 之前（→ §4.2）、Table 4 之前（→ §4.3，去掉与原 md
重复的结尾句）。表格、图注、附录全部沿用官方 md，未改一字。补全后 92,105 字符。

口径影响：robust-clip 的输入字节变了，t19 的 deepcode 树（0.642）和 Codex 树（0.906）都是在截断版上生成的；两臂都要在补全版上重跑
才可比。其他 19 篇扫描过顶层章节编号连续，没有同类截断。
