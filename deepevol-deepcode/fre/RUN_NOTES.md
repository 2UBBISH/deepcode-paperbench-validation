# fre · deepcode（DeepEvol Paper2Code 线）

| | |
| --- | --- |
| 运行目录 | `~/Documents/search/paper2code-runs/fre-t17`（本机，不入库） |
| 线代码 | DeepEvol `0916onmain-experiment`，ADR 0004（蓝图 `Source:` 指针 → manifest → `read_paper` 读回 → 写前检查 → 审计只记录）；生成时义务为"每节至少一页"，`4b5cac6ab` 之后才改为整节每页 |
| 模型口径 | `deepseek-flash` @ api.deepseek.com，思考开，规划 / 实现单次 65536，上下文 1M，规划扇出开，figures off，`DEEPCODE_PAPER_FIDELITY=1`，静态语法检查开 |
| 输入 | PaperBench `fre` 的 paper.md + addendum + blacklist（与桌面臂同一份字节） |
| 过程 | 第一次实现在 32768 输出上限处截断（14/36），`rerun --phase implement` 后 37/37 文件、224 条读回回执、语法错 0；运行时审计 11 条"违规"全部是树拼写差异（后已修比对） |
| 摆卷 | 2026-09-21 01:38，`tree: stage9`（第 9 步，不租机、不执行、不修复），池 `~/pb_submissions/fre/line3` |
| 判分 | `~/Documents/0919-test/grades/fre_deepcode_9871437d_sf_tree_thinkon.grade.json`：**0.9210**，306 叶 0 无效，2026-09-21 12:24；硅基 `deepseek-ai/DeepSeek-V4-Flash` 整树、**思考开**、Flash 解析器（比 09-21 16:40 定的默认口径多"思考开"一项，待按默认口径重判一次） |
