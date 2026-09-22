# fre · deepcode，**保真关**（对照 / "蓝图优化之前"）

| | |
| --- | --- |
| 运行目录 | `~/Documents/search/paper2code-runs/fre-t18-nofid`（`CONTROL.txt`） |
| 唯一变量 | `DEEPCODE_PAPER_FIDELITY=0`：planner 提示词没有 Source 指针段（蓝图 0 指针、无 manifest），写码 agent 没有 `read_paper`、没有写前检查、没有审计；其余与 `../fre` 完全相同（引擎 `00679da1f`，官渠 `deepseek-flash` 思考开，1M，扇出开，65536/65536，figures off，语法检查开） |
| 过程 | 17:20 → 18:43（index 52 min、implement 25 min）；40/39 文件、`paper_reads` 0、语法错 2 → 修复后 0；四道闸过 |
| 树 | 36 py / 874k 字符（`../fre` 31 py / 804k） |
| 摆卷 | `tree: stage9`，池 `~/pb_submissions/fre/line0` |
| 判分 | **0.8624**，306 叶 0 无效（09-21 20:12，默认口径）；子树 数据/环境 0.917 · 方法 0.991 · 训练评估 **0.680**（保真开的 t17：0.833 / 0.991 / 0.939 思考开口径）。`grades/fre_deepcode_line0_nofid_79e85f52_…` |
