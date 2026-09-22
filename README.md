# deepcode-paperbench-validation

PaperBench Code-Dev 上三臂对比（DeepEvol 的 DeepCode 线 · Codex 桌面版 · Claude 桌面版）的全部材料：裁判、输入、生成的代码树、分数。
四个并列目录：

| 目录 | 内容 |
| --- | --- |
| `paperbench/` | **判分**。`frontier-evals/` = openai/frontier-evals @ `patches/UPSTREAM_BASE.txt` 的 vendored 副本 + `patches/paperbench_local_changes.patch`（6 文件：整树裁判、json_object 解析器、并发、路径解析）；`scripts/run_grade.sh`（判池子里一篇论文的所有树）、`scripts/run_judge_eval.sh`（rice/0 裁判校准）、`scripts/run_trial.sh`（原装 DeepCode 基线臂）；`baseline-deepcode/` = HKUDS/DeepCode 上游 + `patches/deepcode_local_changes.patch`（口径补丁，默认等于上游）；`setup.sh`；`docs/`（INPUT_STANDARD 口径、CODEDEV-ARMS 三臂规则、PITFALLS、旧 README）；`runs/`（不入库）|
| `deepevol-deepcode/` | **我们线的生成结果**：`<paper>/submission/` 第 9 步的代码树（DeepEvol `0916onmain-experiment`，ADR 0004 原文保真），`<paper>/RUN_NOTES.md` 运行口径与过程数字。20 篇 + `fre-nofid`（保真关的对照） |
| `result/` | **分数与对照臂**：`grades/*.grade.json` 逐叶判分产物；`codex/<paper>/` Codex 桌面臂的树与审计；`README.md` 分数表（`update_readme.py` 重建）、裁判口径与 JudgeEval 校准；`RESULTS-HISTORY.md` 全部历史 |
| `materials/` | **评测原材料**：`papers/<paper>/` 三臂共用的同一份输入字节（`paper.md` + `addendum.md` + `blacklist.txt` + `rubric.json` + `config.yaml`；robust-clip 的 `paper.md` 是补全版，`hydration/` 记原版与补全过程）；`desktop-prompt/` 桌面臂的题面、附注、准备 / 收尾 / 审计脚本 |

## 口径（详见 `paperbench/docs/INPUT_STANDARD.md`、`result/README.md`）

- 三臂同一底座 `deepseek-flash` @ api.deepseek.com（官渠别名，09-22 起对应 **DeepSeek-V4.1-Flash**；Paratera 的 `DeepSeek-V4-Flash` 是旧的 V4），思考开，1M 上下文；同一份输入字节；不给 PDF、assets、rubric。
- 执行规则：命令允许，只有长时间 CPU/GPU 训练或评估不允许（题面告知，`audit_desktop.py` 事后审计 10 min / 60 min）；我们线的写码 agent 无执行工具，生成后只做 `compile()` 级语法检查。对比停在第 9 步的树。
- 裁判（09-21 起）：硅基流动 `deepseek-ai/DeepSeek-V4-Flash` 整树、思考关；解析器 `deepseek-ai/DeepSeek-V4-Pro`；`num_invalid_leaf_nodes ≤ 2` 才有效；JudgeEval rice/0 准确率 0.70–0.72（178/178 有效）。

## 跑

```bash
cd paperbench && bash setup.sh                       # 判分环境 + 基线臂环境 + 黑名单
# 判分：一次一棵树放进 ~/pb_submissions/<paper>/<arm>/，key 在 ~/Documents/env/siliconflow.env
PAPER=<paper> bash scripts/run_grade.sh              # 产物 runs/<paper>/grades/，拷到 ../result/grades/
python3 ../result/update_readme.py                   # 重建分数表
```

生成侧在 DeepEvol 仓库（`apps/v2/agent/paper2code/README.md` "生成到第 9 步并摆卷"）；Codex / Claude 桌面臂按 `materials/desktop-prompt/` 与分支 `0919-test` 的 README 跑。
