# sequential-neural-score-estimation 结果汇总

> 2026-09-14 · 三份有效判分完成（无效叶 0/67）
> 判分口径：PaperBench Code-Dev（`code_only=True`，**不执行代码**）、67 叶、
> 裁判恒定 DeepSeek-V4-Pro @ Paratera（`judge_type=simple`）；判分日志 `logs/grade_ours.out`、`logs/grade_trial2.out`
> 论文：Sharrock et al., *Sequential Neural Score Estimation*（ICML 2024，PaperBench `sequential-neural-score-estimation`，CPU、无外部数据集）

---

## 1. 总分

| 提交 | 配置 | 得分 | 规模 |
| --- | --- | --- | --- |
| **deepevol_s10** | DeepEvol 复现线，Stage 10 结束时的导出（写码 + 真环境 + 冒烟修复轮后） | **0.7729** | 25 py / 3,854 行 |
| **deepcode_trial2** | DeepCode + V4-Pro（思考关，索引 max_tokens 16000） | **0.7280** | 35 py / 11,431 行 |
| **deepevol_s9** | DeepEvol 复现线，Stage 9 结束时的导出（只写码 + judge，未经修复轮） | **0.6854** | 25 py / 3,817 行 |

写码模型三份相同：DeepSeek-V4-Pro @ Paratera，思考关。

**论文里的分数**（对照口径）：PaperBench 论文 o1 Code-Dev 全集 43.4%；这篇论文在 PaperBench 全复现榜上 o1 IterativeAgent 均值 0.466。DeepCode 论文自报全集 73.5%，没有单独给这篇。

## 2. 主结论

- **同一写码模型下，我们 Stage 10 版 0.7729 > DeepCode 0.7280 > 我们 Stage 9 版 0.6854**。修复轮（Stage 10 冒烟格 + fixture + 两轮修复）把我们从 0.6854 拉到 0.7729（+0.0875），拉开的正是和 DeepCode 的差距。
- DeepCode 写了三倍的代码（11.4k 行 vs 3.8k），叶得分却没有更高：50 个满分叶 vs 我们 51 / 48。
- 一份对一份，不是分布对分布：rice 那次两轮 DeepCode 差了 0.107，这里单轮结论只能说"同量级、我们略高"，不能说"显著"。

## 3. 维度分解

| 一级维度（权重） | deepevol_s10 | deepevol_s9 | deepcode_trial2 |
| --- | --- | --- | --- |
| 任务集（App. E.1）(10) | **0.889** | 0.778 | 0.778 |
| VESDE / VPSDE (10) | **0.944** | 0.833 | 0.833 |
| 基线 NPE / SNPE / TSNPE (10) | **0.667** | 0.500 | **0.000** |
| C2ST (1) | 0.000 | 0.000 | 0.000 |
| NPSE (20) | **1.000** | 0.750 | **1.000** |
| TSNPSE (20) | 0.660 | 0.706 | **0.752** |
| 第 5 节结果复现 (20) | 0.607 | 0.607 | **0.755** |
| **总分** | **0.7729** | **0.6854** | **0.7280** |

**我们赢在基线**：NPE / SNPE 都写了并有训练定义（Stage 3 的冻结判据把每个基线列为必须实现的方法，Stage 7 蓝图规则 (q) 要求每个方法一个类），DeepCode 三个基线全 0——它索引了 `sbi` 仓库（NPE/SNPE 就在里面）却一个没写。
**DeepCode 赢在结果复现与 TSNPSE**：`run_snpse_variants.py` / `evaluate_benchmarks.py` / `make_figures.py` 这类实验脚本它铺得更全（第 5 节 0.755 vs 0.607）；TSNPSE 的 proposal prior 那一叶它 0.93、我们 0.53/0.72。
两边共同的零分：C2ST（都没用 `sbibm` 的默认实现）、TSNPE（GitHub 参考实现，两边都没照搬）、TSNPSE 第 r 轮数据集构造（|M| 个样本那一叶）。

## 4. 过程事实（影响对比口径的）

- **DeepCode trial1 作废**：思考开、索引器 `max_tokens` 8000/4000（仓库默认）→ 51 个参考文件里 20 个分析结果被截断、索引丢失；中止。
- **DeepCode 第一次 trial2 作废**：把索引器两处 `max_tokens` env 化（`DEEPCODE_ANALYSIS_MAX_TOKENS` / `DEEPCODE_RELATIONSHIP_MAX_TOKENS`=16000）后重跑，但思考仍开着——和我们（思考关）不是同一口径；跑到索引 8/51 时中止。
- **有效 trial2**：`DEEPCODE_THINKING=off`（`DeepCode/core/providers/openai_compat.py` 与 `tools/code_indexer.py` 两处 `extra_body.enable_thinking=False`）+ 索引 16000。14:08 起、18:34 完成（4 h 26 min，含索引 5 个仓库 108 个文件）；写码循环里 Paratera 空响应重试 8 次、`finish_reason='length'` 1 次（工具调用被丢），是它慢的主因，不影响产物完整性（流水线状态 completed）。
- **我们**：Stage 1–10 共 68 次模型调用、约 1,015 万 tokens ≈ ¥34（含 Stage 7 五次重问、三次 Stage 9），跑在租的阿里云 `ecs.c7.2xlarge`（真 SetupX 搭环境、冒烟格、fixture、修复轮）；导出用 `scripts/reproduction_export_submission.py --stage 9|10`。
- **参考仓库不对称**：我们 Stage 6 用文献层（论文引用 + GitHub 搜索）拿到 `gpapamak/epsilon_free_inference`、`hojonathanho/diffusion`、`mackelab/tsnpe_neurips`、`sbi-benchmark/sbibm`，加上 PaperBench 提供的两份上传（`tsnpe_implementation`、`pyloric_simulator`）；DeepCode 自己搜到的是 `flow-matching-posterior-estimation`、`score_sde_pytorch`、`sbi`、`CSDI`、`sbibm`。两边都拿到了 sbibm；只有我们拿到了 TSNPE 的参考实现（但 TSNPE 那一叶我们照样 0），只有 DeepCode 拿到了 sbi。
- **没起裸跑第三臂**（owner 定）；rice/fre 的裸跑对照见各自 RESULTS.md。
- 判分成本：每份约 ¥38（V4-Pro）。

## 5. 文件

- `grades/deepevol_s10.grade.json`、`grades/deepevol_s9.grade.json`、`grades/deepcode_trial2.grade.json`：判分树（每叶得分与裁判解释）。
- `submissions/deepevol_s10`、`submissions/deepevol_s9`、`submissions/trial2`：三份提交原样。
- `logs/`：DeepCode 两次作废与有效 trial2 的完整日志、两次判分日志。
- `task_archives/`（不入库，913 MB）：DeepCode 的 task 目录归档，含索引产物。
