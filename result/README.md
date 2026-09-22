# result/ · Code-Dev 三臂对比的分数与产物（2026-09-21 起）

owner 09-21：从这一天起只保留并对比这里的提交；以后每一份新提交（任何论文、任何臂）都放到这里，旧池 / 旧归档不再引用。

## 布局

```
result/grades/<paper>_<arm>_<runid>_sf_tree_thinkoff_pro.grade.json   ← PaperBench 判分产物（逐叶分数与裁判理由）
result/codex/<paper>/codex[-hydrated]/   submission/ AUDIT.txt RUN_NOTES.md PROMPT.txt START_EPOCH   ← Codex 桌面臂（无会话日志、无下载的数据集）
result/RESULTS-HISTORY.md                                             ← 所有分数的历史与作废记录
result/update_readme.py                                               ← 从 grades/ 重建下面的分数表
result/memorisation_probe.py                                          ← 生成树 vs 作者仓库的抄袭探针
```

deepcode 臂的树在 `../deepevol-deepcode/<paper>/submission/`（同目录 `RUN_NOTES.md` 记运行口径），Claude 臂尚无。
本机原始存档（含会话日志、数据集）在 owner 的 `~/Documents/0919-test/`。

## 裁判口径（09-21 16:40 定，之后每一份都按这个判）

硅基流动 `deepseek-ai/DeepSeek-V4-Flash` 做裁判：**整棵代码树**进每叶提示词（`PB_JUDGE_WHOLE_CODEBASE=1`，前缀缓存命中 ~98%）、**思考关**；结构化解析器 `deepseek-ai/DeepSeek-V4-Pro`（json_object + "回实例不回 schema"引导语）。`run_grade.sh` / `run_judge_eval.sh` 的默认。

JudgeEval 校准（rice/0，178 人工标注 Code-Dev 叶）：

| 口径 | acc | prec / rec / F1 | 有效叶 |
| --- | --- | --- | --- |
| 09-17 上游行为：每叶选 10 文件、Paratera、Pro 解析器 | 0.719 | — | 178 |
| 09-21 整树、裁判思考关、Flash 解析器 | 0.722 | 0.744 / 0.703 / 0.702 | 169（9 无效：Flash 解析器随机把 `Score: 0` 判无效） |
| **09-21 整树、裁判思考关、Pro 解析器**（同一批裁判文本离线重解析） | **0.702–0.720** | 0.719 / 0.688 / 0.686 | **178 / 0 无效** |

结论：整树 + 思考关不掉准确率（三种口径都在 0.70–0.72），解析器必须用 Pro 才能 0 无效。一份 306 叶的 fre 约 ¥32（02:00–08:00 半价 ¥16）+ 解析器几块钱。

## 汇总（按论文 × 臂；分数 = 上面的裁判口径）

<!-- scores:start -->
| 论文 | deepcode（保真开） | codex | claude | deepcode 保真关 |
| --- | --- | --- | --- | --- |
| fre | **0.961**（补解析） | **0.914**（补解析） | — | **0.862** |
| rice | **0.978**（补解析） | **0.976**（补解析） | — |  |
| adaptive-pruning | — | — | — |  |
| all-in-one | — | — | — |  |
| bam | — | **1.000** | — |  |
| bbox | — | — | — |  |
| bridging-data-gaps | — | — | — |  |
| ftrl | — | **0.602** | — |  |
| mechanistic-understanding | — | — | — |  |
| pinn | **1.000** | **1.000** | — |  |
| lbcs | **0.987** | **0.993** | — |  |
| lca-on-the-line | **0.925** | **0.898** | — |  |
| sapg | — | — | — |  |
| sequential-neural-score-estimation | — | — | — |  |
| robust-clip | **0.642** | **0.906** | — |  |
| robust-clip† | **1.000** | **0.941** | — |  |
| sample-specific-masks | — | — | — |  |
| stay-on-topic-with-classifier-free-guidance | — | — | — |  |
| stochastic-interpolants | — | — | — |  |
| test-time-model-adaptation | — | — | — |  |
| what-will-my-model-forget | **0.990** | **0.988** | — |  |

有效对比（同一裁判口径下两臂都有分）：**7 篇**：fre、rice、pinn、lbcs、lca-on-the-line、robust-clip、what-will-my-model-forget。

† = 补全版 `paper.md`（`materials/papers/robust-clip/hydration/`）：两臂都在补全后的输入上重跑（09-22）；未加 † 的 robust-clip 行是截断版输入，留作对照。
<!-- scores:end -->

"判出"是 `run_grade.sh` 落盘的分（无效叶按 0 计，>2 个即作废）；"补解析"= 同一批裁判文本里无效叶用同一个 Pro 解析器离线再解析（09-21 全部第一次重试即过，都是解析器瞬时抽风，`grades/*.reparsed.json`）。09-21 19:20 起 `simple.py` 解析失败自动重试 3 次（`PB_PARSER_ATTEMPTS`），之后的判分不再需要补解析。fre 早前思考开那份 0.9210（`fre_deepcode_9871437d_sf_tree_thinkon`）留档不入表。

Codex 臂（owner 09-20 起跑，`codex/results/`，10 篇全部 `CALIBER_REVIEW`：单条命令最长 30 s，总计最多 28 min，没有一篇越过 10 min / 60 min 线；每篇 RUN_NOTES 都有"blacklist mentioned in README"提示 —— 是引用还是抄代码，owner 过一眼）：

| 论文 | 起跑 | 结束（墙钟） | 跑过的命令 / 总时长 | py 文件 |
| --- | --- | --- | --- | --- |
| fre | 09-20 16:40 | 188 min | 14 / <1 min | 30 |
| rice | 09-20 16:40 | 193 min | 105 / 28 min | 53 |
| adaptive-pruning | 09-20 19:52 | 852 min | 93 / 20 min | 33 |
| all-in-one | 09-20 19:52 | 908 min | 45 / 6 min | 30 |
| bam | 09-20 19:51 | 853 min | 158 / 15 min | 30 |
| bbox | 09-20 19:52 | 908 min | 48 / 6 min | 67 |
| bridging-data-gaps | 09-20 19:52 | 1093 min | 91 / 9 min | 39 |
| ftrl | 09-20 19:52 | 1093 min | 60 / 11 min | 50 |
| mechanistic-understanding | 09-20 19:52 | 853 min | 48 / 6 min | 41 |
| pinn | 09-20 19:51 | 853 min | 73 / 10 min | 25 |

（09-20 晚 8 篇是并行起的，墙钟含等待；`codex/work/` 里还有 10 个工作目录没有 results —— lbcs、lca-on-the-line、robust-clip、sample-specific-masks、sapg、sequential-neural-score-estimation、stay-on-topic…、stochastic-interpolants、test-time-model-adaptation、what-will-my-model-forget。）

**fre 消融**（同一天同口径，唯一变量 `DEEPCODE_PAPER_FIDELITY`）：保真开 0.961 / 保真关 0.862 / Codex 0.914 —— 蓝图 `Source:` 指针 + 整节读回 + 写前检查这一套值 **+0.10**；关掉之后线比 Codex 还低 0.05。一篇一份，噪声 0.025。

**robust-clip 0.642 vs 0.906 的原因（09-22 查）**：两个因素。① 输入：官方 `paper.md`（PDF 是 LFS 指针，md 是唯一输入）在 §1 引言第 4 段句中截断（"it is foreseeable that they"），下一行就是 Table 1 和 `\subsection*{4.1…}`——§1 后半、§2 相关工作、**§3 方法（TeCoA/FARE 损失、冻结文本编码器、训练设置）**、§4 开头整体缺失；§4.1、§4.4、附录 B.1–B.10（B.6 无目标攻击细节、B.8 定向攻击细节）都在。② planner：14 条 `Source:` 全部写 `§Addendum`，一条都没指向存在的 §4.1 / B.6 / B.8；`train_robust_clip.py` 当 glue。deepcode 独丢 16 叶：TeCoA 交叉熵损失、文本编码器冻结属于缺失的 §3（无解）；VQAv2 数据集（§4.1 有）、定向攻击流程（Table 3 / B.8 有）、APGD 10000 步（B.8/B.9 有）是 planner 没指、写码没读。Codex 没抓论文（审计只有 auto-attack / cider 的 curl），靠记忆补的。对比表里这篇单独标注。

有效对比的定义：同一论文三臂都在上面的裁判口径下有分。目前 0 组。前史里的数字（bam 09-15 DeepCode 0.837 vs Codex 0.734 单次样本；fre 09-20 白天 Codex 带蓝图且跑了实验 0.785）都不作先验。

## 待办

1. 判分（v4 在 `validation/runs/grade_night_v4.sh`，只在 02:00–07:10 跑，pid 见 `runs/grade_night_v4.pid`；每晚自动续）：第一夜（09-22）收 13 棵，7 篇成对；剩下约 23 棵（13 棵 deepcode + 10 棵 codex）第二夜 09-23 02:00 起。分数表由 `update_readme.py` 重建。
2. owner：Claude 桌面臂；Codex 还缺 sequential-neural-score-estimation 等篇（`codex/results/` 里没有的会被跳过并记 "no tree yet"）。
3. 线：t19 的 18 棵树已同步到 `deepcode/<paper>/`（含 RUN_NOTES）；robust-clip 0.642 vs Codex 0.906 的丢分项值得看（官方 paper.md 缺方法章）。
