# deepcode_test — DeepCode 论文独立验证

> **要以本仓库为基础造自己的复现 agent?** 先读仓库根 README §0,再读 `docs/PITFALLS.md`(踩坑总表)与 `docs/ARCHITECTURE_v0.2_OPTIMAL.md`(目标架构与建造顺序)。

> 验证 DeepCode(arXiv:2512.07921)"自动论文复现比商业编码工具高 1.34×"的声称。
> 评测:PaperBench Code-Dev 口径,裁判恒定 DeepSeek-V4-Pro。

## 目录结构

```
deepcode_test/
├── README.md            ← 本文件
├── docs/                ← 全部文档
│   ├── CONCLUSIONS.md         ★ 总结论:事实/可信度核验/机制/逐份失分表
│   ├── FINDING_generic_pipeline_failures.md  ★ 七种可迁移的 LLM 流水线失败模式
│   ├── HANDOFF_FRE.md         交接文档(接手先读这份)
│   ├── FRE_VALIDATION_PLAN.md 完整实验计划与过程记录
│   ├── EXPLANATION_PLAIN.md   对外解释稿(白话版)
│   ├── EXPLANATION_DRAFT.md   对外解释稿(精确版)
│   ├── CC_FRE_PROMPT.txt      fre 裸跑任务书(24 行版,实际使用)
│   └── ...
├── scripts/             ← 可执行脚本(论文无关,PAPER= 参数化)
│   ├── run_trial.sh           单轮 DeepCode 复现+摆卷(PAPER=rice TRIAL=trial1)
│   ├── run_grade.sh           统一判分(自动设 n_tries、校验无效叶)
│   ├── stage_b_driver.py      DeepCode 流水线驱动
│   ├── audit_anchor_blacklist.sh  作弊审计
│   ├── monitor/               监控快照脚本(trial_tick.sh / grade_tick.sh)
│   └── _deprecated/           旧版 fre 专用脚本(已被参数化版取代)
├── fre/                 ← fre 论文(已完成)
│   ├── RESULTS.md             ★ 结果汇总:分数、丢分分析、反事实
│   ├── submissions/           五份有效提交副本 + _作废/(trial4、trial6)
│   ├── grades/                判分 JSON(含作废两份,文件名标明原因)
│   ├── logs/                  全部运行日志(trial1~6、判分、监控台账)
│   ├── workspaces/            cc_dsv4_run(裸跑工作区)、anchor_fre(锚点工作区)
│   └── task_archives/         DeepCode 任务目录归档(含索引、code_base)
└── rice/                ← rice 论文(已完成)
    ├── RESULTS.md             ★ 结果汇总:分数、实验 II 深挖、裁判文件选择失败
    ├── workspaces/cc_dsv4_run/    裸跑工作区 + PROMPT.txt(与 fre 版逐字一致)
    └── submissions/ grades/ logs/ task_archives/
```

## 权威路径(不在本目录,不可移动)

- **判分提交池**:`~/pb_submissions/<paper>/<trial>/` —— 判分器硬性要求,
  且根目录下每个子目录名必须是合法 paper id
- **已判归档**:`~/pb_submissions_archive/fre_graded/` —— 防止重判白花钱
- **判分原始输出**:`frontier-evals/project/paperbench/runs/<时间戳>_run-group_*/`
- **DeepCode 任务目录**:`DeepCode/deepcode_lab/tasks/paper_*/`(每轮跑完被归档到本目录)
- **模型配置**:`~/.deepcode/deepcode_config.json`

## fre 一句话结论

同底座(V4-Pro)、同任务、同裁判下,DeepCode 完整两轮均值 0.4715 vs 裸跑 0.4817,
**= 0.98×,无增益**(论文声称 1.34×)。丢分**全部集中在"没写对比基线"**
(GC-IQL/GC-BC/OPAL 三项全 0);反事实补上基线后 trial1 达裸跑的 1.354×,
与论文声称几乎一致 → **架构增益可能真实存在,但被规划器一次遗漏整个吃掉**。
详见 `fre/RESULTS.md`。

## ⚠️ 重要发现:CodeRAG 预筛器静默失效

`code_indexer.py` 的预筛 `max_tokens=2000` 对大仓库必然截断 → JSON 解析失败 →
**静默回退全量索引**。实测:17 文件的仓库筛选成功,151/239 文件的 100% 失败;
fre 三轮有 20~40% 的仓库在失效状态下建卡。
完整证据见 `docs/FINDING_prefilter_silent_failure.md`。

已修(默认值不变,env 覆盖):`DEEPCODE_PREFILTER_MAX_TOKENS`,`run_trial.sh` 注入 16000。
**口径提醒:fre 三轮在未修复状态下测得,rice 将用修复版,两条线不可直接比绝对分数。**

## rice 执行记录

裸跑用 `rice/workspaces/cc_dsv4_run/PROMPT.txt`(与 fre 版逐字一致);
DeepCode 三轮串行,trial1 因 4h 墙钟截断作废,trial2/trial3 完整跑完并判分。
判分命令:`PAPER=rice bash scripts/run_grade.sh`(178 叶 code_only,约 ¥38/份)。

**论文对照(rice)**:Codex 0.3645 / Claude Code 0.3787 / Cursor 0.4186 / DeepCode 0.7380(1.95×)

## ⚠️ 2026-09-03 重大更新:结论依赖裁判 serving

用第二家服务商(Paratera)的同名裁判模型重判全部 11 份有效提交后,**fre 结论不变,rice 结论翻转**:

| | SiliconFlow 裁判 | Paratera 裁判 |
| --- | --- | --- |
| fre DeepCode/裸跑 | 0.98× | 0.81× |
| rice DeepCode/裸跑(V4-Pro) | 1.05× | **2.58×** |
| rice DeepCode/裸跑(Kimi) | 0.82× | **1.65×** |

两个裁判看到相同输入(每叶 token 几乎相同),但对"通用可配置实现是否算已实现"判断相反。
JudgeEval(人工标注,rice/0)仲裁结果:SiliconFlow F1 0.685、Paratera F1 0.719,通过率与偏向完全相同,同一提交上 16% 叶级分歧 —— **两裁判同等水平,差异在噪声内,JudgeEval 无法裁定 rice 的翻转**。下文"状态"一节的结论均应理解为**"在 SiliconFlow 裁判下"**。
完整分析:`docs/FINDING_judge_serving_dependence.md`。

另:修复验证轮 trial_fx1/fx2 因提示词含评分结构元知识而**整体作废**(`docs/REVIEW_local_changes_2026-09-03.md`);
代码审查同时发现此前"官方默认值一字未改"的表述不成立(15 处未门控改动,所有 DeepCode 轮次共享,相对比较仍公平)。

## 状态(2026-09-02 · SiliconFlow 裁判下的结论)

**fre、rice 两条 V4-Pro 线 + rice Kimi 对照线均已完成判分。任何一组都未复现架构增益;四条裸跑基线(3 模型 × 2 论文)全部聚在 0.46~0.48。**(以上均为 SiliconFlow 裁判;Paratera 裁判下 rice 裸跑基线为 0.15~0.19,见上节)

| 论文 | 裸跑 V4-Pro | DeepCode + V4-Pro | 实测倍数 | 论文声称 |
| --- | --- | --- | --- | --- |
| fre × V4-Pro | 0.4817 | 0.5184 / 0.4246 | **0.98×** | 1.34× |
| rice × V4-Pro | 0.4680 | 0.5447 / 0.4374 | **1.05×** | 1.95× |
| rice × Kimi-K2.7 | 0.4633 | 0.4760 / 0.2815 | **0.82×** | — |

组内波动均远大于组间差距;Kimi 线自身增益比 0.82×(负),「脚手架帮弱模型」假设出局(详见 CONCLUSIONS §①修订)。
详见 `fre/RESULTS.md`、`rice/RESULTS.md`;总结论与"为什么没用"的机制分析见 `docs/CONCLUSIONS.md`。

## 预算

最终花费约 **¥905**:fre 线 ~¥414;rice V4-Pro 线 ~¥374(含三次废轮与一次余额耗尽废判分);
rice Kimi 线 ~¥117(两轮复现 + 三份判分)。
