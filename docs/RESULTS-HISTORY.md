# 历史结果总表（2026-08-25 → 2026-09-15）

所有数字一处查全；原始 `grade.json`、提交产物、日志与旧分析文档已从仓库移到本地
`~/Documents/env/paperbench-judge/archive/`（不再入库，见文末 §9）。**每张表都标了口径与是否作废**，
引用前先看 §0。分数一律是 PaperBench Code-Dev（`code_only=True`，不执行代码），裁判模型名 + serving 随分数一起给。

## 0. 作废规则与两次全体作废

作废 = 不再作为对比结论引用，只作可追溯记录。

| 作废原因 | 波及 | 说明 |
| --- | --- | --- |
| **输入有偏**（2026-09-14 定） | 09-14 之前全部对比 | 裸跑提示词在官方指令外多给了六类偏帮；DeepCode 臂没拿到 `addendum.md`；"思考关"从未成立（Paratera 忽略 `enable_thinking:false`，只认 `thinking:{type:disabled}`）；裸跑壳是 Claude Code 而非论文里差距最大的 Codex |
| **裁判选文件根目录 bug**（2026-09-15 修） | 09-15 修前全部分数 | 裁判让模型列"最相关文件"，约五分之一回答省掉树根 `submission/`，上游原样拼路径一个读不到，整份摆卷逐叶被判"没实现"。修后 bam 三份分别 0.7644→0.9073、0.6659→0.8367、0.6530→0.7343 |
| 单轮作废 | 见 §6 | 撞墙钟 / stall 熔断 / 假计划 / 余额耗尽 / 提示词含评分元知识 |

仍然成立、与输入无关的：§7 JudgeEval 校准、§5 裁判 serving 依赖、§8 四处静默降级。

## 1. bam 三方对照（2026-09-15，当前唯一有效的对比数字）

口径：`docs/INPUT_STANDARD.md`（材料五样字节级相同、裸跑用官方指令原文 + 冻结后缀、**思考开**、裁判恒定
DeepSeek-V4-Pro @ Paratera、`PB_JUDGE_CONCURRENCY=20`、裁判已修选文件根目录 bug）。

| 臂 | 系统 | 状态 | 总分 |
| --- | --- | --- | --- |
| 03_bare.gpt5-codex-high | Codex 桌面版 + gpt-5.5 high（底座不同，不进主表） | 完成，12.5 min | **0.9073** |
| 02_deepcode | DeepCode（e0767d0 + 旧补丁，`PAPER=bam TRIAL=trial1`）+ V4-Pro | 完成，3 h 13 min | **0.8367** |
| 03_bare | Codex 桌面版 + V4-Pro | 完成，26 min | **0.7343** |
| 01_deepevol | DeepEvol 复现线（旧架构）+ V4-Pro | 停跑，无分：Stage 9 五个 attempt 未过，owner 09-15 叫停 | — |

同底座（V4-Pro）：DeepCode 0.8367 > 裸跑 Codex 0.7343，差 0.10，来自 §5.2/§5.3 两个实验节；核心算法节 DeepCode 反而低（0.721 vs 0.821）。
三臂全丢的 38 片叶子几乎都是"ADVI 学习率网格搜索"。修裁判前的三份分（0.7644 / 0.6659 / 0.6530）与并发 16 对照（0.6317）作废。
论文 Table 1 bam 列（Sonnet 4.5-think，o3-mini 裁判）：Codex 0.1937 / Claude Code 0.3829 / Cursor 0.3779 / DeepCode 0.8530——只看相对关系：Codex 0.19 的差距在同底座同裁判下不复现。

叶子级明细、各臂 RUN_NOTES 与原始 grade.json：`archive/deepcode_test/bam/` 与 owner 本机 `~/Documents/env/bam-threeway/`。

**2026-09-17 起的新一批**：DeepEvol 的 Paper2Code 线（原装 DeepCode 引擎嵌入）vs 本仓库的基线运行，口径改为 **DeepSeek-V4-Flash、思考关**（`reasoning_tokens == 0`），第一篇 `sapg`。两边并排的过程数字在 DeepEvol 仓库 `apps/v2/agent/paper2code/HANDOFF.md`「Second batch」，判分后再回填到这里。

| 2026-09-17 sapg | 基线运行 `trial1`（本仓库，DeepCode 21ebc57f + 补丁） | DeepEvol 线 C9 运行 |
| --- | --- | --- |
| 状态 | 完成，未判分；摆卷 `~/pb_submissions/sapg/trial1/`（30 文件） | 完成，未判分 |
| 计划 / 参考 / 克隆 / 索引 | 9,733 字符 `generated` 分段 · 5 URL · 5 仓库 · 5 索引（2,103 s） | 9,672 字符 `generated` 分段 · 4 URL · 4 仓库 · 4 索引（1,711 s） |
| 写码 | 29/29 文件，518 s，24 py / 7,319 行；上游未发现测试命令、未执行 | 26/26 文件，599 s，22 py / 6,349 行；远端 compileall 通过、入口冒烟失败（嵌套包） |
| 调用 | 413 次，`reasoning_tokens` 0，无 `length` 截断，每次 `max_tokens` **8192**（目录钳制，下一轮 32768） | 346 次，`reasoning_tokens` 0，每次 32768 |
| 备注 | 后置闸门因运行中改脚本而手动补跑（同一段代码），全过；未判分 | 阿里云 ecs.c7.xlarge，两次 environment_run；未判分 |


### 1.1 sapg 第一对分数（2026-09-17，裁判 DeepSeek-V4-Flash，解析器 V4-Pro，code_only，77 个 Code-Dev 叶）

口径：两边 DeepSeek-V4-Flash、思考关（全程 `reasoning_tokens` 0）、每次调用 32768、同一份 `paper.md` + addendum、同黑名单；本线跑在阿里云 ecs.c7.xlarge，基线在本机。

| 提交 | 总分 | SAPG 实现 (w1) | 实验设置 (w1) | Fig.2 / 5 / 7 / 8 | 产物 | 过程 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 `trial2`（DeepCode 21ebc57f + 补丁） | **0.3374** | 0.413 | 0.611 | 0 / 0 / 0 / 1 | 19 py / 5,143 行 | 96 min，702 次调用，IsaacGymEnvs 预筛回退全量 |
| DeepEvol 线 `deepevol_2`（run 09170541607c） | **0.3180** | 0.382 | 0.526 | 0 / 0 / 0 / 1 | 21 py / 6,279 行 | 76 min，544 次调用，入口冒烟通过 |

叶级：77 个 Code-Dev 叶里两边同过 22，只有本线过 9，只有基线过 11。差 0.019，远小于历史组内摆动（0.09–0.19），**不能说谁优**；两边一起丢的是三张结果图的整条子树（Figure 2/5/7 全 0）、"五个种子"、"六个策略"、"0.001 熵系数"，本线另丢了"熵系数二选一"和 hard 任务的一半。
判分：两份共 4 分钟；Flash 每份约 3.1–3.2M 入 / 0.28–0.30M 出 token，Pro 解析器每份 0.1M 入。原始 `grade.json` 在本地 `archive/deepcode_test/sapg/grades/`。
同批还有一轮基线 `trial1`（`max_tokens` 被钳在 8192，未判，产物在 `~/pb_submissions_archive/sapg/trial1_maxtok8192/`）和本线 C9 运行（未判）。

## 2. sequential-neural-score-estimation（2026-09-14，对标前的数，裁判修 bug 前）

裁判 DeepSeek-V4-Pro @ Paratera，67 叶，无效叶 0；**三份都是思考开的分数**（当时 `enable_thinking:false` 无效，49 次调用 reasoning 78.7 万 / completion 113 万 token）；
三方输入不对标（DeepCode 只拿 paper.md、DeepEvol 只拿 pdf + 手工上传两仓库、裸跑未起）。

| 提交 | 配置 | 得分 | 规模 |
| --- | --- | --- | --- |
| deepevol_s10 | DeepEvol 复现线（旧架构）Stage 10 导出（写码 + 真环境 + 冒烟修复轮后） | 0.7729 | 25 py / 3,854 行 |
| deepcode_trial2 | DeepCode + V4-Pro（索引 max_tokens 16000） | 0.7280 | 35 py / 11,431 行 |
| deepevol_s9 | DeepEvol Stage 9 导出（只写码 + judge，未经修复轮） | 0.6854 | 25 py / 3,817 行 |

| 一级维度（权重） | deepevol_s10 | deepevol_s9 | deepcode_trial2 |
| --- | --- | --- | --- |
| 任务集（App. E.1）(10) | 0.889 | 0.778 | 0.778 |
| VESDE / VPSDE (10) | 0.944 | 0.833 | 0.833 |
| 基线 NPE / SNPE / TSNPE (10) | 0.667 | 0.500 | 0.000 |
| C2ST (1) | 0.000 | 0.000 | 0.000 |
| NPSE (20) | 1.000 | 0.750 | 1.000 |
| TSNPSE (20) | 0.660 | 0.706 | 0.752 |
| 第 5 节结果复现 (20) | 0.607 | 0.607 | 0.755 |

观察（机制层面仍成立）：DeepCode 索引了 `sbi`（NPE/SNPE 就在里面）却一个基线没写；DeepCode 的实验脚本铺得更全。
过程事实：DeepCode trial1（思考开、索引 8000/4000 截断丢 20/51 文件）与第一次 trial2（索引 16000 但思考开）作废；有效 trial2 4 h 26 min，写码期 Paratera 空响应重试 8 次。
DeepEvol 侧 68 次调用约 1,015 万 token ≈ ¥34，跑在租的阿里云 `ecs.c7.2xlarge`。判分每份约 ¥38。

## 3. fre（2026-08-26 → 08-29；**作废：输入有偏**）

306 叶；裁判 DeepSeek-V4-Pro；SF = SiliconFlow serving，PT = Paratera serving（09-03 重判）。DeepCode = e0767d0 + 旧补丁（15 处未门控改动，见 §8）。

| 提交 | 臂 | 规模 | 耗时 | SF | PT | 失分要点（SF） |
| --- | --- | --- | --- | --- | --- | --- |
| anchor | Claude Code + Sonnet 4.5 裸跑 | 15 py / 2,657 行 | ~1 h | 0.4839 | 0.5044 | 数据集/环境 0.333；GC-IQL/GC-BC 满分、OPAL 0 |
| bare_v4 | Claude Code + V4-Pro 裸跑 | 15 py / 3,070 行 | ~5 h | 0.4817 | 0.4807 | 主方法 0.815、三基线 0.67/0.80/0.94；数据集/环境 0.417 |
| trial1 | DeepCode + V4-Pro | 21 py / 6,751 行 | ~4 h | 0.5184 | 0.4682 | 数据集/环境 0.833 最高；**三基线全 0（文件不存在）** |
| trial5 | DeepCode + V4-Pro | 28 py / 8,438 行 | ~5 h | 0.4246 | 0.3101 | 写得最多分最低；三基线全 0；主方法 0.685 |
| **DeepCode / 裸跑** | | | | **0.98×** | **0.81×** | 论文声称 1.34× |

三个一级维度（权重均 3，SF）：数据集与环境搭建 bare 0.417 / trial1 0.833 / trial5 0.750 / anchor 0.333；方法实现 0.809 / 0.412 / 0.343 / 0.792；训练与评估 0.219 / 0.310 / 0.181 / 0.327。
方法实现细目：FRE 主模型 (3) 0.815 / 0.824 / 0.685 / 0.917；GC-IQL (1) 0.667 / 0 / 0 / 1.0；GC-BC (1) 0.800 / 0 / 0 / 1.0；OPAL (1) 0.944 / 0 / 0 / 0。
机制：规划器一次定死文件树，漏掉的对比方法后面几百轮写码补不回来（trial1 计划里"baselines"只在散文出现 2 次，trial5 0 次）。
反事实"补上基线 → trial1 0.6524 = 裸跑的 1.354×"**被修复轮推翻**（§6：基线有分、主方法下滑）。
CodeRAG 预筛失效不是低分主因：规划先于索引（21:17:12 两条日志）；失效率 40%/20%/40% 与分数 0.5184/0.4378/0.4246 不相关；全量索引是筛选的超集。其代价是时间（google-research 8,885 py 全量需 ~140 h）。
判分侧"无文件"叶：bare_v4 13/306、trial5 13、trial1 9、anchor 3；全按上界修正后倍数 0.95×，主结论不变。

## 4. rice（2026-08-29 → 09-02；**作废：输入有偏**）

178 叶；同上两裁判。

| 提交 | 臂 | 规模 | 耗时 | SF | PT | 失分要点（SF） |
| --- | --- | --- | --- | --- | --- | --- |
| bare_v4 | Claude Code + V4-Pro 裸跑 | 11 py / 4,173 行 | 单次 | 0.4680 | **0.1452** | 环境搭建 0.000（9 个环境一个没写）；实验 II (w4) 0.750 全场最高 |
| trial2 | DeepCode + V4-Pro | 22 py / 12,583 行 | 5 h 21 m，¥31 | 0.5447 | 0.4033 | 环境 0.389、解释方法 0.821、实验 III 0.833 |
| trial3 | DeepCode + V4-Pro | 39 py / 25,630 行 | 6 h 46 m，¥45 | 0.4374 | 0.3446 | 环境 0.575、策略网络 0.875；实验 II 0.396（其中 7 叶是判分侧空 `<files>`） |
| **DeepCode / 裸跑（V4-Pro）** | | | | **1.05×** | **2.58×** | 结论随裁判翻转；论文声称 1.95× |
| bare_kimi | Claude Code + Kimi-K2.7-Code 裸跑 | 18 py / 2,463 行 | 22 min | 0.4633 | 0.1865 | 与裸跑 V4-Pro 几乎重合 |
| trial_k1 | DeepCode + Kimi | 36 py / 12,056 行 | 3 h 02 m | 0.2815 | 0.2403 | 实验逻辑还原不足 |
| trial_k2 | DeepCode + Kimi | 32 py / 8,266 行 | 4 h 15 m | 0.4760 | 0.3754 | 1 个语法错误文件 |
| **DeepCode / 裸跑（Kimi）** | | | | **0.82×** | **1.65×** | 同样随裁判翻转 |

维度（SF，bare_v4 / trial2 / trial3）：环境搭建 (1) 0.000 / 0.389 / 0.575；策略网络 (1) 0.375 / 0.625 / 0.875；解释方法 (2) 0.626 / 0.821 / 0.511；实验 I (3) 0.375 / 0.312 / 0.375；精化方法 (2) 0.463 / 0.500 / 0.350；实验 II (4) 0.750 / 0.583 / 0.396；实验 III (2) 0.306 / 0.833 / 0.444；实验 IV (2) 0.333 / 0.333 / 0.333。
实验 II 按精化方法：Ours 1.00 / 0.88 / 0.75；StateMask-R 0.88 / 0.75 / 0.25；JSRL 0.50 / 0.62 / 0.25；PPO fine-tuning 0.75 / 0.25 / 0.38。
这次不是"没写"：trial3 基线相关文件 7/7/10 个，是裸跑的两倍以上，得分只有 41%；7 个失分叶的裁判原话是 "no files or code"——**裁判在大目录树上挑不到文件**（判分侧静默失败，见 §8）。
"2.58×"来自裸跑塌方（−0.32）而非 DeepCode 上涨（−0.14 / −0.09）：严格裁判专门惩罚"抽象类 + 可配置参数"的写法。
合规审计：两份 DeepCode 产物零上游指纹；trial2 日志里唯一一次黑名单命中是我们补的 URL 拦截挡下了对 `chengzelei/rice` 的 fetch。

## 5. 裁判 serving 依赖（2026-09-03；与输入无关，仍成立）

同一提交、同 rubric、同 PaperBench 版本，只换裁判服务商（模型名都是 DeepSeek-V4-Pro）：

| | SiliconFlow 裁判 | Paratera 裁判 |
| --- | --- | --- |
| fre DeepCode/裸跑 | 0.98× | 0.81× |
| rice DeepCode/裸跑（V4-Pro） | 1.05× | **2.58×** |
| rice DeepCode/裸跑（Kimi） | 0.82× | **1.65×** |

rice bare_v4 判分侧诊断（SF vs PT）：每叶输入 60,757 vs 62,852 token；输出 2,937 vs 1,267；零分叶 87 vs 131；疑似"没看到文件"零分叶 39 vs 35；45 个叶 1→0、1 个 0→1。
两裁判看到相同输入，对"通用可配置实现是否算已实现"判断相反；JudgeEval（§7）裁不了。**任何分数必须连同裁判 serving 一起报告；LLM 裁判分不做目标函数。**

## 6. 作废轮与修复轮（不入统计）

约 30 轮复现里 7 轮因流水线自身问题作废：

| 轮 | 配置 | 原因 | 沉没 |
| --- | --- | --- | --- |
| fre trial2 | DeepCode + V4-Pro | 撞 4 h 写码墙钟 | ¥2.90 |
| fre trial3 | DeepCode + V4-Pro | 写码被 900 s stall 熔断，残缺（判了 0.4378，只作参考） | 已判 |
| fre trial4 | DeepCode + V4-Pro | 语料仅 1 仓库（下载侧；报告本身完整 5 条） | — |
| fre trial6 | DeepCode + V4-Pro | 白天 API 限流，写到 9/24 三次重试打完，`status=incomplete` | ¥19.08 |
| rice trial1 ×5 | DeepCode + V4-Pro | ① 克隆 TLS 断流 ② 换网络节点 ③ 预筛静默回退全量索引（140 h）④ 假计划 ⑤ stall 1800 s 熔断 | ≈¥115 |
| rice trial2（首次） | DeepCode + V4-Pro | 随 trial1 一起因 stall 阈值作废 | ¥15.29 |
| fre trial_fx1 首跑 / 二跑 | + 修复①②③ | 挖掘报告截断只见 1 仓库；调大上限重启 | ≈¥3 |
| **fre trial_fx1** | DeepCode + V4-Pro（PT）+ 修复①②③④ | 完整跑完、PT 0.3618；**提示词含评分元知识，整体作废** | ¥22 |
| **fre trial_fx2** | 同上 | 完整跑完、PT 0.4873；同因作废 | ¥21 |
| 判分批 ×2 | rice 三份 / fre 六份 | 余额耗尽 161~168/178 叶无效；裸模型名未登记上下文表 6 份 64 秒全失败 | ≈¥76 |
| snse trial1、trial2 首次 | DeepCode + V4-Pro | 思考开 + 索引截断；思考仍开 | — |
| bam 修裁判前三份 + 并发 16 对照 | 三臂 | 裁判选文件根目录 bug | 已重判 |

修复轮的维度证据（Paratera 裁判，虽作废但机制观察成立）：

| 细目（权重） | bare_v4 | fx1 | fx2 | trial1 | trial5 |
| --- | --- | --- | --- | --- | --- |
| FRE 主模型 (3) | 0.981 | **0.259** | 0.704 | 0.824 | 0.769 |
| GC-IQL (1) | 1.000 | 0.667 | 1.000 | 0.000 | 0.333 |
| GC-BC (1) | 1.000 | 0.600 | 0.200 | 0.000 | 0.000 |
| OPAL (1) | 0.944 | 0.222 | 0.556 | 0.000 | 0.000 |

基线补上了，主方法塌了；覆盖审计两轮均 `ran + adopted`，计划 16,470→19,534 / 16,257→19,231 字符。这就是 README 里"基线运行不开 fix-①②③"的实证：
①②的提示词按评分维度去补文件，得到的是对 rubric 的拟合，不是引擎能力。

## 7. 裁判校准（JudgeEval，rice/0 作者官方仓库，178 叶，code_only；与输入无关）

| 裁判 | 准确率 | macro F1 | 通过率（人工 0.539） | 偏向 | token（入 / 出） | 花费 | 用时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro @ SiliconFlow（08-26） | 0.685 | 0.685 | 0.449 | 严 9.0 pp | — | ¥27.7 | — |
| DeepSeek-V4-Pro @ Paratera（09-03） | 0.719 | 0.719 | 0.449 | 严 9.0 pp（FP 17 / FN 33） | 7.98M / 0.44M | ¥28 | — |
| **DeepSeek-V4-Flash @ Paratera（09-17）**，解析器仍 V4-Pro | **0.719** | **0.716** | **0.562** | **宽 2.2 pp**（FP 27 / FN 23） | 7.68M / 1.19M（+ Pro 解析 0.22M / 0.02M） | 未查账单；按 Flash ≈ Pro 单价 1/4 折算约 ¥13–25 | 9.5 min |

Flash vs Pro（Paratera，同一批叶子）：叶级一致 138/178（77.5%）；都对 108、只 Flash 对 20、只 Pro 对 20、都错 30。
准确率相同、偏向相反：Pro 偏严（把人判过的判挂），Flash 偏宽且更接近人工通过率。两者都在 gpt-4o 档（0.681），谁也不比谁准。
Flash 的输出 token 是 Pro 的 2.7 倍（思考开着；PaperBench 的判分请求没有关思考的开关，三次都是思考开）。
**坑**：Flash 不能做二级结构化解析器——Paratera 上带 `response_format` 的请求返回前缀重复的坏 JSON（90 秒 7 个叶子失败），
`PB_STRUCTURED_PARSER_MODEL` 必须留 V4-Pro（见 README §6）。原始输出在本地 `archive/paperbench_changes/judge_eval_results_rice_flash/`。

同一提交上两裁判一致 150/178（84.3%）；28 处分歧里都对 111、只 SF 对 11、只 PT 对 17、都错 39。
官方对照（5 卷宏平均 Code-Dev）：o1-high 0.740 / o3-mini 0.720 / gpt-4o 0.681 / gpt-4o-mini 0.588——两个 serving 都在 gpt-4o 档，无法仲裁 rice 的翻转。
原始 JudgeEval 输出：`archive/paperbench_changes/judge_eval_results_rice{,_paratera}/`。

## 8. 工程发现（独立于分数，可单独引用）

DeepCode（e0767d0）流水线里同一模式的四处静默降级——LLM 输出超限或为空后，下游当正常继续、只留 INFO 日志：

| 位置 | 现象 | 后果 |
| --- | --- | --- |
| CodeRAG 预筛（`tools/code_indexer.py`，`max_tokens=2000`） | 大仓库 JSON 截断 | 静默回退全量索引（17 文件仓库成功，151/239 文件 100% 失败） |
| 参考挖掘报告（`maxTokens=4096`） | 报告截断，续写只留尾段 | 下载侧只看见 1/5 仓库，整轮语料贫瘠 |
| 预筛返回合法空列表 | 与"调用失败"共用分支 | 同样回退全量 |
| 判分侧文件选择返回空 | `<files>` 为空 | 叶子静默得 0，`valid_score` 仍为 True |

另：写码阶段从不验证自己的产物——跨 31 个归档 2,443 次工具调用，`command_executor` 被调用 50 次全是 `mkdir`/`touch`/`find`/`ls`/`cat`，没有一次运行生成的代码；索引模式下写码 agent 只有 `write_file` 与 `search_code_references` 两个工具。
旧补丁里 15 处未门控改动（挖掘 maxTokens 4096→8192、`max_iterations` 8→80、墙钟 7200→14400、stall 300→1800、下载 agent 提示词重写 + `max_iterations=40`、fetch 限流、空 code_base fail-fast、分段提示加固等）使"官方默认配置"的表述不成立；本仓库 2026-09-17 起的补丁已全部 env 门控、默认等于上游（README §对上游的改动）。
编译检查：fre/rice 全部有效提交 `py_compile` 通过（trial_k2 31/32）；可运行性不构成任何一方的优势。

试点期（不入统计）：阶段 A 白卷 0.000；阶段 B（规划 DeepSeek + 写码 Kimi，无 CodeRAG）43.3；阶段 B′（+CodeRAG，8 轮拼装、人工裁剪语料）60.5，正式协议下不再现；E1 全 Kimi 干净端到端 14 分钟塌方。

时间线与成本：装环境 08-25~26（13 个坑）→ 试点 08-26 → 协议定案 08-26 晚 → fre 08-26~29 → rice 08-29~31 → Kimi 09-01~02 → 修复验证 09-02~03 → 双裁判重判 09-03 → 公开 09-03 → snse 09-14 → 输入标准 + bam 三方 09-14~15。
总花费约 ¥1,600（约 30 轮复现 + 25 份判分 + 2 次 JudgeEval，含全部废轮）；单价：一轮复现 ≈ ¥20 / 3~6 h，一份判分 ≈ ¥38 / 40~100 min（V4-Pro 裁判）。

## 9. 原始文件在哪（本地，不入库）

`~/Documents/env/paperbench-judge/archive/`：

- `deepcode_test/{fre,rice,sequential-neural-score-estimation,bam}/`：`RESULTS.md`、`grades/*.grade.json`（Paratera 版带 `paratera_` 前缀；作废的文件名标明原因）、`submissions/`、`logs/`、`workspaces/`、`task_archives/`
- `deepcode_test/docs/`：CONCLUSIONS / FINDING_judge_serving_dependence / FINDING_prefilter_silent_failure / FINDING_generic_pipeline_failures / REVIEW_local_changes_2026-09-03 / RESULTS_MASTER / PROJECT_CHRONICLE / DECISIONS / DEEPCODE_INTERNALS / OPTIMIZER_NOTICE / PAPERBENCH_RUNBOOK / ARCHITECTURE_* / HANDOFF_* / CC_FRE_PROMPT.txt（作废的裸跑任务书）
- `paperbench_changes/{judge_eval_results_rice,judge_eval_results_rice_paratera,rubrics}/`
- `DeepCode-e0767d0-patched/`：跑出上面全部 DeepCode 数字的那份源码（上游 e0767d0 + 旧补丁）
