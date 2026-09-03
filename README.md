# DeepCode × PaperBench 独立复现验证

> 验证 DeepCode(arXiv:2512.07921,HKUDS/DeepCode)"自动论文复现比商业编码工具高 1.34×(fre)/ 1.95×(rice)"的声称。
> 口径:PaperBench Code-Dev(只看代码不执行),同一底座模型双切(裸跑 vs DeepCode),裁判固定。
> 本仓库包含:修改版 DeepCode 源码、PaperBench 补丁、全部脚本、全部提交产物与判分 JSON、分析文档。**clone 后跑 `setup.sh` 即可复现整个流程。**

**English summary.** An independent replication of the DeepCode paper's claim that its paper-to-code scaffold beats bare coding agents on PaperBench Code-Dev. Same base model (DeepSeek-V4-Pro, plus a Kimi-K2.7 line) on both arms, fixed judge. Headline: on `fre` the scaffold shows no gain under either of two judge servings (0.98× / 0.81×); on `rice` the verdict **flips with the judge serving** (1.05× under SiliconFlow-served DeepSeek-V4-Pro vs 2.58× under Paratera-served DeepSeek-V4-Pro), because the two servings disagree on whether "generic, configurable" implementations count. PaperBench JudgeEval (human-graded ground truth, rice/0) rates both servings as equally accurate (macro F1 0.685 vs 0.719, identical pass rate and bias) while they disagree on 16% of leaves — so the ground truth cannot arbitrate the flip, and that disagreement is the judge-noise floor of Code-Dev scoring with these models. We also document four silent-degradation defects in the DeepCode pipeline and a full audit of our own modifications.

---

## 0. 如果你是来造自己的复现 agent 的(先读这一节)

**这个仓库对你的意义**:它不是一个可以直接用的复现 agent,而是三样东西——(a) 上游 **DeepCode**(HKUDS,`e0767d0`)的修改版源码与全部改动记录,是目前唯一有成果可参照的同类开源实现;(b) 对它做的两周独立验证的**全部证据**(11 份提交 × 两个裁判 serving、维度级失分分析、JudgeEval 校准);(c) 从证据推出的**自建 agent 架构设计与踩坑总表**。用户在从 0 到 1 自建论文复现 agent(不能接 Claude Code 壳,模型只走 Paratera 的 OpenAI 兼容端点),本仓库是它的参考基线。

**和上游 DeepCode 的关系**

| | 是什么 | 在哪 |
| --- | --- | --- |
| 上游原版 | https://github.com/HKUDS/DeepCode @ `e0767d0` | `patches/UPSTREAM_BASE.txt` |
| 我们的改动 | 12 文件 +462/−46:env 门控的抗限流与截断修复、4 个实验开关、**15 处未门控改动**(REVIEW 文档逐条核验) | `patches/deepcode_local_changes.patch`;`DeepCode/` 是打过补丁的完整源码 |
| 哪些改动值得带走 | 工具名消毒、URL 黑名单、假计划闸、状态闸、预筛/挖掘/下载上限 env 化、persistent 重试 | `deepcode_test/scripts/run_trial.sh` + patch |
| 哪些改动不要带走 | 两个实验开关提示词里的评分元知识("Graders assign…");写死的 `max_iterations=80` 等未门控值 | `docs/REVIEW_local_changes_2026-09-03.md` |
| DeepCode 里值得复用的 | 前半段:参考挖掘、下载、CodeRAG 索引与检索、文档分段(产物是独立文件) | `docs/ARCHITECTURE_v0.2_OPTIMAL.md` §4 A1、§8 复用地图 |
| DeepCode 里要整体替换的 | 后半段 Phase 9 写码循环(不执行、规划冻结、预算常量、clean-slate、2 个工具) | 同上 §4 A3 |

**阅读顺序(约 1 小时)**

1. 本 README §1(结论)与 §1.3(工程发现)—— 5 分钟
2. `deepcode_test/docs/PROJECT_CHRONICLE.md` —— **前因后果全纪事**:从 08-25 装环境到 09-03 转向自建,每一步做了什么、得到什么数字、用户在对话里问了什么、我们因此改了什么;配套 `RESULTS_MASTER.md`(全部结果总表,含作废轮与试点)与 `DECISIONS.md`(22 条决策记录)—— 20 分钟
3. `deepcode_test/docs/PITFALLS.md` —— **踩坑总表,60 余条,分 PaperBench / DeepCode 配置 / DeepCode 源码 / 供应商 / 方法学 / 运维**,大多数会在新系统里原样重现 —— 15 分钟
4. `deepcode_test/docs/ARCHITECTURE_v0.2_OPTIMAL.md` —— 自建 agent 的目标架构、建造顺序、每层验收、复用地图、给下一个对话的操作指引 —— 15 分钟
5. 需要细节时:`docs/CONCLUSIONS.md`(失分机制)、`docs/FINDING_judge_serving_dependence.md`(为什么裁判分不能做目标)、`docs/FINDING_prefilter_silent_failure.md`(四种静默降级)、`docs/REVIEW_local_changes_2026-09-03.md`(我们自己的改动哪些没守住纪律)、`docs/ARCHITECTURE_PROPOSAL_v0.1.md`(周末降配版与 AutoSOTA 逐条)

**五条不要做的事**(每条都有真金白银的教训):不用 LLM 裁判分做目标函数;不让任何评分表信息进入流水线或提示词;不在没有执行验证的循环里写码;不让 LLM→程序的任何接缝静默降级;每组 <5 轮不下结论。

## 1. 结论(先看这个)

### 1.1 双裁判对照

所有提交、rubric、PaperBench 版本完全相同;只换裁判的服务商(模型名都是 DeepSeek-V4-Pro)。

**fre(306 叶)**

| 提交 | SiliconFlow 裁判 | Paratera 裁判 |
| --- | --- | --- |
| bare_v4 —— Claude Code 壳 + V4-Pro 裸跑 | 0.4817 | 0.4807 |
| anchor —— Claude Code + Sonnet 4.5 裸跑 | 0.4839 | 0.5044 |
| trial1 —— DeepCode + V4-Pro | 0.5184 | 0.4682 |
| trial5 —— DeepCode + V4-Pro | 0.4246 | 0.3101 |
| **DeepCode / 裸跑** | **0.98×** | **0.81×** |

**rice(178 叶)**

| 提交 | SiliconFlow 裁判 | Paratera 裁判 |
| --- | --- | --- |
| bare_v4 —— 裸跑 V4-Pro | 0.4680 | **0.1452** |
| trial2 / trial3 —— DeepCode + V4-Pro | 0.5447 / 0.4374 | 0.4033 / 0.3446 |
| **DeepCode / 裸跑** | **1.05×** | **2.58×** |
| bare_kimi —— 裸跑 Kimi-K2.7-Code | 0.4633 | **0.1865** |
| trial_k1 / trial_k2 —— DeepCode + Kimi | 0.2815 / 0.4760 | 0.2403 / 0.3754 |
| **DeepCode / 裸跑** | **0.82×** | **1.65×** |

论文声称:fre 1.34×、rice 1.95×。

### 1.2 结论总结

**能站住的**

| # | 结论 | 证据 |
| --- | --- | --- |
| 1 | **fre 上没有增益** | 两个裁判一致(0.98× / 0.81×);失分位置一致(规划器漏掉 GC-IQL / GC-BC / OPAL 全部基线);"补上基线即可到 1.35×"的反事实被修复轮推翻(基线有分、主方法下滑) |
| 2 | **Code-Dev 分数的精度不足以支撑论文声称的量级** | 同名裁判换一家 serving,同一份代码 16% 叶级分歧;人工标注上两裁判同等准确(F1 0.685 / 0.719,通过率与偏向完全相同);这个噪声底与论文的效应量同量级,而论文只用一个裁判、未报告此方差 |
| 3 | **开源版有系统性静默降级** | 检索 / 挖掘 / 判分三侧四种"输出超限或为空 → 当正常继续"的缺陷,均有 A/B 实证(§1.3) |
| 4 | **本项目自身口径有瑕疵** | 15 处未门控改动(相对比较仍公平);修复轮因评分知识泄漏作废(§4.1) |

**两边都站不住的**

- rice 有没有增益:1.05× 或 2.58×,取决于裁判,人工标注裁不了。
- 论文夸大了:不能这么说,论文的数字可以由一个合法的 setup 产生。
- "相对效果在某些裁判下确实有这么大":也不能这么说,见下。

**2.58× 是怎么来的:不是 DeepCode 涨了,是裸跑塌了**

| rice | SiliconFlow | Paratera |
| --- | --- | --- |
| 裸跑 | 0.468 | **0.145**(−0.32) |
| DeepCode | 0.545 / 0.437 | 0.403 / 0.345(−0.14 / −0.09) |

严格裁判专门惩罚"抽象类 + 可配置参数"的写法;裸跑代码恰好是这种风格,DeepCode 按环境铺具体文件所以扛得住。2.58× 衡量的是**"写得具体 vs 写得抽象"在严格裁判眼里的差距**,不是"复现得更忠实";而且严格裁判在人工标注上并没有更准。即使论文的 o3-mini 裁判恰好是严格型、1.95× 是真实测得的,它证明的也只是 DeepCode 的输出风格更合裁判胃口。

**一句话**:论文的倍数在本项目的精度内不可判定,但它声称的精度本身站不住;fre 上的增益在任何裁判下都没出现。

**对"要不要用 DeepCode"的含义**:fre 上它没帮上忙;rice 上它的产物在严格裁判下更耐看,但那是"具体"而非"正确"的证据;它的工程状态有系统性盲区,本项目 30 轮里 7 轮因流水线自身问题作废。这个决策不应建立在"某些裁判下有 2.58×"上。它可验证的长处只有一个:**覆盖面**(环境 / 数据集维度普遍占优),这更适合作为"规划 + 语料"前端接给一个会执行验证的 agent,而不是整条流水线。

### 1.3 工程发现(独立于分数,可单独引用)

DeepCode 流水线里同一模式的四处静默降级 —— LLM 输出超限或为空后,下游当正常继续、只留 INFO 日志:

| 位置 | 现象 | 后果 |
| --- | --- | --- |
| CodeRAG 预筛(`tools/code_indexer.py`,`max_tokens=2000`) | 大仓库 JSON 截断 | 静默回退全量索引,论文声称的检索从未生效 |
| 参考挖掘报告(`maxTokens=4096`) | 报告截断,续写只留尾段 | 下载侧只看见 1/5 仓库,整轮语料贫瘠 |
| 预筛返回合法空列表 | 与"调用失败"共用分支 | 同样回退全量 |
| 判分侧文件选择返回空 | `<files>` 为空 | 叶子静默得 0,`valid_score` 仍为 True |

另一条与上表同源的观察:**写码阶段从不验证自己的产物**。命令执行器可用且被调用过 50 次,但全部是 `mkdir`/`touch` 建骨架与 `find`/`ls`/`cat` 查看目录;跨 31 个归档、2,443 次工具调用,**没有一次运行生成的代码**(无 python / pytest / import / pip install)。索引模式下工具面只有 `write_file` 与 `search_code_references`(`code_implementation_workflow.py:78`),执行类工具根本不在写码 agent 的可见范围内。

详见 `deepcode_test/docs/FINDING_prefilter_silent_failure.md`、`FINDING_judge_serving_dependence.md`。

---

## 2. 仓库结构

```
.
├── README.md                    ← 本文件
├── setup.sh                     ← 一键环境搭建(clone 后只需跑这个)
├── DeepCode/                    ← 修改版 DeepCode 源码(上游 e0767d0 + 本地改动;不含 .venv / 运行产物)
├── patches/
│   ├── UPSTREAM_BASE.txt              两个上游仓库的固定 commit
│   ├── deepcode_local_changes.patch   DeepCode 全部改动(git diff HEAD,12 文件)
│   └── paperbench_local_changes.patch PaperBench 全部改动(3 文件)
├── paperbench_changes/          ← PaperBench 侧改动文件副本 + 新增文件(fre/rice split、裁判偏差分析、JudgeEval 结果)
├── config/                      ← ~/.deepcode 与 paperbench/.env 的模板(无密钥)
└── deepcode_test/               ← 实验本体
    ├── README.md                      实验总览与状态
    ├── docs/                          结论、发现、交接、审查报告、裸跑任务书
    ├── scripts/                       run_trial.sh / run_grade.sh / stage_b_driver.py / 监控与链式脚本
    ├── fre/  rice/                    每篇论文:RESULTS.md · submissions/ · grades/ · logs/ · workspaces/ · task_archives/
    └── (frontier-evals/ 由 setup.sh 克隆,不入库)
```

`task_archives/` 保留每轮的计划、挖掘报告、下载汇总、规划元数据与 LLM/MCP 日志;克隆下来的参考仓库与索引卡片(7GB)不入库。

---

## 3. 快速开始(clone 即跑)

### 3.1 前置

- Linux / WSL2,Python 3.11+,`uv`,`git`,`curl`,Docker(判分时起沙箱);不需要 git-lfs,论文资产由 `setup.sh` 从 GitHub 直链按固定 commit 下载
- 一个 OpenAI 兼容 API key,底座与裁判共用。我们用 Paratera 的 DeepSeek-V4-Pro(历史数据里另有 SiliconFlow 的同名模型,见 §1.1)。
  换 key 用 `bash deepcode_test/scripts/paratera_key.sh set <新key>` —— 它会先探活、再把 key 同步写进
  `~/.deepcode/credentials.json`(底座)与 `frontier-evals/.../paperbench/.env`(裁判)两处,避免漏改一处后
  「复现正常、判分全部 401」。另有 `check` / `add` / `list` / `next` 子命令(`next` 在当前 key 失效时自动切备用池)
- 裸跑对照需要 Claude Code(或任何交互式编码 agent)+ 同一底座模型

### 3.2 三步

```bash
git clone https://github.com/2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
bash setup.sh          # 克隆 PaperBench@固定 commit 并打补丁、uv sync、写配置模板、设防作弊封锁
```

填两处 key:`~/.deepcode/credentials.json`(底座)与 `frontier-evals/project/paperbench/.env`(裁判)。然后:

```bash
PREFLIGHT_ONLY=1 PAPER=fre bash deepcode_test/scripts/run_trial.sh     # 免费自检
```

### 3.3 跑一轮 DeepCode 复现

```bash
PAPER=fre TRIAL=trial1 nohup bash deepcode_test/scripts/run_trial.sh > run.log 2>&1 &
```

- 全流程 3~6 小时(参考仓库挖掘 → 索引 → 规划 → 写码),V4-Pro 约 ¥20/轮
- 三道闸门:假计划闸(规划失败后上游会伪造通用计划,一律判废)、状态闸、产物归属核验
- 产物摆到 `~/pb_submissions/<paper>/<trial>/`,并复制到 `deepcode_test/<paper>/submissions/<trial>/`
- 所有可调项都是环境变量(见 `run_trial.sh` 头部注释),默认值即本实验用值

### 3.4 裸跑对照

任务书在 `deepcode_test/docs/CC_FRE_PROMPT.txt`(fre)与 `deepcode_test/rice/workspaces/cc_dsv4_run/PROMPT.txt`(rice,逐字一致仅换路径)。把它喂给 Claude Code(底座切成同一模型),产物放到 `~/pb_submissions/<paper>/bare_<model>/`。裸跑工作区与 git 历史导出见 `deepcode_test/<paper>/workspaces/`。

### 3.5 判分

```bash
PAPER=fre DRY=1 bash deepcode_test/scripts/run_grade.sh   # 只清点与报价
PAPER=fre bash deepcode_test/scripts/run_grade.sh         # 真判,约 ¥38/份,40~100 分钟
```

- 判 `~/pb_submissions/<paper>/` 下全部提交;脚本自动设 `n_tries`(不设只判 1 份,其余静默忽略)
- 判完检查 `num_invalid_leaf_nodes ≤ 2`,否则该份作废(余额耗尽等中途出错会把分数压低)
- 已判提交请移出提交池,否则重判白花钱
- 换裁判服务商时,必须把模型名登记进 `preparedness_turn_completer/utils.py` 的上下文长度表(见 patch)

---

## 3.6 上传了什么、没上传什么(以及为什么)

本仓库**不整份分发 PaperBench**,而是「固定 commit + 补丁 + 我们自己的产物」。`setup.sh` 会按 `patches/UPSTREAM_BASE.txt` 里的 commit 稀疏检出上游并自动打补丁,复现性不受影响。

**已上传**

| 内容 | 位置 | 说明 |
| --- | --- | --- |
| 判分树 rubric | `paperbench_changes/rubrics/{fre,rice}.rubric.json` | 上游原样复制,fre 437 叶 / rice 361 叶,共约 750KB。**仅供事后核对失分分析,严禁进入复现流水线** —— 见该目录 README |
| PaperBench 改动 | `patches/paperbench_local_changes.patch` + `paperbench_changes/modified_files/` | 3 个文件、+29/−6 行,可直接对读 |
| 我们新增的文件 | `paperbench_changes/experiments/splits/`、`analyze_judge_eval_bias.py` | fre/rice 单篇 split、裁判偏差分析脚本 |
| JudgeEval 完整结果 | `paperbench_changes/judge_eval_results_rice{,_paratera}/` | 两个 serving 的原始判分(F1 0.685 / 0.719) |
| DeepCode 修改版全源码 | `DeepCode/` | 上游 `e0767d0` + 本地改动(不含 `.venv`、运行产物) |
| 全部提交产物与判分 JSON | `deepcode_test/{fre,rice}/` | 含作废轮,文件名标注原因 |

**未上传**

| 内容 | 体积 | 原因 |
| --- | --- | --- |
| `data/papers/*/paper.md`、`paper.pdf`、`assets/` | 205MB / 23 篇 | **论文正文与插图,版权属原作者与出版方**,不是我们能再分发的;上游用 Git LFS 托管,`setup.sh` 按需拉取 fre/rice 两篇 |
| `data/papers/rice/judge/` | 37MB | 被引论文(JSRL、StateMask)的 PDF 与插图,同样是第三方版权内容 |
| `data/judge_eval/` | 62MB | 上游 JudgeEval 数据集,其 README 明说部分内容不能自动再分发,须用 `download_data.py` 自取 |
| PaperBench 其余源码 | — | 未改动的上游代码,用 pin + patch 表达比复制一份更准确 |
| `.venv/`、`runs/`、DeepCode `deepcode_lab/`、`task_archives/*/code_base` 与 `indexes/` | 约 8GB | 本地环境与运行中间物,可由脚本重建 |

---

## 4. 对上游的改动

### 4.1 DeepCode(`patches/deepcode_local_changes.patch`,12 文件,+462/−46)

**环境变量门控、默认值等于上游的**(用于抗限流与修工程缺陷,不改生成逻辑):

| 变量 | 作用 | 上游默认 |
| --- | --- | --- |
| `DEEPCODE_PREFILTER_MAX_TOKENS` | CodeRAG 预筛响应上限 | 2000 |
| `DEEPCODE_DOWNLOAD_MAX_TOKENS` | GitHub 下载 agent 输出上限 | 4096 |
| `DEEPCODE_LLM_RETRY_MODE` / `_CHAT_RETRY_DELAYS` / `_PERSISTENT_MAX_DELAY` / `_PERSISTENT_IDENTICAL_ERROR_LIMIT` | 重试与退避 | standard / 1,2,4 / 300 / 30 |
| `DEEPCODE_OPENAI_REQUEST_TIMEOUT_S` / `DEEPCODE_CODE_ANALYZER_TIMEOUT_S` | 请求 / 规划超时 | 上游值 |
| `DEEPCODE_URL_DENYLIST` | fetch/下载层强制执行论文黑名单(论文 §4.1 声称但代码未实现) | 空 |

**实验开关(默认关)**:`DEEPCODE_PLAN_COVERAGE_CHECK`(规划后覆盖审计)、`DEEPCODE_ALLOW_PLAN_EXTENSION`(写码时允许扩展文件树)、`DEEPCODE_POSTWRITE_COMPILE`(写后 `py_compile`)、`DEEPCODE_REFERENCE_MAX_TOKENS`。

**⚠️ 未门控 / 默认漂移的改动(15 处,全部 DeepCode 轮次共享)**:参考挖掘 `maxTokens` 4096→8192、`max_iterations` 8→80,写码墙钟 7200→14400,stall 阈值 300→1800,GitHub 下载 agent 提示词重写与 `max_iterations=40`,fetch 同 URL 限流,空 `code_base` fail-fast,分段提示词加固等。这些是 fre 早期为让流水线跑通所做,**使得"官方默认配置"的表述不成立**;但裸跑 vs DeepCode 的相对比较不受影响(所有 DeepCode 轮次用同一套代码)。完整清单与逐条核验见 `deepcode_test/docs/REVIEW_local_changes_2026-09-03.md`。

**⚠️ 实验开关里的一处评分知识泄漏**:`DEEPCODE_PLAN_COVERAGE_CHECK` 与 `DEEPCODE_ALLOW_PLAN_EXTENSION` 的提示词含 "Graders assign separate credit to each baseline; omitting them forfeits those points",属于 PaperBench 评分结构元知识。用这两个开关跑的 trial_fx1/fx2 已**整体作废**(产物保留在 `fre/submissions/_作废/`),patch 保留原样以忠实记录;重跑前须删除该句。

### 4.2 PaperBench(`patches/paperbench_local_changes.patch`,3 文件)

| 文件 | 改动 |
| --- | --- |
| `common/preparedness_turn_completer/.../utils.py` | 上下文长度表登记 `deepseek-ai/DeepSeek-V4-Pro` 与 `DeepSeek-V4-Pro`(该表只认 OpenAI 模型名,无配置项) |
| `paperbench/judge/simple.py` | 裁判的结构化解析模型可由 `PB_STRUCTURED_PARSER_MODEL` 指定(默认不变;本实验设为 DeepSeek-V4-Pro) |
| `paperbench/nano/eval.py` | `paper_split` 允许 `fre` / `rice` 单篇 split |

新增:`experiments/splits/{fre,rice}.txt`、`analyze_judge_eval_bias.py`、`judge_eval_results_rice/`(SiliconFlow 裁判,F1 = 0.685)与 `judge_eval_results_rice_paratera/`(Paratera 裁判,F1 = 0.719)。未改动裁判提示词、评分树、文件选择逻辑。

---

## 5. 实验流程与作废规则

```
论文 paper.md ──► run_trial.sh(DeepCode 全流程,同底座)──► ~/pb_submissions/<paper>/trialN
                └► Claude Code + 同底座 + PROMPT.txt(裸跑)──► ~/pb_submissions/<paper>/bare_<model>
                                                    │
                                        run_grade.sh(code_only,裁判固定)
                                                    │
                                          grade.json → RESULTS.md
```

**只统计完整跑完且判分有效的轮次。** 作废轮及原因:fre trial3(写码被 stall 截断)、trial4(语料仅 1 仓库,下载侧)、trial6(未完成);rice trial1(三次:网络 / 假计划 / 墙钟);fx1/fx2(评分知识泄漏);两份余额耗尽的判分。全部在 `deepcode_test/<paper>/logs/` 与 `submissions/_作废/` 可查。

---

## 6. 文档索引

| 文件 | 内容 |
| --- | --- |
| `deepcode_test/docs/DEEPCODE_INTERNALS.md` | **DeepCode 是怎么运转的**:11 个 Phase 的职责与产出、`task_dir` 文件合同、LLM 调用点与配置流、31 个归档 2,443 次工具调用的实证、怎么只跑前半段当前端 |
| `deepcode_test/docs/PROJECT_CHRONICLE.md` / `RESULTS_MASTER.md` / `DECISIONS.md` | 全程纪事(含对话转折)/ 全部结果总表 / 决策记录 |
| `deepcode_test/docs/PITFALLS.md` | **踩坑总表**(60 余条,分六类,每条现象/根因/修法/证据) |
| `deepcode_test/docs/ARCHITECTURE_v0.2_OPTIMAL.md` | **最优版架构(交接文档,自含全部事实与路径)**:度量体/复现体分离、建造顺序、每层验收、复用地图 |
| `deepcode_test/docs/ARCHITECTURE_PROPOSAL_v0.1.md` | **下一步:自建论文复现 agent 的架构设计**(多 agent 工作流合成:AutoSOTA 研读 + 计划批评 + 3 提案 6 评审;含周末/下周计划) |
| `deepcode_test/docs/CONCLUSIONS.md` | 总结论、可信度五项核验、逐份失分表、§⑦ 裁判依赖 |
| `deepcode_test/docs/FINDING_judge_serving_dependence.md` | 双裁判对照与诊断(本项目最重要的发现) |
| `deepcode_test/docs/FINDING_prefilter_silent_failure.md` | 四种静默降级的证据链 |
| `deepcode_test/docs/FINDING_generic_pipeline_failures.md` | 可迁移的 LLM 流水线失败模式 |
| `deepcode_test/docs/REVIEW_local_changes_2026-09-03.md` | 对我们自己改动的独立审查(纪律 A/B 核验) |
| `deepcode_test/fre/RESULTS.md` / `rice/RESULTS.md` | 每篇的分数、丢分位置、判分侧问题 |
| `deepcode_test/docs/PAPERBENCH_RUNBOOK.md` | 13 个坑与修法 |
| `deepcode_test/docs/HANDOFF_FRE.md` | 接手文档 |

---

## 7. 诚实声明

- **样本量**:每组 2 轮,组内摆动 0.13~0.16;只能看方向。
- **裁判**:两家 serving 的同名模型判分行为不同(同一提交 16% 叶级分歧),JudgeEval 上二者同等水平、无法裁定。绝对分数与倍数**必须连同裁判 serving 一起报告**。
- **配置**:DeepCode 侧有 15 处未门控改动(§4.1);"官方默认"不成立,相对比较成立。
- **修复线**:①②③④ 是否能恢复增益**未得到有效检验**(fx 轮作废)。
- **底座**:论文用 Sonnet 4.5;我们用 DeepSeek-V4-Pro / Kimi-K2.7,同底座双切保证公平但不能直接对照论文数字。
- **费用**:全部实验约 ¥1,450(约 30 轮复现 + 25 份判分 + 2 次 JudgeEval,含作废)。

## 8. 许可证

DeepCode(HKUDS,MIT)与 frontier-evals / PaperBench(OpenAI,MIT)各自的 LICENSE 随源码保留。本仓库新增的脚本、文档与产物同样以 MIT 发布。论文原文不随仓库分发,由 `setup.sh` 从上游 LFS 拉取。
