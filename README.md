# DeepCode × PaperBench 验证仓库

> **是什么**：一套可 clone 即跑的脚手架，用来在 PaperBench Code-Dev 口径下跑 **DeepCode 的基线运行**（原装
> [HKUDS/DeepCode](https://github.com/HKUDS/DeepCode) main `21ebc57f` + 一份逐条说明的补丁）并统一判分，
> 作为 DeepEvol 复现线（同一引擎嵌入 DeepEvol 的 Paper2Code 线）的对照。
> 仓库里**只有**验证脚手架、DeepCode 副本 + 补丁、PaperBench 补丁与判分脚本、文档；产物、日志、判分 JSON 都不入库。
> 2026-08-25 → 09-15 的全部历史数字在 [`docs/RESULTS-HISTORY.md`](docs/RESULTS-HISTORY.md)（含作废标记）。

**English.** Scaffolding for running upstream DeepCode (main `21ebc57f`, plus a fully itemised, env-gated patch)
against PaperBench Code-Dev papers under one fixed caliber (DeepSeek-V4-Flash via Paratera, thinking off, same
model for planning and coding, paper.md + addendum as the only input, blacklist enforced at git and MCP level), and
grading submissions with a pinned, patched PaperBench judge. It is the baseline arm for DeepEvol's Paper2Code line.
All earlier results (Aug 25 – Sep 15) are kept in `docs/RESULTS-HISTORY.md` with their void status.

## 1. 用途

三条线共用一套 PaperBench 论文材料与裁判：

| 线 | 是什么 | 在哪跑 |
| --- | --- | --- |
| **DeepEvol 复现线** | DeepCode 的 Paper2Code 引擎嵌入 DeepEvol（`apps/v2/agent/paper2code/`），自己的 provider / 工具 / 执行端口 / 闸门 | DeepEvol 仓库，`scripts/paper2code_canary.py` |
| **基线运行**（本仓库） | 原装 DeepCode + 本仓库补丁，`run_trial.sh` 一轮一摆卷 | 本仓库 |
| 裸跑（bare） | Codex 桌面版 + 官方指令原文 + 冻结后缀 | `deepcode_test/bare/`，本批暂不起 |

术语按 DeepEvol 根 `CONTEXT.md`：**对比方法**（论文里被比较的算法，rubric 里的 baseline）≠ **基线运行**（原装 DeepCode 在同口径下的一次运行）。本文不用裸的"基线"。

## 2. 口径（两边一致，改一处必须两边同时改）

| 项 | 值 | 谁保证 |
| --- | --- | --- |
| 模型 | `DeepSeek-V4-Flash` @ Paratera（`https://llmapi.paratera.com/v1`），规划与写码同一模型，无阶段覆盖；每次调用 `max_tokens` 32768（模板把它声明成带 `maxOutputTokens: 32768` 的手动模型条目，否则 DeepCode 的模型目录按 deepseek 家族缺省钳到 8192） | `config/deepcode_config.template.json`；`run_trial.sh` 口径闸 |
| 思考 | **关**。每次请求带 `thinking: {"type": "disabled"}`（`compat.thinking=disabled`）；回包 `reasoning_tokens` 必须为 0（Paratera 忽略 `enable_thinking:false`，只认这一种写法） | 补丁 `core/providers/protocol_config.py`；跑完 `run_trial.sh` 汇总 llm 日志核验 |
| 输入 | PaperBench 给 agent 的材料：`paper.md` 末尾并入 `# Addendum`（DeepCode 只吃一个 markdown）；不给 rubric/config | `run_trial.sh` [2/3]（与 DeepEvol 线 `intake.compose_input` 字节一致） |
| 黑名单 | `blacklist.txt` 在两层拦：git `insteadOf`（setup.sh）+ MCP 层 `DEEPCODE_URL_DENYLIST`（补丁） | setup.sh / run_trial.sh |
| 预算 | 参考挖掘 40 轮 / 下载 12 轮；挖掘报告 32768、下载 16384、预筛 32000、分析 16000、关系 16000 token；规划限时 600 s；stall 7200 s；写码墙钟 21600 s；14 h 硬顶 | `run_trial.sh` 注入（补丁只把这些做成 env，默认全等于上游） |
| 实验开关 | fix-①②③ **必须关**（§5.3） | `run_trial.sh` 拒绝 `=1` |
| 判分 | PaperBench Code-Dev `code_only=True`，裁判 `DeepSeek-V4-Pro` @ Paratera（`PB_JUDGE_MODEL` 可换；Flash 当裁判在 JudgeEval rice/0 上准确率与 Pro 相同 0.719、偏向相反，见 RESULTS-HISTORY §7；解析器必须留 Pro），`PB_JUDGE_CONCURRENCY=20`，`num_invalid_leaf_nodes ≤ 2` 才有效 | `run_grade.sh` |

## 3. 快速开始（clone 即跑）

前置：Linux / macOS，`git` `curl` `uv` `node`+`npm`（Node ≥ 18）`patch`；判分时要 Docker。不需要 git-lfs。
一把 Paratera 的 OpenAI 兼容 key（复现用 `PARATERA_API_KEY`，裁判用 paperbench/.env 的 `OPENAI_API_KEY`）。

```bash
git clone git@github.com:2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
PAPERS=sapg bash setup.sh        # 稀疏克隆 PaperBench@固定 commit 并打补丁、水合 sapg 资产、校验 DeepCode/、uv sync、
                                 # 生成 .deepcode-home/deepcode_config.json（口径）、设 git 封锁、建 ~/pb_submissions/sapg
```

key 只经环境变量进入：写一个文件（不进仓库），内容一行 `PARATERA_API_KEY=...`，然后：

```bash
PREFLIGHT_ONLY=1 PAPER=sapg ENV_FILE=~/my.env bash deepcode_test/scripts/run_trial.sh      # 免费自检，过口径闸
PAPER=sapg TRIAL=trial1 ENV_FILE=~/my.env nohup bash deepcode_test/scripts/run_trial.sh > run.log 2>&1 &
```

- 全流程 3~6 小时（分段 → 规划 → 参考挖掘 → 克隆 → 索引 → 写码 → 上游自带的测试验证），V4-Flash 约 ¥5~10/轮
- 三道闸门：口径闸（模型 / 思考 / 阶段覆盖 / maxTokens / 7 个 MCP / key 来源）、假计划闸（`planning_result_meta.json.source == generated`）、状态闸 + 产物归属（`completed*`、本轮 `tasks/` 下、`paper.md` 标题核验、≥5 个文件）
- 产物摆到 `~/pb_submissions/<paper>/<trial>/`；日志、并稿输入、任务目录归档、摆卷副本在 `runs/<paper>/`（不入库）
- 所有可调项都是环境变量，默认值即本实验用值（`run_trial.sh` [2/3] 段）；`--help` 打印用法
- 多轮串行：`PAPER=sapg FROM=1 TO=3 bash deepcode_test/scripts/run_all_trials.sh`；进度快照 `deepcode_test/scripts/monitor/trial_tick.sh`

判分（花钱，先 DRY 看报价）：

```bash
PAPER=sapg DRY=1 bash deepcode_test/scripts/run_grade.sh     # 只清点与报价
PAPER=sapg bash deepcode_test/scripts/run_grade.sh           # 真判，约 ¥38/份，40~100 分钟；结果 runs/sapg/grades/
```

判 `~/pb_submissions/<paper>/` 下全部提交，脚本自动设 `paperbench.n_tries`；判完把提交移到 `~/pb_submissions_archive/`，否则重判白花钱。
**判新论文前**：PaperBench 的 `paper_split` 是硬编码枚举，要在 `paperbench/nano/eval.py` 的 Literal 里加论文 id 并放一个
`experiments/splits/<paper>.txt`（补丁里已有 fre / rice / sequential-neural-score-estimation / bam 的写法）。

跑自优化循环前先过泄漏闸：`bash deepcode_test/scripts/ci/check_no_rubric_leak.sh`（退出码 0 才可跑；扫描 DeepCode 提示词、补丁、裸跑固定件）。

## 4. 目录

```
.
├── README.md                    ← 本文件
├── setup.sh                     ← 一键环境（幂等）
├── DeepCode/                    ← HKUDS/DeepCode main 21ebc57f + patches/deepcode_local_changes.patch（完整副本；.venv / deepcode_lab 不入库）
├── patches/
│   ├── UPSTREAM_BASE.txt              两个上游仓库的固定 commit
│   ├── deepcode_local_changes.patch   DeepCode 全部改动（15 文件，+628/−49，§5）
│   ├── deepcode_patched.sha256        打过补丁的 15 个文件的 sha256
│   ├── verify_deepcode.sh             证明 DeepCode/ = 上游 + patch（setup.sh 自动跑）
│   └── paperbench_local_changes.patch PaperBench 全部改动（5 文件，§5.4）
├── config/                      ← deepcode_config.template.json（口径）、credentials.example.json、paperbench.env.example（无密钥）
├── deepcode_test/
│   ├── scripts/                       run_trial.sh · run_grade.sh · stage_b_driver.py · run_all_trials.sh · paratera_key.sh
│   │   ├── gates/exec_level.py        执行级结构判据（确定性、零成本，给自优化循环当目标函数）
│   │   ├── ci/check_no_rubric_leak.sh 评分知识泄漏扫描
│   │   └── monitor/                   进度快照
│   └── bare/                          裸跑臂固定件：bare_prompt_suffix.txt · paratera_proxy.py
├── paperbench_changes/          ← PaperBench 改动文件副本 + 新增（单篇 split、裁判偏差分析脚本）
├── docs/
│   ├── RESULTS-HISTORY.md             全部历史数字与结论（含作废标记）
│   ├── INPUT_STANDARD.md              三方输入标准（依据、三层规则、起跑前核验）
│   └── PITFALLS.md                    踩坑总表（60 余条）
├── runs/                        ← 每轮的日志 / 输入 / 任务归档 / 摆卷副本 / 判分 JSON（gitignore）
├── .deepcode-home/              ← setup.sh 生成的 DeepCode 配置目录（gitignore；与你机器上其它 DeepCode 完全隔离）
├── .mcp-node/                   ← setup.sh 装的 filesystem MCP 服务器（gitignore）
└── frontier-evals/              ← setup.sh 稀疏克隆的 PaperBench（gitignore）
```

`~/pb_submissions/<paper>/<trial>/` 是判分器硬性要求的提交池，根目录下每个子目录名必须是合法 paper id。

## 5. 对上游的改动

### 5.1 DeepCode（`patches/deepcode_local_changes.patch`，15 文件，+628/−49）

原则：每个改动 env 门控、默认值等于上游；`verify_deepcode.sh` 证明 `DeepCode/` 一个字节不多改。分四组：

**A. 带走子集**——与 DeepEvol 线 `apps/v2/agent_engine/paper2code/VENDOR.md` 第 2–7 条一一对应，两边引擎行为一致：

| 文件 | 改动 | 旋钮（默认 = 上游） |
| --- | --- | --- |
| `core/compat/agent.py` | `tool_filter` 按消毒后的前缀（`-`→`_`）匹配 | — |
| `tools/code_indexer.py` | 预筛 / 逐文件分析 / 关系抽取的 `max_tokens` env 化 | `DEEPCODE_PREFILTER_MAX_TOKENS` 2000 · `DEEPCODE_ANALYSIS_MAX_TOKENS` 1000 · `DEEPCODE_RELATIONSHIP_MAX_TOKENS` 1500 |
| `workflows/agent_orchestration_engine.py` | 下载 agent：工具优先的提示、只给 `git_clone`、一次纠正重试、空 `code_base` fail-fast；挖掘 / 下载的输出上限与迭代预算 env 化 | `DEEPCODE_REFERENCE_MAX_TOKENS` 8192 · `DEEPCODE_DOWNLOAD_MAX_TOKENS` 4096 · `DEEPCODE_REFERENCE_MAX_ITERATIONS` 8 · `DEEPCODE_DOWNLOAD_MAX_ITERATIONS` 8 |
| `workflows/agents/document_segmentation_agent.py` | 分段 agent 必须真的调工具；`document_index.json` 不存在即失败 | — |
| `workflows/code_implementation_workflow.py` | 写码墙钟与 stall 阈值 env 化 | `DEEPCODE_MAX_WALL_SECONDS` 7200 · `DEEPCODE_STALL_THRESHOLD` 不设 = 上游 300 |
| `workflows/codebase_index_workflow.py` | f-string 里的反斜杠提到表达式外（3.11 兼容） | — |

**B. 基线运行必需**——DeepEvol 线在自己的 provider / 工具层原生具备，基线补齐才是同口径：

| 文件 | 改动 | 旋钮 |
| --- | --- | --- |
| `core/agent_runtime/tools/mcp.py` | 模型可见的工具名 `-`→`_`（Kimi 对含连字符的名字静默不调用）；`DEEPCODE_URL_DENYLIST` 在 MCP 层拒绝黑名单 URL（论文 §4.1 声称、开源代码未实现）；同一 URL 只允许 fetch 2 次 | `DEEPCODE_URL_DENYLIST` 空 |
| `core/compat/request_params.py`、`core/providers/base.py` | 重试模式与退避 env 化 | `DEEPCODE_LLM_RETRY_MODE` standard · `DEEPCODE_CHAT_RETRY_DELAYS` 1,2,4 · `DEEPCODE_PERSISTENT_MAX_DELAY` 60 · `DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT` 10 |
| `core/providers/protocol_config.py` | `compat.thinking: enabled\|disabled` → 每次请求 `extra_body.thinking={"type": …}`（口径开关） | 配置项 |
| `workflows/planning_runtime.py` | 只认**未缩进**的 ``` 围栏定界计划；block scalar 里嵌套的 bash/python 围栏曾把完整计划截成半个并判校验失败 | — |
| `workflows/agents/code_implementation_agent.py`、`workflows/code_implementation_workflow.py` | 统计 `write_multiple_files` 写的文件（批量写入的运行曾报 0 个文件、永远到不了完成判定）；写类工具的循环检测按参数摘要键（写不同文件是进展，不是死循环）；验证根目录下钻唯一子项目 | — |
| `core/verification.py` | `resolve_project_root` | — |
| `workflows/environment.py` | stdlib logger 的 `{}` 占位适配 | — |

B 组后四行来自 DeepCode 维护者本机 main 工作树上尚未提交的修复，与 DeepEvol 线 vendor 的引擎相同。

**C. 实验开关（默认关，基线运行禁止开）**：

| 开关 | 做什么 |
| --- | --- |
| `DEEPCODE_PLAN_COVERAGE_CHECK=1`（fix-①） | 蓝图出完后追加一次"审计"调用：按 (1) 每个对比方法 (2) 每个正文实验 (3) 每个数据集/环境 查漏并补文件 |
| `DEEPCODE_ALLOW_PLAN_EXTENSION=1`（fix-②） | 写码循环里告诉模型"蓝图不是上限，缺对比方法或实验的文件就新建" |
| `DEEPCODE_POSTWRITE_COMPILE=1`（fix-③） | 每次写文件后本机 `py_compile`，失败回灌让模型重写 |

**为什么基线运行不开它们**：①② 的提示词就是 PaperBench rubric 的三个评分维度。带着它们跑出来的分数衡量的是"我们对 rubric 的了解"，
不是引擎——这就是对评测过拟合。修复轮 fx1/fx2 的实证（`docs/RESULTS-HISTORY.md` §6）：基线补上了、主方法从 0.82 塌到 0.26。
基线运行的意义是给 DeepEvol 线一个原装的参照，所以必须关着；留在仓库只为让人能 A/B 出这三条各值多少分。
DeepEvol 线也不带 ①②：对比方法的覆盖交给计划审阅（`--ask`）和将来的自建循环；③ 由远端 `compileall` 作业机械完成。

**D. 不再带的**：旧补丁里"索引产物齐全即跳过重建"（静默跳过与按阶段重跑冲突）、`utils/loop_detector.py` 的写类工具豁免（被 B 组按参数摘要键取代）、写死的 `max_iterations=40/80`、墙钟 14400、stall 1800（全部改为 env，默认上游）。

### 5.2 DeepCode 维护者应知道的

`DeepCode/` 是上游 `21ebc57f` 的完整副本（含 desktop / tests / website，1,011 文件），只有补丁里的 15 个文件不同。
换上游 commit 的步骤：`git archive <commit>` 解到 `DeepCode/`，`patch -p1 < patches/deepcode_local_changes.patch`，手工合并失败的 hunk，
`git diff` 重生成补丁，重算 `deepcode_patched.sha256`，改 `UPSTREAM_BASE.txt`。

### 5.3 与 DeepEvol 线的差异（补丁之外）

| 项 | 基线运行（本仓库） | DeepEvol 线 |
| --- | --- | --- |
| provider | DeepCode 自带 `openai_compat`，retry 走 persistent env | 自己的 `ParateraProvider`，每次回包核对 `reasoning_tokens`，非零即事件 |
| 工具 | 7 个 stdio MCP 服务器（npx / uvx / venv python） | 同一批工具进程内包装，无 MCP 进程 |
| 执行 | 上游自带：发现到测试命令就在**本机**跑 | 远端容器（阿里云租期）+ `compileall` + 入口冒烟 |
| 闸门 | `run_trial.sh` 三道 | preflight / plan_source / implementation_status / ownership 四道 |
| 计划审阅 | 无 | `--ask` 文件式审阅 |

### 5.4 PaperBench（`patches/paperbench_local_changes.patch`，5 文件）

| 文件 | 改动 |
| --- | --- |
| `common/preparedness_turn_completer/.../utils.py` | 上下文长度表登记 `DeepSeek-V4-Pro` 与 `DeepSeek-V4-Flash`（各带/不带 `deepseek-ai/` 前缀；该表只认 OpenAI 模型名，换裁判模型要再加） |
| `paperbench/judge/simple.py` | 结构化解析模型可由 `PB_STRUCTURED_PARSER_MODEL` 指定；叶子并发 `PB_JUDGE_CONCURRENCY`（默认 20，上游 100 会被 Paratera 打 429）；**选文件路径解析修复**（只做精确解析，允许带或不带唯一顶层目录，选不到就重问一次，仍空则记无效叶而不是判 0） |
| `paperbench/grade.py` | 摆卷 tar 解开后若只有一个顶层目录就从里面判；每叶日志落到 `runs/<group>/<run>/judge_logs/` |
| `paperbench/nano/eval.py` | `paper_split` 允许单篇 split（fre / rice / sequential-neural-score-estimation / bam / lite） |
| `paperbench/utils.py` | `is_docker_running` 走 `docker.from_env()`，尊重 `DOCKER_HOST`（macOS Docker Desktop 的 socket 不在 /var/run） |

未改动裁判提示词与评分树。

## 6. 坑（先读这一节再开跑；全表 60 余条在 `docs/PITFALLS.md`）

| 现象 | 根因 | 本仓库怎么处理 |
| --- | --- | --- |
| "思考关"没关，70% 输出 token 是思考 | Paratera 忽略 `enable_thinking:false` | 只认 `thinking:{type:disabled}`（`compat.thinking`）；`run_trial.sh` 跑完汇总 `reasoning_tokens`，非 0 即口径失败 |
| 所有 agent 零工具空转、产物为空 | `deepcode init` 不写 `tools.mcpServers` | 模板带 7 个服务器，`python -m tools.xxx` 模块方式启动；filesystem / fetch 装成固定路径（`npx`/`uvx` 首次解析 20 s+ 会撞 MCP 连接超时，`uvx` 还会重编 cryptography）；口径闸检查齐全 |
| `filesystem` MCP 一连就 `Connection closed` | 服务器启动时校验允许目录存在，`deepcode_lab/` 还没建 | `setup.sh` 先 `mkdir -p DeepCode/deepcode_lab` |
| 配置写了 `maxTokens: 32768`，日志却是 `Resolved workflow LLM … max_tokens=8192`，长文件会被截断 | 21ebc57f 的执行档按模型目录的 `maxOutputTokens` 钳 `max_tokens`；手动目录里只写模型名字符串时，deepseek 家族缺省 8192 | 模板把模型写成对象 `{id, contextWindow, maxOutputTokens: 32768}`；口径闸核对。**sapg trial1（2026-09-17）是在 8192 下跑的**，见 RESULTS-HISTORY |
| 参考挖掘 / 下载 agent 8 轮就放弃、报告是 runner 的"到达上限"文本 | 上游 `max_iterations=8` | 40 / 12（两次真机 8 都不够） |
| 挖掘报告截断，下载侧只见 1 个仓库 | `maxTokens=4096/8192` | 32768 / 16384 |
| CodeRAG 预筛 JSON 截断 → 静默回退全量索引（8,885 文件仓库需 140 h） | `max_tokens=2000` | 32000；分析/关系 16000 |
| 规划三连败后上游伪造通用脚手架计划并标 `completeness_score=1.0` | `coerce_text_to_minimal_plan` | 假计划闸；规划限时 600 s |
| 完整计划被判校验失败 → 假计划 | 计划里嵌套的 bash 围栏截断了 YAML 提取 | 补丁 B：只认未缩进围栏 |
| 写码报 0 个文件、永远不完成 | 只统计 `write_file`，模型用了 `write_multiple_files` | 补丁 B：批量写也计数 |
| 连续 `write_file` 被当死循环杀掉 | 循环检测只看工具名 | 补丁 B：写类工具按参数摘要键 |
| 白天 429/5xx/空响应，三次重试打完整轮报废 | 上游 1/2/4 秒三次 | persistent：10/30/60/180/300 s，上限 900 s，同错 30 次 |
| 300 s 无落盘即熔断，白天空响应期一次 30~50 分钟 | stall 阈值 | 7200 s；墙钟 21600 s；14 h 硬顶 |
| 下载 agent 自主克隆论文官方仓库 | 论文声称的黑名单开源版没有 | git insteadOf + MCP 层 `DEEPCODE_URL_DENYLIST`（实测挡下过一次） |
| `DEEPCODE_WORKSPACE=<路径>` 一设，加载配置就报 `error parsing value for field "workspace"` | 上游 `DeepCodeConfig` 用 pydantic-settings 的 `DEEPCODE_` 前缀读环境变量，同名变量被当成 `workspace` 配置对象解析 | 不设它；工作区用 cwd 默认 `DeepCode/deepcode_lab`（`run_trial.sh` 在 DeepCode/ 里起 driver） |
| 老任务目录混入新轮 / 拿错论文摆卷 | `deepcode_lab/tasks` 未清、交接文件跨论文 stale | 开跑前归档全部 `paper_*`；按论文分交接文件 + `paper.md` 标题核验 |
| 只判了 1 份，其余无声忽略 | 每个 task 实例只 `pop()` 一份提交 | `run_grade.sh` 自动数目录设 `n_tries` |
| 判分中途余额耗尽，分数被压低但看似正常 | 150+ 叶无效仍出总分 | `num_invalid_leaf_nodes ≤ 2` 否则作废；Paratera 余额耗尽不报 402 而是 403 `team_model_access_denied` + 模型表从 93 掉到 8，开跑前 `paratera_key.sh check` |
| Flash 当裁判时二级解析器成片坏 JSON（`{"valid{"valid_score…`） | Paratera 的 Flash 在 `response_format` 请求上返回打乱的正文 | `PB_STRUCTURED_PARSER_MODEL=DeepSeek-V4-Pro` 保持不变，只把裁判换成 Flash |
| 裁判"没看到文件"给 0 且 `valid_score=True` | 模型省掉树根 `submission/` | PaperBench 补丁：精确解析 + 重问 + 记无效叶 |
| 论文资产全是 LFS 指针 | 稀疏/浅克隆下 `git lfs pull` 拿不到对象 | `setup.sh` 从 `media.githubusercontent.com` 按固定 commit 直链下载 |
| `pkill -f "xxx"` 把自己杀了；改运行中的脚本错位执行 | 匹配到自己；bash 逐行读脚本 | `pkill -f "xx[x]"`；运行中的脚本不改 |
| 提示词里一句 "Graders assign separate credit…" | 评分元知识进流水线 | 两轮整体作废；`ci/check_no_rubric_leak.sh`；rubric 物理不进工作区 |
| 每组 2 轮就下结论 | 组内摆动 0.13~0.16，组间 0.01~0.02 | n ≥ 5 才说"优于"；任何分数带裁判 serving |

## 7. 作废规则

只统计**完整跑完且判分有效**的轮次。一轮作废的条件：

- 口径闸没过就跑了（模型 / 思考 / 阶段覆盖任一不符）；跑完 `reasoning_tokens` 合计非 0
- 假计划（`planning_result_meta.json.source != generated`）、流水线状态非 `completed*`、产物不在本轮 `tasks/` 下或 `paper.md` 标题对不上、产物 < 5 文件
- 判分 `num_invalid_leaf_nodes > 2`（余额耗尽、裁判模型名未登记上下文表）
- 任何评分知识进入提示词或工作区（rubric / config.yaml 出现在 agent 目录即作废）
- 三方输入不一致（材料五样 sha256 不同、裸跑提示词 diff 非空）

作废轮的产物与日志照样归档，文件名标明原因，只作机制分析。

## 8. 历史结果索引

全部在 [`docs/RESULTS-HISTORY.md`](docs/RESULTS-HISTORY.md)：§1 bam 三方（09-15，当前唯一有效）· §2 snse（09-14，对标前）· §3 fre · §4 rice + Kimi（08-25→09-03，输入有偏作废）·
§5 裁判 serving 依赖 · §6 作废轮与修复轮证据 · §7 JudgeEval 校准 · §8 工程发现 · §9 原始文件位置（本地 archive）。
2026-09-17 起 sapg 的两边并排数字先记在 DeepEvol 仓库 `apps/v2/agent/paper2code/HANDOFF.md`，判分后回填。

## 9. 诚实声明

- 09-14 之前的全部对比作废（输入有偏），09-15 修裁判前的全部分数作废；两条都在 RESULTS-HISTORY §0。
- 样本量：至今每组 1~2 轮，组内摆动远大于组间差距；只能看方向。
- 裁判：两家 serving 的同名模型判分行为不同（同一提交 16% 叶级分歧），JudgeEval 上二者同等水平；绝对分数与倍数必须连同裁判 serving 一起报告，且不能与论文数字直接相减。
- 配置：旧补丁曾有 15 处未门控改动；本版全部 env 门控、默认等于上游（`verify_deepcode.sh` 可证）。上游自带的测试验证会在本机跑生成的代码（DeepEvol 线在容器里跑），这是两边一处已知不对称。
- 底座：论文用 Sonnet 4.5 / o3-mini 裁判；我们用 DeepSeek-V4-Flash（复现）/ V4-Pro（裁判），同底座保证公平但不能直接对照论文数字。
- 费用：历史全部实验约 ¥1,600；V4-Flash 一轮基线运行约 ¥5~10，一份判分约 ¥38。

## 10. 许可证

DeepCode（HKUDS，MIT）与 frontier-evals / PaperBench（OpenAI，MIT）各自的 LICENSE 随源码保留（`DeepCode/LICENSE`、`paperbench_changes/LICENSE.md`）。
本仓库新增的脚本、补丁与文档同样以 MIT 发布。论文原文不随仓库分发，由 `setup.sh` 从上游 LFS 直链拉取。
