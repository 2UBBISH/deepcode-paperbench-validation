# DeepCode × PaperBench 验证仓库

> **是什么**：一套可 clone 即跑的脚手架，用来在 PaperBench Code-Dev 口径下跑 **DeepCode 的基线运行**（原装
> [HKUDS/DeepCode](https://github.com/HKUDS/DeepCode) main `21ebc57f` + 一份逐条说明的补丁）并统一判分，
> 作为 DeepEvol 复现线（同一引擎嵌入 DeepEvol 的 Paper2Code 线）的对照。
> 仓库里**只有**验证脚手架、DeepCode 副本 + 补丁、vendored 的 PaperBench（裁判 + 数据）与判分脚本、裸跑臂的固定件、文档；产物、日志、判分 JSON 都不入库。
> 2026-08-25 → 09-18 的全部历史数字在 [`docs/RESULTS-HISTORY.md`](docs/RESULTS-HISTORY.md)（含作废标记）。
> **2026-09-19 起的批次**：四臂（DeepCode 基线 / Codex CLI / Claude Code CLI / DeepEvol 线 stage 9）× 五篇，同模型同 serving 同思考状态同输入同裁判，
> 一页说明在 [`docs/CODEDEV-ARMS.md`](docs/CODEDEV-ARMS.md)，跑法在 [`deepcode_test/bare/README.md`](deepcode_test/bare/README.md)，本文 §2.1。

**English.** Scaffolding for running upstream DeepCode (main `21ebc57f`, plus a fully itemised, env-gated patch)
against PaperBench Code-Dev papers under one fixed caliber (DeepSeek V4 Flash — Paratera until 2026-09-18, api.deepseek.com
since 09-19 — thinking off, same model for planning and coding, paper.md + addendum as the only input, blacklist enforced
at git and MCP level), running the Codex CLI and Claude Code CLI as bare arms on the same input through a thinking-off
proxy, and grading every submission with the vendored, pinned, patched PaperBench judge. It is the baseline arm for
DeepEvol's Paper2Code line.
All earlier results (Aug 25 – Sep 15) are kept in `docs/RESULTS-HISTORY.md` with their void status.

## 1. 用途

三条线共用一套 PaperBench 论文材料与裁判：

| 线 | 是什么 | 在哪跑 |
| --- | --- | --- |
| **DeepEvol 复现线** | DeepCode 的 Paper2Code 引擎嵌入 DeepEvol（`apps/v2/agent/paper2code/`），自己的 provider / 工具 / 执行端口 / 闸门 | DeepEvol 仓库，`scripts/paper2code_canary.py` |
| **基线运行**（本仓库） | 原装 DeepCode + 本仓库补丁，`run_trial.sh` 一轮一摆卷 | 本仓库 |
| **裸跑臂 Codex CLI** | codex-cli（论文用的形态）+ 官方 `code_only_instructions.txt` + PaperBench 自己的 `ADDITIONAL NOTES`；本地代理注入思考关、换 User-Agent（DeepSeek 对 Codex 客户端无视 `effort=none`） | `deepcode_test/bare/run_bare.sh codex <paper>` |
| **裸跑臂 Claude Code CLI** | `claude -p` + 同一份题面；代理注入 `thinking:{type:disabled}`（Claude Code 自带开关只是不发字段，DeepSeek 缺省开） | `deepcode_test/bare/run_bare.sh claude <paper>` |

术语按 DeepEvol 根 `CONTEXT.md`：**对比方法**（论文里被比较的算法，rubric 里的 baseline）≠ **基线运行**（原装 DeepCode 在同口径下的一次运行）。本文不用裸的"基线"。

## 2. 口径（两边一致，改一处必须两边同时改）

| 项 | 值 | 谁保证 |
| --- | --- | --- |
| 模型 | **09-19 起**：`deepseek-flash` @ api.deepseek.com（`deepcode_config` 的 `deepseek` 档，`DEEPCODE_CONNECTION=deepseek`），四臂同一 serving；**09-18 前**：`DeepSeek-V4-Flash` @ Paratera（`paratera` 档，`RESULTS-HISTORY` 里的数字）。规划与写码同一模型，无阶段覆盖；每次调用 `max_tokens` 32768（模板把它声明成带 `maxOutputTokens: 32768` 的手动模型条目） | 模板 + `run_trial.sh` 口径闸（`DEEPCODE_EXPECT_MODEL`） |
| 思考 | **开**，三臂一致（09-19 晚改定；09-18 前的数字是"关"）。DeepCode / 本线：`compat.thinking=enabled`（`thinking: {"type": "enabled"}`），回包 `reasoning_tokens > 0`（`run_trial.sh` 的 `DEEPCODE_EXPECT_THINKING=enabled` 闸）；Codex / Claude 桌面版：DeepSeek 缺省即开，什么都不用做，`audit_desktop.py` 从会话日志核每次调用的思考 token > 0。（关思考在桌面版做不到：DeepSeek 对 Codex 客户端无视 `effort=none`，Claude Code 的开关只是不发字段） | 补丁 + `run_trial.sh` 闸 + `audit_desktop.py` |
| 执行 | **命令随便跑，只有长时间的 CPU / GPU 训练或评估不行**（09-20 晚定；白天的"完全禁执行"和"逐条人工审批"两版都作废）。Code-Dev 判分本来就不执行；本线写码 agent 只有 write_file / search_code_references / read_paper、没有解释器，`PAPER2CODE_IMPLEMENT_VERIFY` 关（生成后不跑 pytest），生成后做一次 `compile()` 级语法检查 + ≤ 2 轮 `edit_file` 修复（`syntax_check.py`，不执行）；桌面臂正常全自动跑，只靠题面附注 Execution 一条告诉它"命令照常跑，只有长时间训练 / 评估不行、那些以后在远程跑"，不装规则、不设权限、不人工审批；`audit_desktop.py` 列出跑过的命令和耗时，单条 > 10 min（装依赖除外）或合计 > 60 min 作废（`CALIBER_BROKEN`），跑过但在线内的标 `CALIBER_REVIEW` 由 owner 扫一眼（09-19 的 fre / rice Codex 运行各跑了 177 min 实验，作废） | 本线 `syntax_check.py` + 题面 + `audit_desktop.py` |
| 输入 | **四臂同一份字节**：`paper.md`（PaperBench 的 Mathpix 式 OCR）+ `addendum.md` + `blacklist.txt`；不给 PDF、不给 assets、不给 rubric/config。DeepCode / DeepEvol 线把 addendum 并进 md 末尾；CLI 臂拿目录 + 官方题面（题面里 "in both PDF and markdown format" 改成 "in markdown format"，除路径外唯一改字）。选论文前 `check_paper_md.py` 核 md 是否缺章（robust-clip 的官方 md 缺 §2–§3，剔除） | `run_trial.sh` [2/3] / `render_prompt.sh` / `check_paper_md.py` |
| 黑名单 | `blacklist.txt` 在两层拦：git `insteadOf`（setup.sh）+ MCP 层 `DEEPCODE_URL_DENYLIST`（补丁） | setup.sh / run_trial.sh |
| 预算 | 参考挖掘 40 轮 / 下载 12 轮；挖掘报告 32768、下载 16384、预筛 32000、分析 16000、关系 16000 token；规划限时 600 s；stall 7200 s；写码墙钟 21600 s；14 h 硬顶 | `run_trial.sh` 注入（补丁只把这些做成 env，默认全等于上游） |
| 实验开关 | fix-①②③ **必须关**（§5.3） | `run_trial.sh` 拒绝 `=1` |
| 判分 | PaperBench Code-Dev `code_only=True`（裁判只判 Code Development 叶，不执行代码），裁判 `PB_JUDGE_MODEL`：09-17 起 `DeepSeek-V4-Flash`，结构化解析器恒为 `DeepSeek-V4-Pro`（bam 批是 Pro 裁判）；`PB_JUDGE_CONCURRENCY=20`，`num_invalid_leaf_nodes ≤ 2` 才有效；四臂同一个裁判同一天判 | `run_grade.sh` |

## 2.1 2026-09-20 批：三臂 × 20 篇（当前进行中）

| | |
| --- | --- |
| 问题 | 同一底座、同一输入、都不执行代码的条件下，Codex 桌面版、Claude 桌面版和我们的 DeepCode 线差多少——论文 Table 1 没做这件事（Codex 等用 Sonnet 4.5-thinking，DeepCode 用自己的配置，然后说 4.4×；bam 同底座对照下 Codex 是 0.73 不是 0.19） |
| 臂 | **Codex 桌面版、Claude 桌面版（Code 标签）**（分支 `0919-test` 的 `desktop_prep.sh` / `desktop_finish.sh`，owner 跑）、**DeepCode 臂 = DeepEvol Paper2Code 线**（另一个仓库，`--until compute` 的第 9 步树，`submit --dest-root`）。本仓库的 DeepCode 基线 `run_trial.sh` 不在本批里，只作历史对照 |
| 论文 | PaperBench 全部 20 篇（`data/papers/` 的 `paper.md` + `addendum.md` + `blacklist.txt`）；`robust-clip` 的官方 `paper.md` 缺 §2–§3（`check_paper_md.py`），照跑但结论里单独标 |
| 钉死的量 | 模型 `deepseek-flash` @ api.deepseek.com、**思考开**、**命令随便跑、只有长时间训练 / 评估不行（题面告知，事后审计）**、**上下文窗口三边都按 1M**（Codex `model_context_window`、DeepCode `contextWindow`、Claude 模型名带 `[1m]`）、输入同字节（md + addendum + blacklist，无 PDF/assets/rubric）、黑名单、裁判 Flash + Pro 解析器；每篇每臂 1 份；桌面臂题面 = 官方 `code_only_instructions.txt` + 官方附注 + 官方 `time_limit_template` 的"3 小时"句（官方语义：预期用满，除非核心贡献已复现完；不是硬上限，人不按表停，实际用时记 RUN_NOTES） |
| 起跑 | 桌面臂：每篇每臂 `desktop_prep.sh` → 人在 app 里粘题面 → `desktop_finish.sh`（含 `audit_desktop.py`），产物 `results/<paper>/<arm>/`。本线：DeepEvol 仓库 `scripts/paper2code_canary.py init … --thinking enabled` + `run --until compute --env-file ~/Documents/env/deepseek.env`（`DEEPCODE_PAPER_FIDELITY=1` 默认开），见那边 README「生成到第 9 步并摆卷」。判分时把各 `submission/` 拷进 `~/pb_submissions/<paper>/<arm>` |
| 判分 | 池子齐了按论文 `PAPER=<id> PB_JUDGE_MODEL=DeepSeek-V4-Flash bash deepcode_test/scripts/run_grade.sh`，数字回填 `RESULTS-HISTORY.md`（fre 第一对见 §1.3） |
| 读法 | 单篇噪声 0.025（sapg 同份重跑）、历史组内摆动 0.09–0.19：20 篇看方向和一致性，差值 < 0.03 的篇补一份；n < 5 不说"优于" |
| 局限 | 三臂都在无 GPU 的 Mac 上；桌面臂能做秒级检查（语法 / 导入 / 单测），本线对应加了 `compile()` 级语法检查（09-20 晚），但导入错误 / 属性不存在这类本线仍看不到；"短命令 vs 长实验"只靠题面约束 + 事后审计（10 min / 60 min 线），不靠机制拦；桌面版的会话审计依赖 app 本地日志格式 |

## 3. 快速开始（clone 即跑）

前置：Linux / macOS，`git` `curl` `uv` `node`+`npm`（Node ≥ 18）`patch`；判分时要 Docker。不需要 git-lfs。
一把 Paratera 的 OpenAI 兼容 key（复现用 `PARATERA_API_KEY`，裁判用 paperbench/.env 的 `OPENAI_API_KEY`）。

```bash
git clone git@github.com:2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
PAPERS=sapg bash setup.sh        # 校验 vendored 的 PaperBench（上游固定 commit + patch）、水合 sapg 资产、校验 DeepCode/、uv sync、
                                 # 生成 .deepcode-home/deepcode_config.json（口径）、设 git 封锁、建 ~/pb_submissions/sapg
```

key 只经环境变量进入：写一个文件（不进仓库），内容一行 `DEEPSEEK_API_KEY=...`（`deepseek` 档）或 `PARATERA_API_KEY=...`（`paratera` 档），然后：

```bash
PREFLIGHT_ONLY=1 PAPER=sapg ENV_FILE=~/my.env bash deepcode_test/scripts/run_trial.sh      # 免费自检，过口径闸
PAPER=sapg TRIAL=trial1 ENV_FILE=~/my.env nohup bash deepcode_test/scripts/run_trial.sh > run.log 2>&1 &
```

连接与模型由 `$DEEPCODE_HOME/deepcode_config.json` 决定（`setup.sh` 生成时按 `DEEPCODE_CONNECTION`（`paratera` | `deepseek`）与 `DEEPCODE_MODEL`；换档：`DEEPCODE_REGEN_CONFIG=1 DEEPCODE_CONNECTION=deepseek DEEPCODE_MODEL=deepseek-flash bash setup.sh`，跑时 `DEEPCODE_EXPECT_MODEL=deepseek-flash`；
`manualModels` 里 `DeepSeek-V4-Flash` 与 `DeepSeek-V4-Flash-Vision-Exp` 都是 32768 / 思考关）；`run_trial.sh` 的口径闸用
`DEEPCODE_EXPECT_MODEL` 核对。**S9 成对重跑（2026-09-18 起）**：基线与 DeepEvol 线同模型 `DeepSeek-V4-Flash-Vision-Exp`、同两处规划补丁都开——

```bash
DEEPCODE_EXPECT_MODEL=DeepSeek-V4-Flash-Vision-Exp DEEPCODE_PLANNING_FANOUT=1 DEEPCODE_PLANNER_CONTEXT_WINDOW=1000000 \
PAPER=sapg TRIAL=vexp1 ENV_FILE=~/my.env nohup bash deepcode_test/scripts/run_trial.sh > runs/sapg/console_vexp1.log 2>&1 &
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
`experiments/splits/<paper>.txt`（已登记：fre / rice / sequential-neural-score-estimation / bam / sapg / pinn / robust-clip / self-expansion /
test-time-model-adaptation / adaptive-pruning / stochastic-interpolants；改完重生成 `patches/paperbench_local_changes.patch` 并跑 `patches/verify_paperbench.sh`）。
**选新论文前**：`DeepCode/.venv/bin/python deepcode_test/scripts/check_paper_md.py <ids>`，`SUSPECT` 的（官方 md 缺章）不进对比。

跑自优化循环前先过泄漏闸：`bash deepcode_test/scripts/ci/check_no_rubric_leak.sh`（退出码 0 才可跑；扫描 DeepCode 提示词、补丁、裸跑固定件）。

## 4. 目录

```
.
├── README.md                    ← 本文件
├── setup.sh                     ← 一键环境（幂等）
├── DeepCode/                    ← HKUDS/DeepCode main 21ebc57f + patches/deepcode_local_changes.patch（完整副本；.venv / deepcode_lab 不入库）
├── patches/
│   ├── UPSTREAM_BASE.txt              两个上游仓库的固定 commit
│   ├── deepcode_local_changes.patch   DeepCode 全部改动（16 文件，+830/−70，§5）
│   ├── deepcode_patched.sha256        打过补丁的 16 个文件的 sha256
│   ├── verify_deepcode.sh             证明 DeepCode/ = 上游 + patch（setup.sh 自动跑）
│   ├── paperbench_local_changes.patch PaperBench 全部改动（5 文件，§5.4）
│   └── verify_paperbench.sh           证明 frontier-evals/ = 上游 + patch + 新增文件，数据文件哈希 = 上游 LFS oid（联网）
├── config/                      ← deepcode_config.template.json（口径）、credentials.example.json、paperbench.env.example（无密钥）
├── deepcode_test/
│   ├── scripts/                       run_trial.sh · run_batch.sh（09-19 批一条命令）· run_grade.sh · stage_b_driver.py · run_all_trials.sh · check_paper_md.py · paratera_key.sh
│   │   ├── gates/exec_level.py        执行级结构判据（确定性、零成本，给自优化循环当目标函数）
│   │   ├── ci/check_no_rubric_leak.sh 评分知识泄漏扫描
│   │   └── monitor/                   进度快照
│   └── bare/                          裸跑臂：README.md（跑法 + 为什么必须有代理）· run_bare.sh · render_prompt.sh · additional_notes.txt · continue_message.txt · paratera_proxy.py · bare_prompt_suffix.txt（bam 批历史）
├── paperbench_changes/          ← PaperBench 改动文件副本 + 新增（单篇 split、裁判偏差分析脚本）
├── docs/
│   ├── RESULTS-HISTORY.md             全部历史数字与结论（含作废标记）
│   ├── INPUT_STANDARD.md              三方输入标准（依据、三层规则、起跑前核验）
│   ├── PITFALLS.md                    踩坑总表（60 余条）
│   └── CODEDEV-ARMS.md                Code-Dev 口径一页：题面 vs 答案卡、四臂怎么接、为什么 vendored PaperBench（2026-09-19）
├── runs/                        ← 每轮的日志 / 输入 / 任务归档 / 摆卷副本 / 判分 JSON（gitignore）
├── .deepcode-home/              ← setup.sh 生成的 DeepCode 配置目录（gitignore；与你机器上其它 DeepCode 完全隔离）
├── .mcp-node/                   ← setup.sh 装的 filesystem MCP 服务器（gitignore）
└── frontier-evals/              ← PaperBench（openai/frontier-evals @ UPSTREAM_BASE.txt，project/paperbench + project/common）**vendored**，patch 已打；核验 patches/verify_paperbench.sh；.venv/ runs/ 仍 gitignore
```

`~/pb_submissions/<paper>/<trial>/` 是判分器硬性要求的提交池，根目录下每个子目录名必须是合法 paper id。

## 5. 对上游的改动

### 5.1 DeepCode（`patches/deepcode_local_changes.patch`，16 文件，+830/−70）

原则：每个改动 env 门控、默认值等于上游；`verify_deepcode.sh` 证明 `DeepCode/` 一个字节不多改。分四组：

**A. 带走子集**——与 DeepEvol 线 `apps/v2/agent_engine/paper2code/VENDOR.md` 第 2–7、10、11 条一一对应，两边引擎行为一致：

| 文件 | 改动 | 旋钮（默认 = 上游） |
| --- | --- | --- |
| `core/compat/agent.py` | `tool_filter` 按消毒后的前缀（`-`→`_`）匹配 | — |
| `tools/code_indexer.py` | 预筛 / 逐文件分析 / 关系抽取的 `max_tokens` env 化；**预筛止血（2026-09-17）**：输出只要 `file_path` + `confidence`（两个长文本字段下游从不读；263 文件的仓库曾把回复写到 11.8 万字符撞顶、解析失败、静默全量索引），两句作者残留的"推荐系统 / GNN / 扩散模型"改成中性表述，日志打实际选中数，`finish_reason=length` 视为失败走重试 | `DEEPCODE_PREFILTER_MAX_TOKENS` 2000 · `DEEPCODE_ANALYSIS_MAX_TOKENS` 1000 · `DEEPCODE_RELATIONSHIP_MAX_TOKENS` 1500 |
| `workflows/agent_orchestration_engine.py` | 下载 agent：工具优先的提示、只给 `git_clone`、一次纠正重试、空 `code_base` fail-fast；挖掘 / 下载的输出上限与迭代预算 env 化 | `DEEPCODE_REFERENCE_MAX_TOKENS` 8192 · `DEEPCODE_DOWNLOAD_MAX_TOKENS` 4096 · `DEEPCODE_REFERENCE_MAX_ITERATIONS` 8 · `DEEPCODE_DOWNLOAD_MAX_ITERATIONS` 8 |
| `workflows/agents/document_segmentation_agent.py` | 分段 agent 必须真的调工具；`document_index.json` 不存在即失败 | — |
| `workflows/code_implementation_workflow.py` | 写码墙钟与 stall 阈值 env 化；**写码单次输出上限 env 化（2026-09-18，VENDOR 12）**：上游写死 8192，一个 30 KB 文件把 `write_file` 的 JSON 截断、整轮写码中止 | `DEEPCODE_MAX_WALL_SECONDS` 7200 · `DEEPCODE_STALL_THRESHOLD` 不设 = 上游 300 · `DEEPCODE_IMPLEMENT_MAX_TOKENS` 不设 = 上游 8192（`run_trial.sh` 注入 32768） |
| `workflows/codebase_index_workflow.py` | f-string 里的反斜杠提到表达式外（3.11 兼容） | — |
| `workflows/agent_orchestration_engine.py`、`tools/document_segmentation_server.py` | **规划两处（2026-09-18，VENDOR 11，PLAN-3 第 7 / 7b 项）**：① 上游 `c9090c1a` 删掉的规划扇出搬回——Concept + Algorithm 两个分析 agent 先看论文，输出以 `# Worker outputs` 接在规划器消息后（与旧 `ParallelLLM` 一字不差），开关默认关；② 分段后规划器上下文原来写死 8 段 / 24 000 字符（sapg 56k 的论文规划器只看 24k，附录超参表从没进过规划器）——给了上下文窗口就按 (窗口 − max_tokens − 12k 提示预留) × 0.85 × 3 字符/token 定预算，整篇放得下全进、按原顺序，放不下才按上游相关度排序截断；分段器让参考文献之后的附录（`\section*{A. …}` / `# Appendix`）单独成段并给高 code_planning 相关度（这一条不带开关：分段产物本身变了，sapg 从 9 段变 10 段）。两个开关不设时规划路径与上游逐字节一致 | `DEEPCODE_PLANNING_FANOUT` 不设 = 关 · `DEEPCODE_PLANNER_CONTEXT_WINDOW` 不设 = 上游 8 段 / 24k（S9 成对重跑时基线与本线**都开**：`1` 与 `1000000`） |

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

`DeepCode/` 是上游 `21ebc57f` 的完整副本（含 desktop / tests / website，1,011 文件），只有补丁里的 16 个文件不同。
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
| `paperbench/nano/eval.py` | `paper_split` 允许单篇 split（fre / rice / sequential-neural-score-estimation / bam / sapg / pinn / robust-clip / self-expansion / test-time-model-adaptation / adaptive-pruning / stochastic-interpolants / lite） |
| `paperbench/utils.py` | `is_docker_running` 走 `docker.from_env()`，尊重 `DOCKER_HOST`（macOS Docker Desktop 的 socket 不在 /var/run） |

未改动裁判提示词与评分树。

**PaperBench 在本仓库里的形态（09-19 起）**：`frontier-evals/`（上游 `openai/frontier-evals @ UPSTREAM_BASE.txt` 的 `project/paperbench` + `project/common`）
以普通文件 vendored 入库，补丁已打，用过的论文的 LFS 资产已水合（其余仍是指针文本，`PAPERS=<id> bash setup.sh` 按需补）；`.venv/`、`runs/`、`nanoeval/records/` 不入库。
`patches/verify_paperbench.sh` 联网把上游拉到临时目录逐文件比：源码 = 上游 + patch + `paperbench_changes/` 的新增文件，数据文件要么相同、要么哈希等于上游 LFS 指针的 oid。
上游 `.gitattributes` 改名 `.gitattributes.upstream`，数据以真实字节存。这样文档能指着仓库里的文件和行号，同事 clone 即有裁判和题面。

## 6. 坑（先读这一节再开跑；全表 60 余条在 `docs/PITFALLS.md`）

| 现象 | 根因 | 本仓库怎么处理 |
| --- | --- | --- |
| "思考关"没关，70% 输出 token 是思考 | Paratera 忽略 `enable_thinking:false` | 只认 `thinking:{type:disabled}`（`compat.thinking`）；`run_trial.sh` 跑完汇总 `reasoning_tokens`，非 0 即口径失败 |
| 所有 agent 零工具空转、产物为空 | `deepcode init` 不写 `tools.mcpServers` | 模板带 7 个服务器，`python -m tools.xxx` 模块方式启动；filesystem / fetch 装成固定路径（`npx`/`uvx` 首次解析 20 s+ 会撞 MCP 连接超时，`uvx` 还会重编 cryptography）；口径闸检查齐全 |
| `filesystem` MCP 一连就 `Connection closed` | 服务器启动时校验允许目录存在，`deepcode_lab/` 还没建 | `setup.sh` 先 `mkdir -p DeepCode/deepcode_lab` |
| 配置写了 `maxTokens: 32768`，日志却是 `Resolved workflow LLM … max_tokens=8192`，长文件会被截断 | 21ebc57f 的执行档按模型目录的 `maxOutputTokens` 钳 `max_tokens`；手动目录里只写模型名字符串时，deepseek 家族缺省 8192 | 模板把模型写成对象 `{id, contextWindow, maxOutputTokens: 32768}`；口径闸核对。**sapg trial1（2026-09-17）是在 8192 下跑的**，见 RESULTS-HISTORY |
| 参考挖掘 / 下载 agent 8 轮就放弃、报告是 runner 的"到达上限"文本 | 上游 `max_iterations=8` | 40 / 12（两次真机 8 都不够） |
| 挖掘报告截断，下载侧只见 1 个仓库 | `maxTokens=4096/8192` | 32768 / 16384 |
| CodeRAG 预筛 JSON 截断 → 静默回退全量索引（8,885 文件仓库需 140 h；sapg trial2 的 263 文件仓库在 32000 下照样撞顶） | 每个候选文件一条带理由的长记录 | 32000 之外，补丁把输出改成路径 + 置信度（缩 4 倍）并把 `length` 当失败重试；分析/关系 16000 |
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
