# 论文复现 Agent 架构设计(v0.1)

> 状态:草案,2026-09-03。骨架取自评审均分最高的提案「DeepEvol-Repro v0 周末闭环」,吸收了六份评审的 borrow_for_synthesis,并修掉了它们指出的致命缺陷(主张选取不可判定、周末预算不闭合、假环境扫描误伤/漏检、宿主机执行冲突、WSL 内存超物理、开发容器与干净容器漂移、PaperBench 判分配置写错、留出集选篇泄题)。文中所有行号、路径、数字均在 2026-09-03 于本机核实;与前几份文档冲突处以本文为准,冲突点在 §8 列出。

---

> **生成说明**:本文由多 agent 工作流合成(2026-09-03):3 条线研读 AutoSOTA(评测逻辑 / 复现手法 / 结果与局限,共 35 条带证据的发现)、2 条线用本仓库证据审视用户五点计划、3 个角度独立出架构提案、每提案 2 位评审六维打分(均分 41.0 / 39.5 / 39.0),最后以最高分提案为骨架合成并修掉评审指出的致命缺陷。文中行号、硬件参数、归档路径已由维护者抽检(WSL 内存 7GB、RTX 4060 8GB、`code_implementation_workflow.py:78/90/989`、`code_implementation_server.py:683/774`、归档 ddaecdf1 `source=generated` 均属实)。

## 0. 一页摘要

**目标(本周末)**:在 rice(PaperBench `data/papers/rice`)的 MuJoCo Hopper 单环境子集上,让「论文 → 计划+代码 → 容器环境 → GPU 降配复现 → `outputs/metrics.json` + `reproduce.log` + `scores.jsonl`(iteration 0)」这条链的每一步都**以磁盘产物与退出码为判据**全绿一次。总费用上限 ¥120,总墙钟 周六 10h + 周日 8h。

**目标(下周)**:把周末手写的 manifest / env.lock / 闸门模板自动化;建立留出集与预注册;接 fre(老栈 + 数据预飞);A/B 比较「自建增量写码器 vs DeepCode fast 前端」;启用 PaperBench 从未被论文和我们用过的 Code Execution / Result Analysis 口径做外部对照。

**非目标**:
- 不追求 PaperBench Code-Development 裁判分「逼近论文倍数」——同名裁判换 serving 有 15.7% 叶级分歧(28/178),rice 倍数在 1.05× 与 2.58× 之间翻转,组内轮间波动(0.094/0.107/0.195)是组间差距(0.010/0.023)的 4.6~9 倍,该分数不能做目标函数。
- 不做自进化。前置条件(执行级目标函数、沙箱、留出集、固定 serving)全部就位前不启动。
- 不复现原规模(rice 论文 8×A100、fre 每域 12~24h 单卡);本周验收的是「降配跑通」,full 规模标记为租 GPU 的目标。
- 不接 Claude Code 壳,不用 Anthropic API;模型只走 OpenAI 兼容端点。**供应商决策(2026-09-03):只用 Paratera `https://llmapi.paratera.com/v1`,不再使用 SiliconFlow** —— 历史分数里的 SiliconFlow 数据保持不变作为存档。

**与 DeepCode 的关系**:DeepCode 冻结在 `e0767d0` + `deepcode_test/patches/deepcode_local_changes.patch`,只作**可选前端**(Phase 2–8:文档转换/分段/规划/参考仓库挖掘/下载/CodeRAG 索引),以 `task_dir` 文件合同对外供料;它的 Phase 9 写码循环整体废弃——索引模式工具面只有 `write_file` + `search_code_references`(`workflows/code_implementation_workflow.py:78` `_INDEXED_TOOL_NAMES`),`_MAX_ITERATIONS = 800` 常量(`:90`),`max_tokens=8192` 硬编码(`:989`),`execute_python/execute_bash` 默认 30s(`tools/code_implementation_server.py:683,774`),31 个归档、2,443 次工具调用里 `execute_*` 共 50 次,全部是 `mkdir`/`touch`/`find`/`ls`/`cat`/`grep`,**没有一次运行生成的代码**(无 python/pytest/import/pip)。周末**不跑 DeepCode**:代码种子直接用冻结产物 `deepcode_test/rice/submissions/trial2/`,语料用归档 `deepcode_test/rice/task_archives/archive_task_paper_ddaecdf1_0830_0724/`(`planning_result_meta.json` source=generated、5 个参考仓库、5 份 index 齐全)。DeepCode 被证实的唯一长处是覆盖面(参考仓库挖掘 + CodeRAG 使环境/数据集维度普遍占优),下周作为语料层接回。

**新建的三层**(DeepCode 从未有过):① 环境整备(env.lock + 容器 + import/reset 矩阵);② 自建 OpenAI 兼容 tool-calling 修复环(窄工具面、写后即跑、单场连续对话、失败签名账本、闸门驱动重访、done 不采信);③ 单机作业调度 + 执行级验证(harness 自有评估器 → import → env.reset → 冒烟训练 → 指标 schema → 论文主张方向)。

---

## 1. 对用户五点计划的逐条回应

| # | 计划项 | 结论 | 一句理由 | 本文修订后的做法 |
|---|---|---|---|---|
| 1a | 从各环节判分问题提取结构化通用问题去修 DeepCode 源码 | **部分合理** | 通用问题清单 P1–P8(`deepcode_test/docs/FINDING_generic_pipeline_failures.md`)已存在且与 fre/rice 无关,但失分是结构性的(两工具面、无执行、常量预算、clean-slate),补丁救不了 Phase 9 | 只做三处不改生成逻辑的补丁(§2.2 S1),Phase 9 整体由自建修复环替代;DeepCode 降为前端 |
| 1b | 目标设为「逼近论文效果」 | **不建议(改写)** | 论文效应量(1.95×)在我们的测量噪声之下,且论文只用 Code-Dev 叶(fre 306/437、rice 178/369),用户要的环境/能跑/结果恰在从未被用的 Code Execution(124/170)与 Result Analysis(7/13)叶 | 目标改写为执行级、确定性、论文自带的验收通过率(§6);Code-Dev 分只做离线双 serving 审计 |
| 1c | 不行再靠自进化 | **不建议(推迟)** | σ≈0.1 下检出 Δ=0.05 需每臂约 63 轮,每轮 3~6h、¥20 复现 + ¥38 判分;且 rice 2.58× 证明裁判分可被「写得具体」讨好——Goodhart | 从计划中删除;前置条件(§6.6)满足后目标函数只能是开发集执行级通过率 |
| 2 | 从判分细则提取每步达标校验标准 | **有风险(修订)** | fx1/fx2 因提示词一句「Graders assign separate credit」整体作废;两裁判分歧的 28 叶几乎全是实例化级条目,恰是 checklist 最想抄的那类 | 改为从 paper.md + addendum + 被引文献编译 `manifest.json`(§2.2 S0),判据形态对应 CE/RA 叶但来源不是 rubric;`rubric.json` 物理不进工作区,CI grep 禁词 |
| 3 | 用 DeepCode 输出的 environment_setup 在 GPU 上装依赖 | **部分合理(修订)** | 它是 YAML 块标量散文,从未被执行;实锤幻觉依赖(`exorl` egg,`code_base/exorl` 无 setup.py)、包名错(`metadrive` vs `metadrive-simulator`)、版本互斥(gym<0.26 + gymnasium、torch<2.0 + SB3 2)、违反 addendum(D4RL 89141a6 为 2024-11,要求 2024-06 前) | 当候选不当真值;三路来源按可信度合并(参考仓库自带 conda_env.yml/setup.py > addendum > LLM 文本)生成 `env.lock.yaml`,在容器里用真实 pip 做解析回路(§4.1) |
| 4 | 数据集下载限流 → 先确认可下载,否则用户上传 | **合理(补全)** | 获取物实际三类(数据集 / 环境资产 / 第三方仓库),失败模式不同;历史网络错误全在模型 API 侧,git clone 零失败,真正的失败是 agent 不调工具并静默报成功(D2–D5) | 清单驱动预飞(§4.2):HEAD/Range 探测 → 断点续传 → checksum/一个 batch 校验 → 落盘才标 available;失败生成 `UPLOAD_REQUEST.md`;成功只看磁盘产物 |
| 5 | 换 Paratera 重判旧分看偏差(已做) | **合理(固化为协议)** | 是本项目最重要的发现:同名裁判两家 serving 同一份代码 16% 叶级分歧,JudgeEval 上两者同等准确(acc 0.685/0.719)无法仲裁 | 每个分数带 serving 标签、裁判四元组冻结、对照默认双 serving、n<5 只说方向;自建 agent 每次 LLM 调用做完整性校验(§3.1) |
| — | 目标流水线第 4 步「按需调度 CPU/GPU」 | **部分合理** | Docker 29.6 已注册 nvidia runtime、`--gpus all` 可见 RTX 4060 8GB;但 WSL 当前 7GB RAM(`free -g` 实测),`.wslconfig`(`/mnt/c/Users/43519/.wslconfig`)只有 `[experimental]` 段无 memory 行 | 最小可行「一台机、一条队列、一 job 一容器」(§4.3);先把 `.wslconfig` 设 `memory=10GB swap=8GB`(物理 15.7GB,不能设 16GB) |

---

## 2. 总体架构

### 2.1 架构图

```mermaid
flowchart TD
    P["paper.md + addendum.md + blacklist.txt<br/>(rubric.json / judge/ 物理不拷)"] --> S0["S0 manifest.json<br/>实验清单(周末手写,下周 LLM 抽取)"]
    S0 --> G0{"manifest_check<br/>schema / claims 引用 / gpu_hours ≤ 4"}
    G0 --> S1["S1 code_v0 种子<br/>默认: 冻结产物 trial2<br/>可选: DeepCode fast 前端(后台)"]
    S1 --> G1{"static_scan<br/>file_structure ⊆ 磁盘 / 假环境审计项"}
    G1 --> S2["S2 env.lock.yaml → Dockerfile.paper → 镜像<br/>preflight_env.json"]
    S2 --> G2{"env_gate<br/>pip / import / gym.make+reset+step / cuda"}
    G2 -- 失败签名 --> S3
    G2 --> S3["S3 修复环 repro/agent/loop.py<br/>单场对话 · 写后即跑 · done 不采信"]
    S3 --> G3{"smoke.sh(干净容器)<br/>compile → import → --help → SCALE=smoke reproduce.sh<br/>→ metrics schema → env.__module__ 核验"}
    G3 -- gate 报告回灌 --> S3
    G3 --> LOCK["git tag _smoke_ok<br/>protected_paths.sha256"]
    S4["S4 acquire.py 预飞<br/>datasets / env assets / repos<br/>(rice 子集为空集)"] --> S5
    LOCK --> S5["S5 scheduler.py<br/>一 job 一容器 · GPU flock 互斥<br/>SCALE=smoke → scaled → full"]
    S5 --> G5{"产物判据<br/>exit 0 + metrics.json schema + reproduce.log"}
    G5 --> LEDGER["scores.jsonl iteration 0<br/>baseline_source=measured_by_us"]
    LEDGER --> S6["S6 verify.py → verdict.json<br/>claims pass/fail/na · guardrail · failure_category"]
    S6 -.离线.-> AUDIT["run_grade.sh 单裁判<br/>Code-Dev 只做审计"]
    S6 -.下周.-> IND["独立只读审计会话<br/>real / uncertain / invalid"]
    MON["monitor.py 外层薄监督<br/>停滞检测 · 预算 · 续跑重启"] -.读 agent_events.jsonl.-> S3
    CI["ci/check_no_rubric_leak.sh<br/>每次 commit 与 run 前"] -.-> S0
    CI -.-> S3
```

### 2.2 阶段表

目录约定:代码在 `/home/deepevol/deepevol/repro/`(Python 包 `repro`,子包 `agent/ gates/ schemas/ scripts/ docker/ ci/`);每次运行在 `~/repro_runs/<paper>/<run_id>/`;共享存储在 `~/repro_store/{datasets,models,repos,skills,images}/`(ext4,`/home` 883GB 可用,不放 `/mnt/c`)。

| 阶段 | 输入 | 输出 | 复用(DeepCode / PaperBench / AutoSOTA) | 闸门(fail-closed) | 预算规则 |
|---|---|---|---|---|---|
| **S0 实验清单 manifest** | `frontier-evals/project/paperbench/data/papers/<paper>/{paper.md, addendum.md, blacklist.txt}`,由 `repro/scripts/import_paper.sh` 白名单拷贝并 `assert` 不存在 `rubric.json`/`judge/`/`judge.addendum.md` | `manifest.json`(schema `repro/schemas/manifest.schema.json`):`paper_id / category ∈ {rl_online, rl_offline, supervised, generative, analysis} / scope / environments[{id, source, probe, needs_gpu}] / datasets[] / methods[{name, role∈main|baseline, source}] / metrics[{name, role∈primary|guardrail, direction, parse}] / experiments[{id, grid, entry, scale_knobs}] / claims[{id, statement, comparator, metric, expected_direction, paper_effect_sigma}] / constraints_immutable[] / env_ids_normalized{} / gpu_hours_estimate / failure_category:null` | DeepCode Phase 2–4 的 `tools/pdf_converter.py`、`document_segments/` 只作参考;`initial_plan.txt` 的 `implementation_components` 作**非权威第二意见**(须 `planning_result_meta.json.source=="generated"`) | `repro/gates/manifest_check.py`:schema 通过;每个 environment/dataset 有 source 与 probe;`claims ≥ 1` 且每条 claim 的 metric/method 在表内;**每条 claim 的 `paper_effect_sigma ≥ 2`**(论文自报效应量 ≥ 2σ,否则标 `not_testable_by_us`);blacklist 域名不出现在任何 source;`gpu_hours_estimate ≤ 4`(超出则 scope 缩到子集并记 `scope_reduced=true`);`addendum` 的 out-of-scope 条目必须映射到 `claims.scope=out_of_scope` | 周末:0 次 LLM、≤30 分钟人工,**只读 paper.md + addendum.md**,不读任何 DeepCode 产物 README(trial2 README 的实验编号与论文不一致,是评审揪出的泄漏点);下周:`repro/manifest/extract.py` 1 次调用、max_tokens 32768、超时 600s,失败显式非零退出,不许 coerce |
| **S1 代码种子 code_v0** | 默认:`deepcode_test/rice/submissions/trial2/`(含 `experiments/{train_agent,train_mask,refine,evaluate}.py`、`run_experiments.sh` 已有 `SEED/GPU/N_SEEDS` 变量);可选:DeepCode fast 前端 | `~/repro_runs/<paper>/<run_id>/code_v0/`(git init,tag `_seed`,记录 `seed_source ∈ {trial2, deepcode_fast, deepcode_full}`);可选前端产物 `DeepCode/deepcode_lab/tasks/paper_<hash>/{initial_plan.txt, planning_result_meta.json, generate_code/, logs/}` | 前端:`deepcode_test/scripts/stage_b_driver.py --fast`(`:19 FAST`,`:54 enable_indexing=not FAST`)→ `execute_multi_agent_research_pipeline`;env 注入抄 `deepcode_test/scripts/run_trial.sh:100-118`(`DEEPCODE_URL_DENYLIST`、`DEEPCODE_LLM_RETRY_MODE=persistent`、`DEEPCODE_CHAT_RETRY_DELAYS=10,30,60,180,300`、`DEEPCODE_PERSISTENT_MAX_DELAY=900`、`DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT=30`、`DEEPCODE_OPENAI_REQUEST_TIMEOUT_S=600`)。**三处补丁**(`deepcode_test/patches/weekend_min.patch`):(a) `workflows/code_implementation_workflow.py:989` 的 `max_tokens=8192` 改读 `agents.implementation.maxTokens`(评审已核:rice 写码相位 10~32% 调用被截在 8214~8230 token);(b) 删除 `workflows/agent_orchestration_engine.py:796-798` 与 `workflows/agents/memory_agent_concise.py:1684-1685` 的「Graders assign separate credit … forfeits those points」句;(c) `tools/code_implementation_server.py:683,774` 的 `timeout=30` **保持默认**(宿主机执行不放宽;真正的执行只在 S3 容器里发生),只加环境变量 `DEEPCODE_EXEC_TIMEOUT_S` 读取并 clamp ≤ 300 | 沿用 `run_trial.sh:171-197` 三闸:`planning_result_meta.json.source=="generated"` 且 `plan_validation.valid`;产物属本轮 `tasks/`;`paper.md` 前 4000 字含标题关键词。新增 `repro/gates/static_scan.py`:`initial_plan.txt` 的 `file_structure` 每个文件在磁盘存在且非 0 字节(缺失清单写入报告);**假环境审计项**(见 §3.2,只报告不判红) | 前端不在周末关键路径:若跑,`DEEPCODE_MAX_WALL_SECONDS=7200`、`DEEPCODE_STALL_THRESHOLD=1800`、脚本硬顶 `timeout 3h`,费用 ≤¥15/轮,最多 2 轮,周六夜后台跑;两轮都未过闸 → 保持 trial2 种子 |
| **S2 环境整备** | `code_v0/{requirements.txt, setup.py, README.md}` + `manifest.environments` + 参考仓库自带环境文件(`code_base/*/{conda_env.yml, setup.py}`)+ `initial_plan.txt` 的 `environment_setup` 段(仅提示) | `env.lock.yaml`(`python_version / base_image / torch_index_url / pip_packages(精确 pin) / apt_packages / git_deps / env_ids_normalized{Hopper-v3: Hopper-v4} / needs_gpu / env_vars{}`)、`Dockerfile.paper`(`FROM repro-base`)、镜像 `repro/<paper>:<lock_sha8>`、`preflight_env.json` | 基础镜像从本机已有 `pb-env:latest`(5.38GB,ubuntu24.04 + miniconda py3.12)派生,`conda create -n repro python=3.11`;Docker Hub 直连不可靠(本机 node 镜像经 `dockerproxy.net`),**备选** `deepevol-runtime:py311`(4.03GB)+ pip torch cu121 轮子(nvidia runtime 挂驱动库,无需 CUDA 基础镜像)。预烤「现代 RL 栈」:`torch==2.4.1+cu121 gymnasium==0.29.1 mujoco==2.3.7 stable-baselines3==2.3.2 'numpy<2'`——这是 trial2 requirements 上下限交集里唯一自洽的组合;**不用 gymnasium≥1.0**(已删除 mujoco-py 系 v2/v3 环境);老栈(d4rl/mujoco-py/gym 0.21)留下周单独 py3.9 镜像 | `repro/gates/env_gate.sh` 在容器内:`pip install -r requirements.lock` 退出 0 → `pip check` 零冲突 → `python -c 'import <每个顶层包>'` 全绿 → 对 `manifest.environments[].probe` 逐条执行(rice:`gym.make('Hopper-v4'); env.reset(); env.step(env.action_space.sample())`)→ `needs_gpu` 时 `torch.cuda.is_available()==True` 且 `torch.zeros(1).cuda()` → 写 `preflight_env.json`;任一失败把 stderr 尾 200 行作「失败签名」交给 S3 修 `env.lock`(不重规划) | `docker build ≤30 分钟`(`PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple`);整备修复 ≤6 轮 × 15 分钟;同签名连续 2 次即换策略;6 轮未过 → `failure_category=Setup Failed` 终止并出报告 |
| **S3 修复环** | `code_v0`(git)、镜像、`manifest.json`、`preflight_env.json`、gate 报告 | `code_v1`(每次闸门新绿即 `git commit`,消息含闸门名;tag `_smoke_ok`)、根目录 `reproduce.sh`(`SCALE=smoke\|scaled\|full`)、`outputs/metrics.json`、`notes/code_map.md`、`notes/decisions.md`、`failures.jsonl`、`amendments.jsonl`、`agent_events.jsonl`、`protected_paths.sha256` | **不复用** DeepCode Phase 9;周末用 `openai` SDK 自写 `repro/agent/loop.py`(~400 行),从 DeepCode 抄**规则**不抄**模块**:重试退避 `10/30/60/180/300s`、identical-error 上限 30、工具名 `[a-z_]` 消毒规则(`core/agent_runtime/tools/mcp.py:90 _sanitize_exposed_name` 思路)、上下文尾部保留压缩策略(`core/agent_runtime/compaction.py`);`tools/code_reference_indexer.py` 模块级 `from mcp.server.fastmcp import FastMCP`(`:31`),`search_code_references` 是 `@mcp.tool()` async(`:338-339`)——**复制**三个真纯函数 `load_index_files_from_directory(:69)` / `find_relevant_references_in_cache(:179)` / `format_reference_output(:243)` 到 `repro/agent/refs.py`,不 import 该模块 | `repro/gates/smoke.sh`(**在从镜像新起的干净容器内**执行,不在开发容器):(1) `python -m py_compile` 全部 `.py`;(2) 每个 `manifest.experiments[].entry` `--help` 退出 0;(3) `SCALE=smoke bash reproduce.sh` 退出 0 且 ≤10 分钟;(4) `outputs/metrics.json` 通过 `repro/schemas/metrics.schema.json` 且每个 experiment×env×method ≥1 条记录、每条带 `provenance{env_module, entry, commit}`;(5) **运行时真环境核验**:`provenance.env_module` 的顶层包 ∈ `manifest.environments[].source` 声明的第三方包(rice:`gymnasium`),不得来自 `submission/`;(6) 静态审计项(§3.2)输出报告。全绿 → tag `_smoke_ok` → 生成 `protected_paths.sha256`(`reproduce.sh`、`experiments/evaluate.py`、指标写出模块) | `max_iter = 15 + 5×len(experiments)×len(environments)`(rice 子集 = 20);墙钟 3h;费用 ≤¥20(按 `agent_events.jsonl` 的 usage 实时累计);**smoke 单次 ≤10 分钟且一场最多 12 次**(12×10 = 2h ≤ 3h,与迭代上限乘积闭合);同签名最多 3 次,第 3 次注入「换策略」,第 4 次标 `blocked` 并继续;LLM 单次 `max_tokens 32768`、超时 600s;空正文/`finish_reason=length`/JSON 未闭合一律显式错误重试,不回退;**开发容器里任何 pip install 必须回写 `env.lock` 并重烤镜像后才允许进 S5** |
| **S4 数据与资源获取** | `manifest.datasets[]` / `environments[source=asset\|repo]`(每项:`name, kind∈dataset\|env_asset\|repo, url, mirrors[], size_bytes, sha256, license, gated, dest, env_var, verify`) | `~/repro_store/{datasets,models,repos}/<name>/`、`acquire_report.json`(每项 `available\|failed\|needs_upload` + 原因)、失败时 `UPLOAD_REQUEST.md` | URL 来源复用 `code_base/exorl/download.sh`、`code_base/D4RL/d4rl/infos.py:68,73`;第三方仓库克隆经 `DEEPCODE_URL_DENYLIST` 等价拦截;**不复用** DeepCode 下载 agent(D2–D5:宣布意图不调工具、静默报成功) | `repro/acquire.py` 每项:HEAD/Range 0-0 探测 → 磁盘预算(单项 ≤ `REPRO_MAX_ITEM_GB=20`,总量 ≤ 可用一半)→ `wget -c`(本机无 aria2c;下周 `apt install aria2`)≤3 次 → sha256/文件数 → `verify` 命令(load 一个 batch、shape/条数)→ 通过才 `available`;失败 → `needs_upload` + 上传单;S5 前再核一次磁盘产物 | 单项超时 `max(600s, size/1MBps×2)`;429/503 退避 60/300/900s 三次后转 `needs_upload`;周末 rice 子集为空集,但接口与 schema 必须建好 |
| **S5 调度与复现** | `jobs/<paper>_<exp>.json`(`image, cmd, needs_gpu, timeout_s, mem_limit, scale, mounts{submission, /store:ro, outputs}, seed`) | `runs/<job_id>/{reproduce.log, reproduce.log.creation_time, outputs/metrics.json, exit_code, run_manifest.json(镜像 sha、GPU 型号、墙钟、scale、seed)}`;`submission/scores.jsonl` 追加 iteration 0(`baseline_source=measured_by_us, scale, eval_scope`) | PaperBench 执行合同 `frontier-evals/project/paperbench/paperbench/reproduce.py:39-59`:`bash -c 'cd <submission> && bash reproduce.sh 2>&1 \| tee reproduce.log'` + `date +%s > reproduce.log.creation_time`(`judge/base.py:47-49` 读这两个文件);AutoSOTA `tmp/autosota-ref/record_score.sh:64-138` 的 protected 哈希 + exit 9 逻辑改写为 `repro/record_score.sh` | `docker run --rm --gpus all --memory 8g --shm-size 2g --name <job_id>`,`flock /tmp/repro_gpu.lock` 互斥;完成判据 = 退出码 0 **且** `outputs/metrics.json` 存在且 schema 通过(以产物为准);`protected_paths.sha256` 不匹配 → 该 run `invalid`、不入 `scores.jsonl`(exit 9);超时 kill 进程组记 `Insufficient Resources` 并自动降 scale 一档重试一次;每次状态变更写 `jobs/state.jsonl`,重启进 recovery(`docker inspect` 判活、按产物推断终态) | `needs_gpu` 由静态规则(cuda/`--device`/torch/SB3)+ 60s CPU 冒烟决定;GPU 串行、CPU 并行 ≤2;smoke:`total_timesteps=10000, N_SEEDS=1`, 超时 30 分钟;scaled:`3e5` 步、1 seed、超时 2h;full:论文 `1e6` 步 × 5 seed,标为下周租 GPU;**注**:rice Hopper PPO 是 CPU-bound(MuJoCo 仿真 + 小 MLP),周末 GPU 步验证的是 `--gpus` 管道与 `torch.cuda` 可用,不是真实 GPU 负载 |
| **S6 验证与报告** | `manifest.json`、`preflight_env.json`、smoke/scaled 两次 run 的 `metrics.json`、`scores.jsonl`、gate 报告、`agent_events.jsonl` | `verdict.json`(`stages[]{name, status, gate_results}`, `metrics{env_ready, smoke_pass, claims_evaluable, claims_pass, wall_clock, cost_yuan}`, `failure_category`, `baseline_source`, `scale`, `blocked[]`, `judge_scores[]{score, serving, model, pb_commit, n_invalid_leaves}`, `independent_verdict: null(下周)`)+ `docs/runs/<run_id>.md` | 离线审计复用 `deepcode_test/scripts/run_grade.sh`(code_only)单裁判;下周 CE/RA 口径:`paperbench.reproduction.skip_reproduction=True` + `paperbench.judge.code_only=False`(`nano/task.py:224` `_should_reproduce = not skip_reproduction and not code_only`;若只去掉 code_only,PaperBench 会在无 torch/GPU 的 `pb-reproducer:latest` 里自己再跑一遍) | `repro/verify.py`:每条 claim 按 comparator 对 `metrics.json` 求值 → `pass\|fail\|na`;**smoke 与 1-seed scaled 一律 `na`**(`insufficient_scale` / `no_std`),只有 `N_SEEDS≥3` 且差异 > `2×合并 std` 才判 `pass`;guardrail 指标不得劣于 iteration 0 记录 5%;CI `repro/ci/check_no_rubric_leak.sh` grep `-iE 'grader\|rubric\|forfeit\|credit\|paperbench\|judge\|weight(ed)? score'`(不用裸 `weight`/`points`,会满屏误报) | 验证脚本 0 次 LLM;裁判审计每份 ≈¥38×2 serving,周末可选(上限 ¥80),`num_invalid_leaf_nodes ≤ 2` 否则作废 |

---

## 3. Agent 循环设计(S3 修复环)

### 3.1 运行时与工具面

- **位置**:`repro/agent/{loop.py, tools.py, refs.py, gates.py, memory.py, monitor.py}`;纯 `openai` SDK `chat.completions` + `tools`,不依赖 Claude Code、不依赖 Anthropic、不 import DeepCode `core/`(该栈带 mcp/anyio,`core/compat/agent.py` 注释记录过 CancelledError 毒化历史)。
- **端点与模型**:`REPRO_BASE_URL / REPRO_API_KEY / REPRO_MODEL`,密钥只从 `~/.repro.env` 读入环境变量,仓库零明文。周末:主力 `DeepSeek-V4-Pro`。**注意(2026-09-03 换 key 后核实)**:原计划的 `Kimi-K2.7-Code` **在 Paratera 上不存在**(该模型此前走 SiliconFlow,已停用);Paratera 只有 `Kimi-K2.5/K2.6/K3`,与我们测过的不是同一个,不能直接顶替。若要保留「快模型主力 + 慢模型升级」的两档策略,需先用 §7.1 步骤 0e 的探针在 `Kimi-K2.6` 或 `Qwen3-Coder-Plus` 上验 tool-calling 与速度,否则周末就单模型 V4-Pro 跑,不设升级档;温度 0.2;`max_tokens 32768`;请求超时 600s;退避 `10/30/60/180/300s`;连续同错上限 30。
- **响应完整性校验(每次调用,四处静默降级的共同根因)**:`finish_reason ∈ {stop, tool_calls}`;content 为空且无 tool_calls 计一次「空响应」;tool_calls 的 JSON 参数必须闭合可解析;输出长度贴近 max_tokens 视为截断。任一不满足 = 显式错误进入重试,不允许「当正常继续」;连续 2 轮无工具调用即升级模型;连续 6 次空响应 → 压缩上下文重发。
- **工具面(名字只含 `[a-z_]`,规避 Kimi 对含连字符工具名静默不调用——`docs/CLEAN_E2E_PLAN.md` D5)**:

| 工具 | 说明 |
|---|---|
| `read_file(path, start?, end?)` / `list_tree(path, depth≤3)` / `grep_code(pattern, glob)` | 只读 |
| `write_file(path, content)` / `apply_patch(path, old_str, new_str)` | `apply_patch` 强制唯一匹配,失败回报原文片段;写 `.py` 后 harness 自动 `py_compile`(路径 = workspace 根绝对拼接,不 glob,规避 `REVIEW_local_changes_2026-09-03.md` 🔴-4) |
| `run(cmd, timeout_s≤1800)` | `docker exec -w /home/submission -e SCALE -e CUDA_VISIBLE_DEVICES repro_<paper> bash -lc '<cmd>'`;stdout/stderr 各截尾 8000 字符,全文落 `tool_logs/<n>.log`;命令串含 blacklist URL 直接拒绝(只扫 run/clone 类工具,不扫 `write_file` 内容,规避 🟠-3) |
| `run_gate(name)` | 执行 `repro/gates/` 对应闸门,返回结构化 JSON |
| `propose_manifest_amendment(kind∈add\|downgrade, item, reason)` | 写 `amendments.jsonl`;`add` 自动通过;`downgrade` 只允许 `role=baseline` 且必须给理由,**删除/降级 `role=main` 一律拒绝** |
| `record_decision(text)` / `done(summary)` | `done` 不采信,见 §3.3 |
| `search_refs(target_file, keywords, max_results=10)`(下周) | 包 `repro/agent/refs.py`(复制自 `code_reference_indexer.py:69/179/243`),`indexes_path` 由 harness 固定为 `task_dir/indexes/`,agent 不可改 |

- **执行容器**:`docker run -d --gpus all --memory 8g --shm-size 2g -v <code_v1>:/home/submission -v ~/repro_store:/store:ro --name repro_<paper> repro/<paper>:<sha> sleep infinity`,agent **从不在宿主机执行**;容器内 `git config --global url."https://invalid/".insteadOf` 逐条封锁黑名单(`run_trial.sh:103-105` 自记「insteadOf 挡不住 HTTP 抓取」,下周加容器 DNS 级拒绝)。
- **所有工具结果**为结构化 JSON,连同 usage 写入 `agent_events.jsonl`。

### 3.2 执行验证(写后即跑,done 不采信)

1. `write_file`/`apply_patch` 成功后,循环自动追加一条工具结果:对 `.py` 执行 `python -m py_compile`;若该文件是 `manifest.experiments[].entry` 则再跑 `<entry> --help`(超时 120s);失败信息作为下一条 user 消息回灌,模型不能跳过。
2. agent 调用 `done` 时循环不退出,而是自动执行 `repro/gates/smoke.sh`(**从镜像新起干净容器**),把 gate 报告(失败步骤名、stderr 尾 200 行、归一化签名、剩余尝试次数、已试修法列表)作为下一条 user 消息;只有全绿才真正退出。成功判据永远来自磁盘产物与退出码。
3. **闸门阶梯固定**:compile → import → `--help` → `SCALE=smoke reproduce.sh` → `metrics.json` schema → 运行时真环境核验 → 静态审计项。每轮的目标就是「让当前第一个失败的闸通过」,所以修复天然增量、多趟。
4. **假环境检测(修正后的三层,取代原提案的正则)**:
   - **硬闸(运行时)**:`metrics.json` 每条记录的 `provenance.env_module` 顶层包 ∈ `manifest.environments[].source`(rice 须为 `gymnasium`),不得来自 `submission/`;harness 自有 `repro/gates/eval_runner.py` 用 manifest 声明的真环境包独立加载 agent 的 checkpoint 回放算主指标,回放值为准、agent 自报值只作对照。这是唯一能让 `claims.pass` 站住的判据。
   - **审计项(静态,只报告)**:submission 自有文件里顶层类名匹配 `(Simulat|Simplif|Mock|Fake|Placeholder)`(rice trial2 `envs/malware_env.py:83 class MalConvSimulator`、trial3 `experiments/cage2/env.py:57 class SimulatedCage2Env`、`experiments/malware/env.py:59 class SimulatedMalwareEnv` **全是顶层类**,原提案的「except ImportError 分支内」规则一个都抓不到);继承 `gym.Env`/`gymnasium.Env` 但 `reset/step` 不调用任何 manifest 声明的第三方包;**白名单** `DummyVecEnv`/`SubprocVecEnv` 等库名(trial2 有 6 个文件用 `DummyVecEnv`);禁词表**去掉** `simulated`(MuJoCo 就是 simulator)与 `fallback`(`rice/mask_network.py:381` 的「fallback chain」是合法注释)。
   - 审计项命中不自动判红(否则 agent 学到的是改名不是实现),而是进入 `verdict.blocked[]`/`audit_items[]`,由 S6 与下周的独立审计会话处理。
5. `metrics.json` 用结构化 `provenance` 字段替代 `reproduce.log` 的 grep 禁词;`reproduce.log` 只保留 `NotImplementedError|Traceback` 两个硬禁词。

### 3.3 内存与上下文治理

- **单场连续对话**,不做 DeepCode 的 clean-slate(V4-Pro 实测每文件冷启动 6.8 分钟)。
- 上下文超过 `REPRO_CTX_BUDGET`(默认 200k tokens,按 `usage.prompt_tokens` 估)时,把最旧的工具轮压成一句摘要(文件路径 + 退出码),保留 system、manifest、最近 12 轮;压缩后重新注入 `notes/code_map.md` 与 `notes/decisions.md`。
- **外置记忆三文件**(对应 AutoSOTA `code_analysis.md / idea_library.md / target.md`):`notes/code_map.md`(首轮必须先写:入口脚本、评测命令、指标解析方式、本论文不可变约束清单——即 R7 列表,rice 含「Hopper-v3→v4 漂移已知,eval_scope 与论文不同」);`notes/decisions.md`(修复候选账本:类型/风险/结果);`manifest.json`(结构化目标)。
- **失败签名账本** `failures.jsonl`:`memory.normalize(traceback)` 去路径/行号/哈希/版本号后取最后异常类型 + 末 3 帧作签名,记 `{signature, attempt, fix_summary, outcome}`;下一轮提示自动注入「该签名已尝试的修法(禁止重复)」——AutoSOTA AgentFix 的 retrieval-before-repair。
- **跨论文技能笔记** `~/repro_store/skills/{pip_failures, env_id_migration, cuda_torch_matrix, mujoco_legacy, sb3_gymnasium}.md`,由 `docs/PAPERBENCH_RUNBOOK.md` 13 坑 + D1–D5 + Hopper-v3→v4 + mujoco-py 老栈手写做种;签名匹配关键词时把对应段落塞进提示。

### 3.4 重访与停滞

- **闸门驱动重访**(每轮):见 §3.2 阶梯。
- **R1 覆盖重访**(smoke 首次通过后):harness 把 `manifest.experiments[].grid`(method×env)与 `metrics.json` 对表,列出未覆盖项,agent 对每项选择「实现」或 `propose_manifest_amendment(downgrade)` + 理由;不给任何评分权重信息。
- **R2 约束自审**(smoke 全绿后):agent 对照 `notes/code_map.md` 的不可变约束清单再做一遍自审(有没有改评测/切分/伪造输出),修完再跑 smoke.sh;之后 `protected_paths.sha256` 落锁。
- **停滞规则**:同签名第 3 次注入「换策略」系统提示(允许:pin 版本、换 env_id、用 SB3 默认策略替代自写网络、缩小 scope);第 4 次把组件写入 `blocked[]` 并继续;连续 5 轮无任何 write/patch 或 10 轮 gate 无进展 → monitor 终止本环。
- **monitor.py(外层薄进程,周末先做同进程守卫,下周独立进程)**:读 `agent_events.jsonl`,做阶段推断(installing/patching/running/stuck)、预算(迭代、墙钟、费用)与进程组清理;若模型在预算未尽时提前 `done` 且 gate 未过 → 带「上一轮 gate 报告 + notes 摘要」重启对话(AutoSOTA 续跑监督器:模型自行结束不等于预算用完)。动作只有五种:`continue / inject_guidance(写 steer.md,agent 下轮开头必读后删除) / switch_model / rollback(到上一绿 commit) / terminate`。

### 3.5 取舍规则与禁令

- **允许**:修工程路径、补 glue、改入口脚本、pin 版本、换 env_id(记入 `env_ids_normalized`)、用库默认策略网络替代自写网络(addendum 明说通用策略即可)。
- **禁止**:改评测逻辑、数据划分、指标定义、`reproduce.sh` 的度量部分;用假环境/模拟器替代真实环境;硬编码结果;删除或降级 `role=main` 的方法。
- **系统提示只含**:角色、manifest 摘要、工具约定、闸门顺序、协议保持规则、`reproduce.sh` 合同(根目录、`SCALE=smoke|scaled|full` 三档、输出 `outputs/metrics.json` 含 provenance)。**绝不含** rubric、权重、裁判、评分等词;`repro/ci/check_no_rubric_leak.sh` 在每次提交与每次 run 前执行;设计 manifest 与提示词期间不打开任何 `rubric.json`(在 `docs/prereg/` 签字)。
- **预算随规模缩放**:`max_iter = 15 + 5×len(experiments)×len(environments)`;墙钟 3h;费用 ¥20/环;smoke ≤10 分钟 × ≤12 次。

---

## 4. 环境、数据集与资源调度

### 4.1 环境整备(S2)

- **三路来源按可信度合并**生成 `env.lock.yaml`:参考仓库自带环境文件(`code_base/exorl/conda_env.yml` 写明 python3.8 + cudatoolkit 11.1、`code_base/D4RL/setup.py:62-71`)> addendum 版本/日期约束(fre:D4RL 用 2024-06 前 commit;当前克隆疑为 2024-11 的 commit(归档无 .git,待核),需按 addendum 钉到 2024-06 前)> LLM 写的 `environment_setup` / `requirements.txt`(trial1 同时要 `torch<2.0, gym<0.26, gymnasium>=0.26, mujoco, mujoco-py, d4rl>=1.1`;trial3 `requirements.txt` 0 字节;trial2 `setup.py` 与 `requirements.txt` 不一致)。
- **栈分层不混装**:现代 RL 栈 py3.11 + torch 2.4.1 cu121 + gymnasium 0.29.1 + mujoco 2.3.7 + SB3 2.3.2 + numpy<2;老栈(d4rl / mujoco-py / gym 0.21,fre 用)单独 py3.9 镜像 `repro-legacy`(mujoco210 二进制 + gcc + patchelf);非 RL 论文(留出集候选中的 sbibm / GPT-2 探针)下周预留通用 py311 镜像。PyPI 现役版本约束(2026-09-03 查询):mujoco 3.12 要 py≥3.10、gymnasium 1.3 / SB3 2.9 要 ≥3.10、tianshou 2.0 要 ≥3.11、metadrive-simulator 0.4.3 要 <3.12、d4rl 1.1(2022-09)依赖 mujoco_py + gym<0.24。
- **解析回路**:容器内真实 `pip install` 失败 → stderr 尾 200 行归一化为签名 → 回灌 LLM 改 lock(≤6 轮 × 15 分钟)→ 重烤镜像;先查 `skills/pip_failures.md` 再自由推理。
- **preflight 矩阵**写入 `preflight_env.json`:python 版本 == lock、`pip check`、每包 import、每个 environment 的 probe(`make/reset/step`)、每个 dataset 取一个 batch 的 shape、`torch.cuda`。
- **开发容器与干净容器不漂移**:修复环容器允许 agent `pip install`,但任何依赖变更必须 `propose_manifest_amendment(add)` 并回写 `env.lock` → 重烤镜像 → smoke.sh 的最终通过必须在干净容器里;S5 永远从镜像起新容器;`run_manifest.json` 记镜像 sha 与 lock sha。

### 4.2 数据集与资源预飞(S4)

- **获取物三类**:① 数据集(fre:D4RL hdf5——官方源 `rail.eecs.berkeley.edu` 2026-09-03 探测 502,且 `d4rl/offline_env.py:12-37` 在 `env.get_dataset()` 首次调用才下载,必须在预飞暴露;ExORL 单域 `rnd.zip` 2.59GB/2.26GB,`dl.fbaipublicfiles.com` 支持 Range);② 环境资产(MuJoCo 由 pip `mujoco` 自带;MetaDrive 首次 import 拉资产;老栈 mujoco-py 需 `~/.mujoco/mujoco210`);③ 第三方仓库(rice 计划 5 个 clone:StateMask、JSRL、selfish-mining、CAGE2、malware_rl;均非黑名单,黑名单封 `chengzelei/RICE`、`kvfrans/fre`)。
- **`repro/acquire.py` 流程**:HEAD 或 Range 0-0 探测(状态码、content-length)→ 磁盘预算 → `wget -c` ≤3 次(下周装 `aria2c -c -x4`)→ sha256/文件数 → `verify` 命令(load 一个 batch、shape/条数)→ 通过才 `available`;失败 → `needs_upload` + `UPLOAD_REQUEST.md`(期望路径 `~/repro_store/datasets/<name>/`、格式、校验命令);用户上传后重跑同一 verify,agent 生成格式适配脚本(trial1 期待 `{domain}_{task}.npz` 而官方 zip 是 `buffer/` 逐 episode npz)。
- **路径契约**:容器 `-v ~/repro_store:/store:ro`,`REPRO_DATA_ROOT=/store`;`D4RL_DATASET_DIR`、`EXORL_DATA_DIR`(trial1)/`EXORL_DATA_PATH`(trial5)标准化为 `env.lock.env_vars` 声明;代码禁止硬编码绝对路径(静态审计项加一条 grep)。
- **镜像候选** `repro/acquire_mirrors.yaml`:`hf-mirror.com`、清华 pypi、HF 上的 D4RL/Minari 镜像(格式需转换,下周验证)。
- 周末 rice 子集为空集,整体跳过;fre 留到 D4RL 源恢复或镜像就绪之后。

### 4.3 算力调度(S5)

- **硬件前置**:物理内存 15.7GB,WSL 当前 total 7GB / available 4GB;`.wslconfig` 只有 `[experimental]` 段 → 加 `[wsl2] memory=10GB swap=8GB` 后 `wsl --shutdown`(**不能设 16GB**,会把 Windows 挤死);`docker run --memory 8g --shm-size 2g`;RTX 4060 Laptop 8GB 只跑 MuJoCo/SB3 级别,MetaDrive/CAGE2/malware 下周再排。
- **`repro/scheduler.py` 最小版**:`jobs/<id>.json` → 队列 `jobs/state.jsonl`(每次状态变更立即落盘:stage、容器名、PID、start_ts)→ 执行器 `docker run --rm --gpus all ...`;GPU 作业 `flock /tmp/repro_gpu.lock` 互斥,CPU 作业并行 ≤2;超时 kill 进程组;完成判据 = 产物文件存在且 schema 通过;重启进入 recovery:扫 `state.jsonl`,`docker inspect` 判活,按 `outputs/` 推断终态。周末只做「串行 docker run + timeout + state.jsonl」,队列/恢复下周。
- **降配口径**:`reproduce.sh` 读 `SCALE`(`smoke`:1e4 步、1 seed;`scaled`:3e5 步(rice Table 4 的 3×10^5)、1 seed;`full`:1e6 步 × 5 seed)映射到 `--total-timesteps/--n-seeds`;trial2 `experiments/run_experiments.sh:36-42` 已有 `SEED/GPU/N_SEEDS` 变量作模板。「降配跑通」与「复现原规模」是两个验收目标,后者按 `gpu_hours_estimate` 超阈值标 `Insufficient Resources` 或租卡。
- **PaperBench 执行合同**:submission 根目录 `reproduce.sh`,`reproduce.log` + `reproduce.log.creation_time` 三件套,使产物日后可直接进 CE/RA 叶判分。

---

## 5. 从 AutoSOTA 融入的元素

| # | 元素 | 来源 | 采纳? | 理由 / 我们的实现 |
|---|---|---|---|---|
| 1 | iteration 0 = 实测基线,不拿论文数字当基线 | 论文 §3.8.1 Phase 0(PDF L990-995) | **采纳** | `scores.jsonl` iteration 0 带 `baseline_source ∈ {measured_by_us, paper_reported, mixed}`、`scale`、`eval_scope`;论文报告值单独存 `manifest.claims`;AutoSOTA 自己的 leaderboard 混用三种基线(SAVVY 用论文值、FastGS 单场景 vs 论文多场景平均),我们按 `baseline_source` 分层报 |
| 2 | protected_paths SHA256 锁 → exit 9 不入账 | `tmp/autosota-ref/record_score.sh:64-138`(`:67` 注释「Mismatch → exit 9」、`:100/:105` sha256、`:127` PROTOCOL VIOLATION);cli_guide「切断奖励信号是唯一管用的」 | **采纳(改写)** | `repro/record_score.sh` 在 `_smoke_ok` 后锁 `reproduce.sh`、`experiments/evaluate.py`、指标写出模块;S5 前校验。**局限**:AutoSOTA 锁的是既有仓库的评测脚本,我们锁的是 agent 自己写的——所以指标聚合由 harness 的 `eval_runner.py` 计算而非 agent 代码,锁只防「基线后再改」 |
| 3 | 红线 R1–R7 | 论文 §3.9(L1089-1123) | **采纳(改写为协议保持规则)** | 写进 S3 系统提示 + 要求首轮在 `notes/code_map.md` 列本论文特定不可变约束(R7);R2/R3/R5/R6 由 §3.2 硬闸与 `protected_paths` 代码层执行,不只靠提示 |
| 4 | AgentFix retrieval-before-repair + 失败签名记忆 + 防振荡 | 论文 §2.6 | **采纳** | `failures.jsonl` + `~/repro_store/skills/*.md`;同签名禁止重复修法 |
| 5 | AgentMonitor 外层薄监督 + 续跑监督器 | 论文 §2.5;cli_guide v0.3.0「flash 约 3 轮、pro 约 14 轮就停」 | **采纳** | `repro/agent/monitor.py`,五种粗粒度动作;done 不采信 |
| 6 | 外置记忆三文件(code_analysis / idea_library / research_report)+ 一次性蒸馏 + 按需 grep | 论文 §2.5.2 | **采纳(映射)** | `notes/code_map.md` / `notes/decisions.md` / `manifest.json`;`research_report.md` 对应下周接回的 DeepCode `reference.txt` |
| 7 | 结构化目标(`target.md`:primary/guardrail/direction/eval_command/eval_output_format) | cli_guide `config.yaml`(L729-752) | **采纳** | `manifest.metrics[role]` + `reproduce.sh` 合同 + `metrics.schema.json`;非 primary 一律 guardrail,容忍带 −5% 或 −1·std |
| 8 | steer.md 人→agent 轮间指令注入 | cli_guide `autosota steer` | **采纳** | `inject_guidance` 动作;与硬约束冲突则记 `decisions.md` 并跳过 |
| 9 | AgentScheduler 落盘状态机 + 恢复模式 + 以产物判完成 | 论文 §2.8.2 | **采纳(单 GPU 版)** | `jobs/state.jsonl` + recovery;两 GPU 一单元改为单 GPU `flock` 互斥 |
| 10 | 失败分类枚举 | **CVPR 2026 仪表盘 `cvpr-data.js` 的 `AutoSOTA_Category` 字段**(Missing Repo 2268 / Incomplete Repo 689 / Non-Method Paper 369 / Missing Data 233 / Setup Failed 179 / Insufficient Resources 173 / No Improvement 66 / Succeeded 44 / Missed Claims 23);**论文正文与 cli_guide 全文 grep 不到此枚举** | **采纳(注明出处)** | `verdict.failure_category` 用这九类 + `stage ∈ {manifest, frontend, env, repair, acquire, reproduce, verify}`;漏斗带分母(尝试数、各阶段淘汰数) |
| 11 | 入选门槛「单次完整复现 ≤4 GPU 小时」 | 论文 §4.1(L1168-1185) | **采纳(改为分流)** | `manifest.gpu_hours_estimate ≤ 4` 否则缩子集,不拒绝 |
| 12 | 多指标护栏 R4,每轮上报全部指标 | 论文 R4(L1105-1108);cli_guide 结构化目标 | **采纳** | `verify.py` 对护栏设显式容忍带;报表标「主升/护栏平/护栏退」 |
| 13 | 独立只读评估会话 → `evaluation_verdict.json` real/uncertain/invalid | cli_guide v0.3.0(L929-937) | **采纳(下周)** | `repro/verify/independent.py`:未参与修复的模型读 `_seed.._smoke_ok` diff + `scores.jsonl`;LLM 判断项由 Paratera `DeepSeek-V4-Pro` 单裁判给出;判「不确定」时记 uncertain(原设想的双 serving 交叉验证已取消);周末 `verdict.independent_verdict=null` 预留 |
| 14 | 取消「达标即停」早停 | cli_guide v0.3.0 移除 `target_improvement_pct` | **采纳** | 停止条件只由预算与闸门决定;best-iterate 与 mean-of-reruns 分列 |
| 15 | 多 seed 噪声阈值 | 论文仅 R1 禁 best-of-N;per-paper 报告「best iterate vs mean eval 分歧」 | **采纳并加严** | AutoSOTA 把任何正变化记为 SOTA(CVPR 44 篇 median 0.11%、ICML 80 篇 <1%);我们 `N_SEEDS≥3`、claim pass 需差异 > 2×合并 std |
| 16 | 树状 rubric 权重守恒 BFS | 论文 §2.3 | **不采纳(现在)** | 他们自认只用了 Result Match 一维;我们的 manifest 是平面清单 + 执行级闸门,先跑通再谈分层评分 |
| 17 | idea library / Leap Path / Honeymoon | 论文 §2.7、§3.8.1 | **不采纳** | 无优化阶段;修复候选账本 `decisions.md` 只借其「类型/风险/结果」字段 |
| 18 | Claude Code 子进程作执行体;Anthropic 兼容端点 | cli_guide 流程框图、端点表 | **不采纳(硬约束)** | 用户不能接 Claude Code 壳;我们用 OpenAI 兼容 tool-calling 循环 |
| 19 | deep research 模型 + 引用数验证联网 | cli_guide(o4-mini-deep-research,VERIFIED/UNVERIFIED) | **不采纳(现在)** | 无调研阶段;「以引用/克隆成功数验证真的干了」的思路用于 S1 前端闸(`reference.txt` rank ≥3、`github_download.txt` 成功数) |
| 20 | Repo2Run 自动建环境 + Docker 手动兜底 | 论文 §2.4 | **部分采纳** | 未开源,无法接;兜底链思路落为 §4.1 的三路来源 + 解析回路 + 栈分层 |
| 21 | CLI 版不用 Docker、本机 repo_path 直跑 | cli_guide | **不采纳** | 与「agent 从不在宿主机执行」冲突 |

---

## 6. 评测协议

### 6.1 集合划分(冻结在 `frontier-evals/project/paperbench/experiments/splits/`)
- **开发集** = fre、rice(已被反复看过,只用于调试流水线与回归;其 manifest 已被作者对评分树的记忆污染,不算干净)。
- **留出集** = 从本地 `data/papers/` 其余 21 篇(本地共 23 篇:adaptive-pruning、all-in-one、bam、bbox、bridging-data-gaps、ftrl、lbcs、lca-on-the-line、mechanistic-understanding、pinn、robust-clip、sample-specific-masks、sapg、self-composing-policies、self-expansion、semantic-self-consistency、sequential-neural-score-estimation、stay-on-topic-with-classifier-free-guidance、stochastic-interpolants、test-time-model-adaptation、what-will-my-model-forget)中,**只读 paper.md + addendum.md** 做 S0 分诊(非 gated 数据、CPU 可冒烟、`gpu_hours_estimate ≤ 4`、8GB VRAM 可跑)选 3 篇,写入 `holdout.txt` 并 git 提交后冻结。**选篇依据不得使用 rubric 叶级计数**(前一版提案用 CD/CE/RA 计数选篇,已被评审判为泄题);留出集的 `rubric.json` 用 `git sparse-checkout` 或 `update-index --assume-unchanged` 移出工作树(它们在 git 跟踪中,不能直接 rm)。
- **外部对照臂** = Claude Code 壳 + 同底座裸跑(`deepcode_test/docs/CC_FRE_PROMPT.txt` 与 rice `PROMPT.txt` 逐字任务书),只作参考臂。

### 6.2 预注册
每次对照实验前写 `docs/prereg/<date>_<name>.md`:假设、臂、样本数、主指标、成功阈值、裁判 serving、停止规则;先提交再跑;跑后不改。

### 6.3 主指标(执行级,零裁判噪声,从 `verdict.json`/`scores.jsonl` 程序化统计)
`env_ready` 率、`smoke_pass` 率、`resources_verified` 比例、coverage(manifest grid 单元有 metrics 行的比例)、`claims_evaluable` 率、`claims_pass` 率(仅 full 且 ≥3 seed)、`audit_verdict==real` 比例、按九类 `failure_category` 分层的漏斗、墙钟、费用。报「成功率 = 成功/尝试」,报 median + IQR 与超过噪声阈值的篇数,不报均值。

### 6.4 次指标(裁判审计,不做目标)
- PaperBench Code-Dev `code_only`,裁判固定为「模型名 + serving + temperature + PB commit + `PB_STRUCTURED_PARSER_MODEL`」四元组;grade 文件名带 serving 与模型前缀。
- **单裁判(2026-09-03 定案)**:原计划的双 serving 对照**取消**。理由不是它没价值,而是它服务的问题已经变了 ——
  双裁判是为「DeepCode 到底有没有增益」服务的,那个问题里裁判分就是测量本身;
  本项目的主指标是执行级(§6.3),裁判分只做事后审计,给一个不做目标的指标付双倍钱不划算。
  **固定为 Paratera `DeepSeek-V4-Pro`**,唯一理由是**与已有基线可比** —— 我们手上 11 份提交的
  Paratera 重判分(fre 裸跑 0.4807 / trial1 0.4682 / trial5 0.3101;rice 裸跑 0.1452 / trial2 0.4033 …)
  都出自这个裁判,换任何一个都要重判一遍才有对照。
  ⚠️ 代价必须写明:单裁判下的绝对分与倍数**继承 serving 依赖**(同名模型异 serving 有 16% 叶级分歧),
  所以任何对外报告的 Code-Dev 数字都必须带裁判标签,且不得用它做「优于/劣于」的判断,只看方向。
  ⚠️ 启用新裁判模型前必须先把模型名登记进 `preparedness_turn_completer/utils.py` 的
  `CONTEXT_WINDOW_LENGTHS`,否则整批判分会以 `Grading failed` 静默全废(2026-09-03 踩过,64 秒烧掉一整批);`num_invalid_leaf_nodes ≤ 2` 否则作废;判分前 2000-token 真实成本探针(`refresh_after_fx.sh` 已如此)+ 金丝雀 1 份。
- **新增口径(下周)**:Code Execution / Result Analysis 叶(fre 124/7、rice 170/13,已核 `rubric.json`),配置 `paperbench.reproduction.skip_reproduction=True` + `paperbench.judge.code_only=False`,由我们在自己的 GPU 容器里跑出 `reproduce.log` 后摆卷到 `~/pb_submissions/<paper>/<run_id>/`。
- 任何倍数/差值必须附 serving 标签;每臂 ≥5 轮才允许说「优于」,n<5 只许说方向。

### 6.5 样本量与方差
- 组内 σ≈0.1(V4)/0.2(Kimi),组间效应 0.01~0.02:Code-Dev 上任何「优化」都分不出信号。执行级判据便宜,但**跑轮不便宜**(每轮含 docker build + 修复环 ≤3h + scaled ≤2h,单 GPU 互斥),n≥5 的对照要分散到整周夜间,不能塞进一天。
- scaled/full 规模 iteration 0 至少 3 seed 得 std;claim pass 要求主指标差异 > 2×合并 std;报表强制区分 best-iterate 与 mean-of-reruns。

### 6.6 自进化的前置条件(满足前不启动)
① 执行级目标函数已稳定运行 ≥1 周;② 执行沙箱与留出集冻结;③ 模型与裁判 serving 固定、种子记录;④ 优化器只能看开发集。满足后目标只能是「开发集执行级通过率」,Code-Dev 分只做周期性审计。

### 6.7 防泄漏纪律(做成代码)
- `repro/scripts/import_paper.sh` 白名单只拷 `paper.md / addendum.md / blacklist.txt`,`assert` 无 `rubric.json`/`judge/`。
- `repro/ci/check_no_rubric_leak.sh` 进 pre-commit 与每次 run 前;grep 表 `grader|rubric|forfeit|credit|paperbench|judge|weight(ed)? score`。
- `DEEPCODE_URL_DENYLIST` + 容器 `git insteadOf` 封锁论文官方仓库;下周加容器 DNS 级拒绝与出口白名单(pip index / HF / acquisitions 声明的 URL)。
- 所有提示词冻结版本号写进 `verdict.json`;流水线降级留痕(`source≠generated`、`partial_index`、`verdict=invalid`)的轮次不计入。

---

## 7. 周末计划与下周计划

### 7.1 周末(按小时;硬止损周日 18:00)

**周五晚(准备,¥0,~2h)**
- 0a 密钥卫生:**用户决定不轮换 key**(2026-09-03),该项取消 —— 但请知悉 Paratera key 曾在排查中回显到终端、并被记入本机 `logs/llm.jsonl`(发布仓库已脱敏,本机日志仍在)。仍需做的是:`frontier-evals/project/paperbench/.env.bak_siliconflow_0902` 移出仓库并加 `.gitignore`;新 key 只放 `~/.repro.env`(`REPRO_API_KEY_PARATERA / REPRO_API_KEY_SILICONFLOW`)。`refresh_after_fx.sh:18` 已改为从 `credentials.json` 读,不再是明文,无需改。
- 0b `cd ~/deepevol/DeepCode && git diff HEAD > ../deepcode_test/patches/pre_weekend_0905.patch`;确认 `git rev-parse --short HEAD` = `e0767d0`。
- 0c `mkdir -p ~/deepevol/repro/{agent,gates,schemas,scripts,docker,ci} ~/repro_runs ~/repro_store/{datasets,models,repos,skills,images}`;`cd ~/deepevol/repro && uv init --python 3.12`(与 DeepCode 的 3.11 venv 分离),依赖 `openai pydantic pyyaml jsonschema`。
- 0d Windows 侧 `/mnt/c/Users/43519/.wslconfig` 追加 `[wsl2]\nmemory=10GB\nswap=8GB` → `wsl --shutdown` → 重进后 `free -g` 复核 total ≈10;`docker info | grep -i nvidia`;`nvidia-smi -L` 应见 RTX 4060 Laptop GPU。
- **0e 端点 tool-calling 探针(不过不开工)**:`uv run python -m repro.agent.probe --base https://llmapi.paratera.com/v1 --model DeepSeek-V4-Pro`(如需第二档再加 `--model Kimi-K2.6` 或 `Qwen3-Coder-Plus`;**不要再用 SiliconFlow 端点**):注册 3 个 snake_case 工具,要求完成 `write_file → run` 两步往返,校验 tool_calls JSON 可解析、`finish_reason`、usage;顺带 2000-token 真实请求测单价。任一模型失败 → 周六第一件事修工具层。

**周六(10h,¥≤35)**
- 09:00–10:00 **步骤 1 DeepCode 三处补丁**:先 `grep -c '"finish_reason": "length"' deepcode_test/rice/task_archives/*/logs/llm.jsonl` 复核截断比例;改 `workflows/code_implementation_workflow.py:989` 读 `agents.implementation.maxTokens`;删 `agent_orchestration_engine.py:796-798` 与 `memory_agent_concise.py:1684-1685` 的 Graders 句;`bash repro/ci/check_no_rubric_leak.sh DeepCode/prompts DeepCode/workflows` 确认 0 命中;`git diff > deepcode_test/patches/weekend_min.patch`。(前端本身周末不跑。)
- 10:00–12:00 **步骤 2 客户端 + 工具 + 闸门骨架**:`repro/agent/client.py`(完整性校验、退避、usage 记账、¥熔断)、`repro/agent/tools.py`(§3.1 七个工具的 JSON schema + docker exec 封装)、`repro/schemas/{manifest,metrics,verdict}.schema.json`、`repro/gates/{manifest_check.py, static_scan.py, env_gate.sh, smoke.sh, eval_runner.py}` 各自独立可执行、输出 JSON。
- 12:00–13:00 **步骤 3 基础镜像(与步骤 2 并行后台)**:`repro/docker/Dockerfile.base` = `FROM pb-env:latest` + `conda create -n repro python=3.11` + pip(清华镜像)`torch==2.4.1+cu121 gymnasium==0.29.1 mujoco==2.3.7 stable-baselines3==2.3.2 'numpy<2' pyyaml tqdm`;`docker build -t repro-base repro/docker/`;验收 `docker run --rm --gpus all repro-base bash -lc 'conda run -n repro python -c "import torch,gymnasium as gym;print(torch.cuda.is_available());e=gym.make(\"Hopper-v4\");e.reset();print(e.step(e.action_space.sample())[1])"'` 打印 True 与一个浮点数。Docker Hub 拉不动 → `FROM deepevol-runtime:py311`。
- 13:00–14:00 **步骤 4 手写 manifest(只读 paper.md + addendum.md)**:`bash repro/scripts/import_paper.sh rice`;`~/repro_runs/rice/wk1/manifest.json`:`category=rl_online`;`environments=[{id: Hopper-v4, source: gymnasium, probe: "...", needs_gpu: false}]`(`env_ids_normalized: {Hopper-v3: Hopper-v4}`,论文用 Hopper-v3);`methods=[PPO 基础 agent(main 前置), PPO fine-tune 同额外预算(baseline), StateMask-R(baseline), RICE 精化(main)]`;`metrics=[{episodic_return, primary, higher}]`;`experiments=[{id 由 paper.md 的实验编号命名, entry: experiments/{train_agent,train_mask,refine,evaluate}.py, scale_knobs: {--total-timesteps, --n-seeds}}]`;**claims**:C1「RICE 精化 > 同额外预算的 PPO fine-tuning」、C2「RICE > StateMask-R」——两条都要填 `paper_effect_sigma`,rice Table 1 dense Hopper(`paper.md:218`:No-Refine 3559.44±19.15、StateMask-R 3635.08±9.82、RICE 3663.91±20.98)C2 效应 ≈1.4σ **不合格**,周末标 `not_testable_by_us`,claims 全部预期 `na`;`constraints_immutable=[gymnasium Hopper 默认 1000 步终止, SB3 MlpPolicy 默认结构, 3.4 节/稀疏 Walker2d/自动驾驶定性/Malware 为 addendum out-of-scope]`;`gpu_hours_estimate: 1`;`python -m repro.gates.manifest_check manifest.json`。
- 14:00–15:00 **步骤 5 种子 + 环境整备**:`cp -r deepcode_test/rice/submissions/trial2 ~/repro_runs/rice/wk1/code_v0 && cd code_v0 && git init && git add -A && git commit -m seed && git tag _seed`(`seed_source=trial2`);`python -m repro.gates.static_scan`(报告假环境审计项:预期命中 `envs/malware_env.py:83 MalConvSimulator`,不判红);手写 `env.lock.yaml`(从 `code_v0/requirements.txt` 与镜像预烤栈取交集);`docker build -t repro/rice:<sha8> -f Dockerfile.paper`;`docker run -d --gpus all --memory 8g --shm-size 2g -v ~/repro_runs/rice/wk1/code_v0:/home/submission -v ~/repro_store:/store:ro --name repro_rice repro/rice:<sha8> sleep infinity`;`bash repro/gates/env_gate.sh repro_rice env.lock.yaml` → `preflight_env.json`。
- 15:00–18:00 **步骤 6 修复环 + monitor-lite**:`repro/agent/loop.py`(单场对话、写后 py_compile、done → smoke.sh、失败签名、gate 报告回灌、同进程守卫版 monitor);`repro/agent/memory.py`(签名归一化、`failures.jsonl`、skills 注入);`repro/record_score.sh`(改自 `tmp/autosota-ref/record_score.sh`,只保留 iter/commit/protected 逻辑)。
- **18:00 检查点(硬性)**:client 探针通过、`repro-base` 与 `repro/rice` 镜像可用、`env_gate` 绿、`smoke.sh` 能在干净容器里对 trial2 原样跑出一个(可能失败的)结构化报告。未达标 → 周日的 scheduler/verify/report 全部降为桩(shell 一行脚本),只保住修复环。
- 19:00–23:00 **步骤 7 跑修复环**:`python -m repro.agent.loop --workspace ~/repro_runs/rice/wk1/code_v0 --container repro_rice --manifest manifest.json --max-iter 20 --wall 10800 --budget-yuan 20 --smoke-cap 12 --model DeepSeek-V4-Pro`;首轮要求写 `notes/code_map.md` 与根目录 `reproduce.sh`(smoke = `train_agent.py --env hopper --total-timesteps 10000 --n-envs 1 --seed 0` → `train_mask` → `refine`(含同预算 PPO fine-tune 臂)→ `evaluate` → 写 `outputs/metrics.json` 含 provenance);全绿 → `git tag _smoke_ok` → `protected_paths.sha256`;若有依赖变更 → 回写 `env.lock` 重烤镜像后再跑一次干净容器 smoke。

**周日(8h,¥≤85)**
- 09:00–10:00 **步骤 8 scheduler 最小版**:`repro/scheduler.py`(串行 docker run + timeout + `state.jsonl` + flock);`python -m repro.scheduler submit jobs/rice_smoke.json`(`image repro/rice:<sha8>, cmd 'SCALE=smoke bash reproduce.sh', needs_gpu false, timeout 1800`)→ `runs/<id>/{reproduce.log, reproduce.log.creation_time, outputs/metrics.json}`;`protected_paths.sha256` 校验通过。
- 10:00–12:00 **步骤 9 scaled 复现**:`submit jobs/rice_scaled.json`(`SCALE=scaled`:3e5 步、seed 0、`needs_gpu true`、timeout 7200);完成后 `bash repro/record_score.sh --iter 0 --status success --primary <return> --scale scaled --baseline-source measured_by_us --eval-scope Hopper-v4`;`eval_runner.py` 用 gymnasium 独立回放 checkpoint 得主指标写入 `scores.jsonl`。
- 12:00–13:00 **步骤 10 verify + 报告**:`python -m repro.verify --run ~/repro_runs/rice/wk1` → `verdict.json`(claims 全部 `na`,`reason=no_std/not_testable_by_us`;`env_ready=true`、`smoke_pass=true`、`failure_category=Succeeded(scale=scaled)`);写 `docs/runs/wk1_rice.md`:每阶段耗时、费用、gate 表、失败签名清单、blocked 组件、审计项。
- 13:00–15:00 **步骤 11(时间允许)**再提交 2 个 seed 的 scaled job 得 std 写入 `scores.jsonl`;或跑周六夜后台 DeepCode fast 前端一轮(`PAPER=rice RUN_ID=wk1_fast bash repro/scripts/run_frontend.sh`,`DEEPCODE_MAX_WALL_SECONDS=7200`,¥≤15),只归档不进关键路径。
- 15:00–16:00 **步骤 12 离线审计(可选,¥≤40)**:`code_v1` 摆到 `~/pb_submissions/rice/wk1_repaired/`;`PAPER=rice DRY=1 bash deepcode_test/scripts/run_grade.sh` 报价后,Paratera `DeepSeek-V4-Pro` 判一次(¥38),分数带裁判标签写入 `verdict.judge_scores`;对照 trial2 的 0.5447(SF)/0.4033(PT) 只说方向。
- 16:00–17:30 **步骤 13 沉淀与冻结**:`failures.jsonl` 里的签名与修法整理进 `~/repro_store/skills/`;`manifest.schema.json`、`metrics.schema.json`、`reproduce.sh` 合同、`verdict.json` 字段定稿写入 `repro/README.md`;DeepCode 以 `e0767d0 + pre_weekend_0905.patch + weekend_min.patch` 三件冻结;写 `docs/prereg/` 模板与下周待办。
- **18:00 硬止损**。

**周末硬性成功定义**:`verdict.json` 中 `env_ready=true`、`smoke_pass=true`(干净容器)、scaled run 退出码 0 且 `metrics.json` 含 Hopper×{PPO base, PPO fine-tune, RICE} 至少各一条记录且 `provenance.env_module` 顶层包为 `gymnasium`、`scores.jsonl` 有 iteration 0、`protected_paths.sha256` 校验通过;**claims 一律 `na` 是可接受的**(1 seed 无 std;dense Hopper 效应 <2σ)。任何一步靠假环境/模拟器通过都算失败。总费用 ≤¥120。

### 7.2 下周

- **周一 · 协议与集合冻结**:`docs/prereg` 模板与第一份预注册;只读 paper.md/addendum 对其余 21 篇跑 S0 分诊(手写或 `repro/manifest/triage.py`),选 3 篇留出集写入 `experiments/splits/holdout.txt` 并冻结,留出集 `rubric.json` 移出工作树;裁判四元组写进 `repro/config/judge.yaml`;`run_grade.sh` 固定单裁判(Paratera `DeepSeek-V4-Pro`);`check_no_rubric_leak.sh` 进 pre-commit;`apt install aria2`。
- **周二 · S0/S2 自动化**:`repro/manifest/extract.py`(输入白名单 paper.md/addendum/被引论文,1 次调用);在 fre/rice 上跑,召回只允许离线用开发集 rubric 的 CE/RA 叶算(`repro/eval/manifest_recall.py` 只输出数字);`repro/envprep/lock.py` 三路来源合并 + 解析回路;建老栈镜像 `repro-legacy`(py3.9 + mujoco210 + mujoco-py + d4rl@2024-06 前 commit + gym 0.23);闸门/里程碑模板按 `manifest.category` 分五类,周末的 RL 模板标「手写,待泛化」。
- **周三 · S4 数据预飞(fre)**:`repro/acquire.py` 实跑:D4RL 官方源 502 → 试 HF/Minari 镜像与格式转换,ExORL `walker/rnd.zip` 断点续传与 `buffer/` 逐 episode 适配;`UPLOAD_REQUEST.md` 流程走通一次;容器 DNS 级拒绝 + 出口白名单。
- **周三–周四 · 自建增量写码器 v1 与 CodeRAG 前端**:`loop.py` 加 planner 钩子(按 manifest 实验网格逐组件规划→写→冒烟→再规划),预算 `15 + 5×组件数`;与 DeepCode fast 前端在 rice 上 A/B(执行级指标为主,各 3 轮分散到夜间);前端:在 `agent_orchestration_engine.py:2184` Phase 9 前镜像 `:2162` 写法补 `DEEPCODE_STOP_AT_PHASE=9` 探针(5 行),`orchestrate_codebase_intelligence_agent` 前加单仓 `.py>2000` 跳过 + `skipped_repos` 留痕,`tools/code_indexer.py:558` 的「recommendation systems, graph neural networks, and diffusion models」残留改为运行时注入 manifest 关键词;**「先索引后规划」实验前提**:`codebase_index_workflow.py:196-241` 从 `initial_plan.txt` 抽 file_tree 作 `target_structure`,缺计划会回退到推荐系统默认骨架(P7/P3①),必须先把 `target_structure` 改为从 `manifest.methods/environments` 注入。
- **周四 · AutoSOTA 三件**:`protected_paths` 校验接入 S5(exit 9 不入账);独立只读审计 `repro/verify/independent.py`(未参与修复的独立会话,单裁判);失败分类看板 `repro/report/dashboard.py`(带分母的漏斗)。同日隔离测试 DeepCode `core/providers/openai_compat.py:OpenAICompatProvider`(1387 行,含 persistent 重试/token meter)能否不带 mcp 栈单独 import;能则替换 `client.py`,不能则维持薄客户端。
- **周四–周五 · fre 端到端**:老栈镜像 + 数据预飞 + 四基线清单;多 seed 方差协议(iteration 0 三 seed,claim 阈值 2×std);预算缩放公式用 fre 33+ 文件实测校准。
- **周五 · 留出集第一篇 + 报告**:按预注册跑留出集第一篇(零 rubric 接触);启用 CE/RA 口径(`skip_reproduction=True + code_only=False`)先判 1 份金丝雀再放批;`docs/repro-week1.md`:漏斗表、失败签名 Top-10、DeepCode 语料前端去留(只看执行级指标);租 GPU 方案(full 规模 1e6 步×5 seed)的 job 模板与成本估算;DeepCode venv 重建为 3.12 并逐条核对 15 处未门控改动。
- **整周持续**:每个 run 的 `verdict.json` 入 `docs/runs/`;skills 笔记增量沉淀;所有对照 n≥5 才下结论;自进化与「逼近论文倍数」不列入。

---

## 8. 风险与未决问题

### 8.1 风险与缓解
| 风险 | 缓解 |
|---|---|
| 周末范围仍是可建规模的上限:客户端、镜像、闸门、修复环、调度、verify 约 1500~2500 行新代码 | 周五晚探针 + 周六 18:00 检查点;未达标即把 scheduler/verify/report 降为桩;前端与数据预飞明确移出关键路径;种子用冻结产物 trial2 |
| 供应商限流/空响应/余额(429/5xx、V4-Pro 连续 11 次空响应、402) | persistent 退避、完整性校验为显式错误、2000-token 成本探针、¥20/环熔断、写码主力 Kimi;白天限流则夜间跑 |
| Kimi/V4-Pro 的 tool-calling 方言(连字符工具名静默不调用、并行 tool_calls、JSON 转义) | 工具名纯下划线、schema 极简、每轮校验 tool_calls;周五晚两模型探针不过不开工 |
| 假跑通:agent 用不含禁词的自写模拟器或硬编码 metrics | 硬闸 = 运行时 `env.__module__` 核验 + harness 自有 `eval_runner.py` 回放;审计项只报告不判红;`protected_paths`;下周独立审计 |
| Hopper-v3(论文,mujoco-py)→ Hopper-v4(gymnasium/mujoco3)漂移 | 写进 `env_ids_normalized` 与 `constraints_immutable`,`eval_scope` 标注与论文不同;老栈镜像下周提供 v3 对照 |
| 开发容器漂移 vs 干净容器 | 依赖变更必须回写 `env.lock` 重烤镜像;smoke 最终通过与 S5 均在干净容器 |
| WSL 内存:物理 15.7GB、当前 7GB | `.wslconfig memory=10GB swap=8GB`、`--memory 8g`、`--n-envs 1`、SCALE 三档、超时 kill |
| Docker Hub 拉取不可靠 | 基础镜像从本机 `pb-env:latest` 或 `deepevol-runtime:py311` 派生 |
| 评分元知识泄漏(fx1/fx2 前车之鉴) | import 白名单 + CI grep + manifest 只读 paper.md/addendum + 留出集 rubric 移出工作树 + 选篇不看 rubric |
| 周末产物「跑通」被误读为「复现成功」 | verdict 强制 `scale`/`baseline_source`;1-seed 与 smoke 的 claim 一律 `na`;报告只说执行级通过率 |
| 下周 A/B 若用 Code-Dev 分下结论会被 σ 0.1~0.2 淹没;n≥5 跑轮塞不进一天 | 主指标执行级;裁判分只做审计、带标签、不下优劣结论;n<5 只说方向;对照分散到整周夜间 |
| rice Hopper PPO 是 CPU-bound,周末 GPU 步只验证管道 | 明确写进 `docs/runs/wk1_rice.md`;真实 GPU 负载留给 fre/留出集 |
| DeepCode 本地版 15 处未门控改动、venv 3.11 vs 上游 3.12 | `e0767d0` + 两份 patch 冻结;下周重建 3.12 并逐条回退 |
| 密钥:`credentials.json` 曾回显、`.env.bak_siliconflow_0902` 未 ignore | 周五晚轮换、移出、只经环境变量注入;pre-commit 加 secret 扫描 |

### 8.2 未决问题
1. rice 哪条主张效应量 ≥2σ 且在 8GB VRAM 可测:Figure 2 SparseHopper 或实验 I 的 fidelity 对比是候选,需只读 paper.md 核实后写进下周 manifest;dense Hopper Table 1 已排除。
2. `OpenAICompatProvider`(`core/providers/openai_compat.py`)能否脱离 mcp/anyio 栈单独使用——周四隔离测试后决定是否替换薄客户端。
3. D4RL 官方源 502 是否持续;HF/Minari 镜像的格式转换是否可行——周三实测。
4. PaperBench CE/RA 口径下,我们自己跑出的 `reproduce.log` 是否被 `nano/task.py:266,343` 的 `_executed.tar.gz` 路径接受——启用前先读代码确认摆卷布局。
5. AutoSOTA 的 sota_category A/B/C/D、CVPR「Missed Claims」「No Improvement」精确定义未公开;我们的九类枚举只借标签名,定义以 `repro/schemas/verdict.schema.json` 的注释为准。
6. 非 RL 论文(留出集候选中的 sbibm、GPT-2 探针)的闸门模板(dataset adapter 一个 batch / forward 一次 / loss 有限 / eval 出指标)尚未设计,周二做。
7. 独立审计会话对「方法保真」的 LLM 判断会有相当比例 `uncertain`——它在漏斗里既非 pass 也非 fail,循环无法据此纠偏;是否需要第三种确定性代理(关键模块/损失/切分与论文一致性的结构检查)待下周数据。
8. full 规模(1e6 步 × 5 seed;fre 每域 12~24h 单卡)的租 GPU 预算与调度接口未定。