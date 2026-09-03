# 踩坑总表(给下一个建设者)

> 2026-09-03 汇总。本项目两周内在 DeepCode / PaperBench / 模型供应商 / 评测方法学 / 运维上踩过的全部坑,
> 每条一行:现象 → 根因 → 修法/规避 → 证据所在。**造自己的复现 agent 前先通读一遍,大多数坑会在新系统里原样重现。**
> 编号沿用原文档(RUNBOOK 坑 1–14、CLEAN_E2E_PLAN D1–D5、FINDING P1–P8),便于回查。

## A. PaperBench 侧(评测框架)

| # | 现象 | 根因 | 修法 / 规避 | 证据 |
| --- | --- | --- | --- | --- |
| 坑1 | README 的命令跑不起来 | chz 参数按基类展开,不先写 `paperbench.solver.computer_runtime=...AlcatrazComputerRuntime` 就设不了 `.env.*` 子参数 | 每条命令显式加该行 | `docs/PAPERBENCH_RUNBOOK.md` §落差 1 |
| 坑2 | dummy 裁判也要 `GRADER_OPENAI_API_KEY` | 无条件断言 | 占位值即可 | 同上 2 |
| 坑3 | 复跑阶段去 Docker Hub 拉不存在的 `pb-reproducer:latest` | `pull_from_registry` 默认 True,README dev 示例漏配 | 复跑运行时三行单独配 `pull_from_registry=false` | 同上 3 |
| 坑4 | 官方 `pb-reproducer` 镜像被自家运行时拒绝 | 镜像只有 `python3`,Alcatraz 探测 `python`/`pip` | 镜像补 `python-is-python3 + python3-pip` | 同上 4 |
| 坑5 | 非 OpenAI 裁判直接 `ValueError` | `CONTEXT_WINDOW_LENGTHS` 是 OpenAI 专属硬编码,无配置口 | `preparedness_turn_completer/utils.py` 加模型名;**换供应商/裸模型名要再加一次**(Paratera `DeepSeek-V4-Pro` 曾因此 6 份判分全失败) | 同上 5;`fre/logs/fre_ledger_fix.txt` 09-03 07:45 |
| 坑6 | 非 OpenAI 端点上 178 条判词全灭 | 裁判二级结构化解析器写死 `gpt-4o-2024-08-06` | `judge/simple.py` 改读 `PB_STRUCTURED_PARSER_MODEL`(默认不变) | 同上 6 |
| — | 只判了 1 份,其余无声忽略 | 每个 task 实例只 `pop()` 一份提交 | `paperbench.n_tries=N`;`run_grade.sh` 自动数目录 | `scripts/run_grade.sh` 头注 |
| — | 配置校验阶段直接失败 | `~/pb_submissions/` 下有非 paper-id 目录(如 `fre_archive`) | 归档放 `~/pb_submissions_archive/` | 同上 |
| — | 判分中途余额耗尽,分数被压低但看似正常 | 150~168 个叶无效仍出总分 | 判前 2000-token 真实探针 + 金丝雀 1 份;`num_invalid_leaf_nodes ≤ 2` 否则作废 | `docs/CONCLUSIONS.md`;P6 |
| — | 数据全是 LFS 指针 | `.lfsconfig fetchexclude=project/paperbench/data/**`;稀疏/浅克隆下 `git lfs pull --include` 退出 0 但不取对象;`--filter=blob:none` 下 lfs 逐 blob 懒取挂死 | 按固定 commit 从 `media.githubusercontent.com` 直链下载(`setup.sh` 已实现);checkout 加 `GIT_LFS_SKIP_SMUDGE=1` | `setup.sh` 注释 |
| — | 裁判"没看到文件"给 0 分且 `valid_score=True` | 每叶先让 LLM 从目录树选文件,返回空即 `<files>` 空 | 无法从判分侧修;报告时统计"疑似空文件"零分叶;大目录树更易触发 | `rice/RESULTS.md` §4d;P2/P5 |

## B. DeepCode 配置侧(不改源码就能修)

| # | 现象 | 根因 | 修法 | 证据 |
| --- | --- | --- | --- | --- |
| 坑7 / D1 | 所有 agent 零工具空转;写码无 `write_file`,产物为空;或 `command-executor` 缺失 | `deepcode init` 不写 `tools.mcpServers` | 手工补 7 个 MCP 服务器,**必须 `python -m tools.xxx` 模块方式**启动;`filesystem` 用 npx,`fetch` 用 uvx | RUNBOOK 7;`CLEAN_E2E_PLAN.md` D1 |
| 坑8 | 推理模型写码必死:空响应 → 截断 → 300s stall 熔断,只写 3/24 文件 | 默认 `maxTokens=8192` 装不下"推理 + 代码" | `agents.defaults.maxTokens ≥ 32768`(本项目用 65536);写码可指定快模型 | RUNBOOK 8 |
| — | "完整模式"反而是最弱配置 | `enable_indexing=True` 只给 2 个工具(`write_file` + `search_code_references`),fast 模式给 11 个 | 要么让 CodeRAG 真正生效,要么用 fast;绝不"完整模式 + 空索引" | RUNBOOK 8 ⚠️ |
| 坑9 | 参考挖掘 / 下载 agent 饿死 | `max_iterations` 默认 8 | 提到 40~80(本项目写死 80,**属未门控改动**) | `docs/REVIEW_local_changes_2026-09-03.md` |
| 坑10 | 下载 agent 自主克隆论文官方仓库(作弊) | 论文声称的黑名单在代码里不存在 | git `insteadOf` 封锁 + MCP 层 `DEEPCODE_URL_DENYLIST`(insteadOf 挡不住 HTTP 抓取);实测真挡下过一次 | `scripts/run_trial.sh`;`rice/RESULTS.md` |
| 坑11 | f-string 语法错误 | **我方误判**:DeepCode 要求 py≥3.12,我们的 venv 是 3.11 | venv 重建为 3.12 | RUNBOOK 订正 |

## C. DeepCode 源码缺陷(要改代码;本项目全部 env 门控或已记录)

| # | 现象 | 根因 | 修法 | 证据 |
| --- | --- | --- | --- | --- |
| 坑12 | 连续 `write_file` 被 LoopDetector 当死循环杀掉 | 循环检测不区分写类工具 | 写类豁免(未门控改动) | REVIEW |
| — | 300s 无落盘即熔断;白天空响应期可达 30~50 分钟 | stall 阈值常量 | 1800→7200(未门控/env) | `run_trial.sh` 注释 |
| — | 索引阶段每次全量重建(¥3~30 白烧) | 唯独索引无幂等 | `_index.json` 齐全即复用 | RUNBOOK 7-12 |
| P1/坑 | CodeRAG 预筛 JSON 截断 → **静默回退全量索引**(17 文件仓库成功,151/239 文件 100% 失败;google-research 8,885 py 需 140h) | `code_indexer.py:566 max_tokens=2000` 写死;截断点聚在 2000×4.3 字符 | `DEEPCODE_PREFILTER_MAX_TOKENS`(默认不变) | `docs/FINDING_prefilter_silent_failure.md` |
| P1 | 参考挖掘报告截断,续写只留尾段 → 下载侧只见 1/5 仓库,整轮语料贫瘠 | `reference_params.maxTokens` 写死 | `DEEPCODE_REFERENCE_MAX_TOKENS`;A/B:8192→1 仓库,32768→5 仓库 | 同上 附 |
| P3 | 预筛返回**合法空列表**也回退全量,日志一律写 "failed" | `if selected_file_paths:` 把"选 0 个"和"调用失败"合并 | 分流 + WARNING(未修) | 同上 附二 |
| P7 | 预筛提示词硬编码 "recommendation systems, GNN, diffusion",把 RL 文件判为不相关 | 上一个项目的领域先验残留(`code_indexer.py:558`) | 运行时注入论文关键词(未修) | 同上 |
| P8 / 假计划 | 规划三连败后上游用 `coerce_text_to_minimal_plan` 造通用脚手架假计划并标 `completeness_score=1.0`,整轮静默报废 | `planning_runtime.py:174`;规划超时默认 180s 对推理模型不够 | `DEEPCODE_CODE_ANALYZER_TIMEOUT_S=600` + 摆卷前核 `planning_result_meta.json.source == generated` | `run_trial.sh` 假计划闸 |
| P8 | 规划一次定死文件树,漏掉的基线后面几百轮补不回来(fre 三基线全 0) | Phase 5 冻结 + 写码只填空壳 | 规划必须可增量修订;**整体重生成会改坏主方法**(fx 轮实证) | `fre/RESULTS.md`;`docs/CONCLUSIONS.md` §⑥ |
| — | 写码全程不执行不验证 | 索引模式工具面无执行器;`command_executor` 2,443 次工具调用零次运行代码 | 自建循环把执行放进去 | `release/README.md` §1.3 |
| — | 预算不随计划规模变;每文件 clean-slate 内存(V4-Pro 每文件冷启动 6.8 分钟) | `_MAX_ITERATIONS=800`、墙钟常量;`memory_agent_concise.py` | 预算按组件数 × 优先级;单场连续对话 | `code_implementation_workflow.py:90` |
| D2 | `GITHUB_DOWNLOAD_PROMPT` 是从未接线的死代码,实际用的是一行内联 instruction | 重构失修 | 本项目重写(未门控) | `CLEAN_E2E_PLAN.md` D2 |
| D3 | 参考成功、下载失败 | 下载 agent 17 个工具未过滤 vs 参考 agent tool_filter 收窄到 3 个 | 加 tool_filter | D3 |
| D4 | 下载失败被静默吞成"成功" | 代码侧无校验 | 空 `code_base` fail-fast | D4 |
| D5 | Kimi 对含连字符的工具名**静默不调用**(裸 API A/B 实证) | 工具名 `mcp_xxx-yyy` | 工具名 `-`→`_` 消毒;自建工具面只用 `[a-z_]` | D5 |
| P1 | 判分侧文件选择输出随目录树增长 → 空 `<files>` | 同一模式第四处 | 见 A 表末行 | `FINDING_generic_pipeline_failures.md` |

## D. 模型与供应商

| 现象 | 根因 | 规避 | 证据 |
| --- | --- | --- | --- |
| 白天 429 / 5xx / 连续 11 次空响应,整轮报废 | SiliconFlow 日间限流;上游重试只有 1/2/4s 三次 | persistent 退避 `10/30/60/180/300s`、上限 900s、同错 30 次;夜间跑;后改用 Paratera(30 分钟零异常) | `run_trial.sh` 注释;台账 |
| **同名裁判模型换一家 serving,rice 倍数 1.05× ↔ 2.58×**;同一提交 16% 叶级分歧;JudgeEval 上二者同等准确 | serving 差异(对话模板、推理量:输出 token 减半) | 任何分数带"模型名 + serving + temperature + 提示词版本";LLM 裁判分不做目标函数 | `docs/FINDING_judge_serving_dependence.md` |
| 坑14:Paratera 余额耗尽**不报 402**,付费模型 403 `team_model_access_denied`、免费档仍 200、模型表从 93 掉到 8 | 平台降级到免费档;无余额查询端点 | 开跑前 `scripts/paratera_key.sh check`(三件同时出现 = 余额耗尽) | RUNBOOK 坑 14 |
| Paratera 没有 Kimi-K2.7-Code、Qwen3-Coder-Plus | 模型表差异 | 换模型前 `GET /v1/models` 核对;tool-calling 先探针 | `docs/ARCHITECTURE_v0.2_OPTIMAL.md` §0.3 |
| 推理模型输出被 `max_tokens` 截断且无告警 | 上限写死在四处(agents / 预筛 / 挖掘 / 下载) | 每次调用校验 `finish_reason`、长度、JSON 闭合、空返回 | P1 |
| 密钥被写进脚本、被 DeepCode 记进 `logs/llm.jsonl` | 明文与日志 | 只经环境变量注入;发布前全树扫描;日志脱敏 | REVIEW 🔴 |

## E. 评测方法学(最贵的坑)

| 坑 | 后果 | 规则 |
| --- | --- | --- |
| 修复提示词里写了 "Graders assign separate credit to each baseline" | 两轮 ¥45 + 判分全部作废(评分元知识泄漏) | rubric 物理不进工作区;CI grep `grader|rubric|forfeit|credit|paperbench|judge|weight`;重要性只来自论文自身 |
| 按评分表权重规划/校验 | 过拟合 fre/rice | 标准从 paper.md + addendum 编译 |
| 反事实推算"补基线 → 1.35×"假设维度可加 | 实测基线有分、主方法下滑,总分未升 | 反事实只作假设,必须实验验证 |
| 每组 2 轮就下结论 | 组内摆动 0.13~0.16,组间 0.01~0.02,全在噪声里 | n≥5 才说"优于";预注册;留出集 |
| "官方默认值一字未改"的表述 | 独立审查发现 15 处未门控改动 | 每个改动 env 门控 + 默认等于上游,或如实列出 |
| 只用一个裁判 | 结论随 serving 翻转 | 双模型/双 serving 并列,或执行级主指标 |
| 用裁判分做目标函数 | 2.58× 来自"写得具体"讨好裁判,不是复现更忠实 | 执行级判据为主 |

## F. 运维杂坑(会吃掉整晚)

| 现象 | 根因 | 规避 |
| --- | --- | --- |
| `pkill -f "xxx"` 把自己的 shell 杀了(exit 144) | 命令行/heredoc 里含匹配串 | 用 `pkill -f "xx[x]"` 括号技巧;kill 与 heredoc 分开两条命令 |
| 后台脚本改了不生效 / 错位执行 | bash 逐行读取正在运行的脚本 | 运行中的脚本不改;改副本或等结束 |
| 监控哨报的数字不更新 | 启动时锁定了旧日志路径 | 每 tick 重新解析最新日志 |
| 另一个会话拉 GitHub 导致克隆 TLS 断流,三轮作废 | 网络争用 | 大克隆期间避免并发拉取;换节点 |
| Docker Desktop 没起,判分 sanity check 失败 | — | `docker info` 进闸门 |
| 判分池残留已判副本,重判白花 ¥150 | 判完不清池 | 判完即归档,`~/pb_submissions/<paper>/` 保持空 |
| WSL 只有 7GB 内存 | `.wslconfig` 无 `[wsl2] memory=` | `memory=10GB swap=8GB`(物理 15.7GB,别设 16) |
| 老任务目录混入新轮 | `deepcode_lab/tasks/paper_*` 未清 | `run_trial.sh` 开跑前归档全部旧目录并校验为空 |
| 产物拿错论文摆卷 | 交接文件跨论文 stale | 按论文分交接文件 + `paper.md` 标题核验 |

## G. 哪些不是坑(被我们误判过)

- 坑11 f-string:我方 3.11 环境的问题,不是上游 bug。
- fre trial4 "只提名 1 个仓库":报告完整 5 条,是下载侧问题;fx1 首跑的"1 仓库"才是报告截断。
- CodeRAG 预筛失效不是低分主因(时序上规划先于索引;失效率与分数不相关;全量是筛选的超集)——它的代价是时间与成本。
