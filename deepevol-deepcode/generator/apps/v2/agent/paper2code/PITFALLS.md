# Paper2Code 线踩坑总表

一处维护所有坑（PLAN-2 D4）。来源：本线 C8/C9 与 sapg 复跑的 HANDOFF 记录、验证仓库 `docs/PITFALLS.md` 与 README §6（DeepCode / PaperBench 侧，2026-08-25 起 60 余条）、`bam-threeway` 的 HOWTO/HANDOFF、`search/SETUP.md`。
每条：现象 → 原因 → 本线怎么处理 / 哪个旋钮（或哪个测试钉住）。HANDOFF 只留运行记录，坑一律加在这里。
"本线" = `apps/v2/agent/paper2code/`；"基线运行" = 验证仓库 `2UBBISH/deepcode-paperbench-validation` 里原装 DeepCode 的 `run_trial.sh`。

## A. 模型与供应商（口径）

| 现象 | 原因 | 本线怎么处理 / 旋钮 |
| --- | --- | --- |
| "思考关"没关：70% 输出 token 是推理，计划被截断，续写把推理文当正文接回去，规划三连败 | Paratera 的 OpenAI 兼容路由忽略 `enable_thinking:false`，也拒绝 `reasoning_effort:"none"`；只认 DeepSeek 官方的 `thinking:{"type":"disabled"}` | `provider.py` 每次请求带 `thinking:{"type":"disabled"}`，回包 `reasoning_tokens != 0` 即 `ThinkingNotDisabled`（`test_provider.py`）。基线运行靠补丁 `compat.thinking`（`protocol_config.py`），跑完 `run_trial.sh` 汇总 `reasoning_tokens` |
| V4-Flash 某次回包 `reasoning_content` 非空但 `reasoning_tokens=0`，旧守卫直接中止，引擎悄悄退回全文规划 | 供应商偶尔回显空推理块 | 口径按 token 计：只在 `reasoning_tokens != 0` 时中止，`reasoning_content_chars` 记进 `llm/<seq>.json` 并告警；`PAPER2CODE_STRICT_REASONING_CONTENT=1` 恢复严格 |
| 同一把 key 对所有 DeepSeek 模型 403 `team_model_access_denied`，`/v1/models` 只剩 8 个 GLM | Paratera 余额耗尽**不报 402**，而是把团队降到免费档 | 开跑前 `gates.preflight` 探针（`--no-probe` 只给离线测试）；换 `~/Documents/env/paratera_backup.env`；三件同时出现（403 + 免费档 200 + 模型表 93→8）= 余额耗尽 |
| 白天 429 / 5xx / 连续空响应，三次重试打完整轮报废 | 上游只有 1/2/4 秒三次 | `ENV_DEFAULTS`：`DEEPCODE_LLM_RETRY_MODE=persistent`、退避 `10,30,60,180,300`、上限 900 s、同错 30 次；`provider.py` `RetryPolicy` 同语义 |
| 复现正常、判分全部 401 | 底座与裁判各读一处 key，只改了一处 | 本线 key 只走 `--env-file`（`PARATERA_API_KEY`）；裁判 key 在 `vendor/paperbench-judge/frontier-evals/.../paperbench/.env`；验证仓库 `paratera_key.sh set` 两处同写 |
| 基线运行配置写了 `maxTokens: 32768`，DeepCode 日志却是 `Resolved workflow LLM … max_tokens=8192` | 21ebc57f 的执行档按模型目录的 `maxOutputTokens` 钳每次调用；手动目录只写模型名时 deepseek 家族缺省 8192。本线自己的 provider 不查目录，`max_tokens` 就是 32768 | 验证仓库模板把 V4-Flash 声明成 `{id, contextWindow, maxOutputTokens: 32768}` 的条目并在预飞核对；sapg trial1（2026-09-17）是 8192 下跑的，两边并排时注明 |
| 推理模型输出被 `max_tokens` 截断且无告警（预筛 JSON 半截、挖掘报告只剩尾段） | 上限写死在四处 | `ENV_DEFAULTS`：预筛 32000、分析/关系 16000、挖掘 32768、下载 16384；`runner.py` 对 `finish_reason=length` 续写 ×3、空响应重试 ×2 |
| 同名裁判模型换一家 serving，rice 倍数 1.05× ↔ 2.58×，同一提交 16% 叶级分歧 | serving 差异（对话模板、推理量） | 任何分数带"模型名 + serving + 思考状态"；裁判恒定 DeepSeek-V4-Pro @ Paratera；LLM 裁判分不做目标函数 |
| GLM-4.5-Flash 一次调用 20–60 s，挖掘几轮就是几分钟；在已有文件上继续写码会漂移一小时改写文件 | 替代模型只做管路冒烟 | 只在 `ema-glm` / `sapg-glm` 做管路验证，永不判分；漂移时 `rerun --phase implement`（干净树 13 分钟跑完） |

## B. 引擎（vendored DeepCode 业务层）

| 现象 | 原因 | 本线怎么处理 / 旋钮 |
| --- | --- | --- |
| 参考挖掘 8 轮用完，`reference.txt` 里是 runner 的"到达最大迭代"文本，下游当"没有参考仓库"继续跑 | 引擎把 analyzer 最后一句原样写盘；V4-Flash/GLM 会把 56k 字论文重读六次 | `phases.reference_report_is_degenerate` 让 `references` 阶段失败并点名旋钮；默认 `DEEPCODE_REFERENCE_MAX_ITERATIONS=40`、`DOWNLOAD=12`（PLAN §6，D0）；`test_degenerate_reference_reports_are_named` |
| 下载 agent 传 `target_path=""`，两个仓库克隆进仓库根目录 | 引擎工具把空/相对路径按进程 cwd 解析 | `tools/kernel_servers.CloneIntoCodeBase`：只取模型给的最后一段或从 URL 推名，一律放 `code_base/`；`GIT_TERMINAL_PROMPT=0` 让私有/不存在的仓库立刻失败（`test_kernel_servers.py`） |
| 下载 agent 只叙述不调工具，`code_base/` 为空，流水线照样进"索引模式 + 空索引" | 上游一行指令 + 17 个未过滤工具；空 `code_base` 只打印 | VENDOR 4：工具优先提示、只给 `git_clone`、一次纠正重试、空 `code_base` 即失败；参考报告没点名任何仓库时 `acquire` 记 `skipped` 而不是失败（EMA-Detect） |
| 大仓库预筛 JSON 在 11.8 万字符处断掉（`LLM pre-filtering failed: Expecting ',' delimiter`），引擎回退全量索引，263 个文件逐个过模型（基线 trial2 IsaacGymEnvs，2026-09-17；本线同一仓库同一轮预筛正常，101 文件） | 预筛对每个候选文件各出一条 JSON 记录，263 文件 ≈ 3 万 token，`DEEPCODE_PREFILTER_MAX_TOKENS=32000` 也兜不住；上游 P1 静默降级原样在两边引擎里 | `phase_index` 结果按仓库记 `files_found / files_analyzed / prefilter_fallback_suspected`，事件 `index.prefilter_fallback`；**已修（VENDOR 10，验证仓库同补丁）**：预筛只要路径 + 置信度（缩 4 倍）、`_call_llm` 见 `length` 即失败重试、日志打实际选中数；`test_engine_patches.py` 三个用例钉住 |
| 预筛提示词末尾"Focus on recommendation systems, graph neural networks, diffusion models"把 RL 论文的环境/PPO 文件往下压 | 作者自己示例工程（gcn.py / diffusion.py）的领域先验，2025-07-20 首发就在，上游从未改 | VENDOR 10 改成中性表述；`test_prefilter_prompt_is_domain_neutral_and_asks_for_paths_only` 钉住。**量过（09-18）**：IsaacGymEnvs 263 文件，修前唯一样本（C9）选 52，修后三次选 101 / 54 / 83，修前的 52 全在修后集合里（没有被偏置排除的 RL 文件）；修后三次都选、修前没选的是 PBT 配置与 AllegroKuka/dextreme 任务文件。方向对，但预筛本身的运行间波动（54–101）比修前后差大，单样本证明不了它伤过分数 |
| 参考报告写 `Repository: owner/repo` 不写 URL，`acquire` 认为没有仓库 | 模型的简写 | `phases.github_urls_in` 认简写（`test_github_urls_in_reference_report`） |
| 规划三连败后上游伪造通用脚手架计划并标 `completeness_score=1.0`，整轮静默报废 | `coerce_text_to_minimal_plan` | `gates.plan_source`：`planning_result_meta.json.source != generated` 即失败（`test_gates.py`）；规划限时 `DEEPCODE_CODE_ANALYZER_TIMEOUT_S=600` |
| 完整计划被判校验失败 → 假计划 | block scalar 里嵌套的 bash 围栏截断了 YAML 提取 | 引擎 `planning_runtime.extract_yaml_candidate` 只认未缩进围栏（vendor 自带；基线补丁 B 组同款） |
| 写码报 0 个文件、永远到不了完成判定 | 只统计 `write_file`，模型用 `write_multiple_files` | 引擎统计批量写（vendor 自带；基线同款） |
| 连续 `write_file` 被 LoopDetector 当死循环杀掉 | 循环检测只看工具名 | 写类工具按参数摘要键（vendor 自带；基线同款） |
| 分段 agent 只叙述不调工具，`segments_available=True` 但 `document_segments/` 不存在 | 相信 agent 的话 | VENDOR 5：`document_index.json` 不存在即错误；`phase_plan` 记 `segmentation_status` |
| 300 s 无落盘即熔断；写码 2 h 墙钟掐掉健康运行 | stall / 墙钟常量 | VENDOR 6：`DEEPCODE_STALL_THRESHOLD=7200`、`DEEPCODE_MAX_WALL_SECONDS=21600`（`ENV_DEFAULTS`） |
| 引擎"验证"只在发现测试命令时才跑；无测试的仓库从不碰执行端口，租的机器白租 | `_verify_generated_code` 只跑发现到的命令 | `environment_run` 固定跑一个 `python -m compileall -q .` 作业（`_compile_check`），再做入口冒烟（`entry_smoke.py`，只记录）；`test_run_until_environment_run_all_green` |
| 全部文件写完但没有测试，引擎报 `unverified` / `no_tests_discovered`，闸门把它当失败 | 引擎的措辞 | `gates.implementation_status`：所有计划文件已写即通过并记 `verified: false`；`test_failed` 与提前停止仍失败 |
| 空 `blacklist.txt` 让 preflight 失败 | 旧规则要求非空 | 规则改为"每条黑名单都在 run 的 denylist 里"（`gates.preflight`） |
| `rerun` 覆盖了早先的 `llm/<seq>.json` | 序号从 1 重数 | `provider._next_seq` 跨进程续号 |
| `import` 后的 f-string 里有反斜杠，3.11 报语法错 | 上游按 3.12 写 | VENDOR 7 hoist；本线 venv 是 3.12 |

## C. 工具与网络

| 现象 | 原因 | 本线怎么处理 / 旋钮 |
| --- | --- | --- |
| `fetch` 打不开任何 GitHub 页：`_SafeResolver` 拒绝 198.18.0.61 | 这台 Mac 的代理 DNS 返回 fake-IP（198.18/15，Clash/Surge） | `tools/fetch.py`：解析落在 `FAKE_IP_NETWORK` 时退回系统解析器；URL 校验与 denylist 不变，真私网地址仍拒（`test_fetch_policy.py`）。云上机器不受影响 |
| 模型对同一个死 URL 反复 fetch，把迭代预算烧光 | 无记忆 | `tools/registry` 重复 fetch 账本（同 URL ≤ 2 次）；基线补丁 `mcp.py` 同款 |
| 下载 agent 自主克隆论文官方仓库（作弊） | 论文声称的黑名单开源版没有 | `DEEPCODE_URL_DENYLIST` 由 `run.json.denylist` 生成，`DenylistedTool` 在工具层拒（`test_tools_registry.py`）；`gates.preflight` 核对 `blacklist.txt` ⊆ denylist；基线运行两层拦（git insteadOf + MCP） |
| Kimi 对含连字符的工具名静默不调用 | `mcp_github-downloader_git_clone` 这种名字 | 本线工具名一律 `mcp_<server>_<tool>`，`-`→`_`（`build_registry`）；VENDOR 2 让 `tool_filter` 按消毒后前缀匹配 |
| 这台 Mac 的网络抖动能同时杀掉 SSH 会话与阿里云 API 调用 | 代理/网络 | 见 D 组：容器后台化、ECS 传输重试、`release` 兜底 |

## D. 执行与机器

| 现象 | 原因 | 本线怎么处理 / 旋钮 |
| --- | --- | --- |
| 开机 60 秒后 SSH 已通、bootstrap 已跑完，`DeleteInstance` 却报 `IncorrectInstanceStatus.Initializing`，之后几分钟都如此；main 的 `ExperimentLease.release` 只重试 3 次（2/4/6 s）就判释放失败，机器留在账上（2026-09-17 4a 真机检查，靠 `force_release` 补删） | 阿里云的"初始化中"比 sshd 就绪晚得多，而编排器的重试是按 API 抖动设计的 | `AliyunLeaseBackend.release` 见 `IncorrectInstanceStatus` 就每 10 s 重试、最多 5 分钟（`DELETE_RETRY_*`）；`lease.json` 仍会记 `release_failed`，`release` 子命令兜底 |
| SetupX 的 `json_mode` 会带 `response_format: json_object`，而 Flash 在带它的请求上返回打乱的正文（E 表的裁判坑同源） | 供应商 | 回环端点不转发 `response_format`（记 `dropped`），靠提示词要 JSON + SetupX 自己的一次显式重试 |
| 租机两分钟后才报 `ModuleNotFoundError: No module named 'docker'`——SetupX 在模块顶部 import docker SDK，它在 main 的 `agent-runtime` extra 里，本线 venv 没装 | 依赖只在运行时暴露 | `experiment_step.missing_modules()` 在租机**之前**探 `docker / dotenv / httpx / asyncssh / rsa / run_flow`；venv 用 `.tools/bootstrap/bin/uv pip install docker` 补 |
| 判据文件 `criteria_G0.py` 语法错：`unterminated string literal` 在目标句的第二行 | RSA 把目标句渲染成一条 `# goal:` 注释（`rsa/render.py:270`），多行目标句第二行落在注释外 | `build_goal` 压成一行（`one_line`），测试断言无换行 |
| 判据被证伪拒绝：`command references 'runs/smoke_sapg', which does not exist at commit …` | 证伪器把命令里所有相对路径当输入、只豁免固定的输出 flag 名单（`--output-dir` 在、`--output_dir` 不在）和绝对路径 | 目标句要求输出目录用仓库外的绝对路径（`/workspace/out/…`）、只断言入口命令自己写的产物；被拒时带原因自动重编译一次 |
| SetupX 环境 30 步就装好（CPU torch 2.14）却接着用 70+ 步诊断代码 bug（`int(None)`、`policy.sample` 4 值 vs 3 值解包），甚至去改 site-packages 里的拷贝 | 它的契约只说"改仓库无效"，没有"这是代码 bug、交出去"的出口；RSA 默认 200 步 × 8 轮 | 第 0 轮预算压到 60 步 × 3 轮（`PAPER2CODE_SETUPX_MAX_STEPS/_MAX_ROUNDS`）；代码 bug 归修复轮 |
| SetupX 第一次装 torch 从 PyPI 拉 CUDA 全家桶（555 MB 的 torch 之后还有 nvidia-*），20 分钟后才自己想到装 CPU 版 | 它不知道机器没有 GPU | 给它的系统提示副本追加"机器须知"（CPU 索引、长命令后台 + 轮询）——`append_setupx_addendum`，用 main 打契约补丁的同一机制 |
| nohup 起的后台进程收不到 SIGINT（非交互 shell 的后台作业忽略 SIGINT），`kill -INT` 无效；`kill -TERM` 直接死、不走释放路径 | POSIX 后台作业信号规则 | 杀掉后**必须**跑 `release` 备份命令；真机上 `force release … deleted` 证明机器确实还在 |
| `docker pull python:3.11-slim` 经 docker.sock 隧道在机器上挂满超时；同一台机器 SSH 直接 pull 7 秒 | vendored 隧道 | `SshDockerHost`：所有 docker 命令经 SSH 在机器上跑（PLAN C5 的记录在案偏差）；`SshOnlyDaemon` 不开隧道；本机测试用 `PAPER2CODE_BASE_IMAGE` 指本地镜像（`test_job_executor.py`） |
| 机器上 Docker Hub 拉不动 | 区域网络 | `ImagePolicy`：先 `docker pull`（5 分钟），再 `docker.1ms.run` / `daocloud` / `dockerproxy` 镜像，全记 `jobs/image-build.log`；build 用 `--pull=false` |
| `'RemoteDaemon' object has no attribute 'machine'` 把 20 分钟的 build 挂死盖住了 | 属性错误吞掉真实异常 | 机器名来自租约，作业起不来时记真实异常，`JobResult.machine` = `aliyun:<instance>@<ip>` |
| 长 `docker run`（装 torch）随 ssh 会话一起死；重试撞 "container name already in use"（exit 125） | 前台容器绑在会话上 | `_run_detached`：`docker run -d` + 10 s `docker inspect` 轮询 + `docker logs`，名字带时间戳，显式删除（`test_job_executor.py`） |
| compileall 的字节码同步回本地生成目录 | 写在工作区里 | 容器内 `PYTHONPYCACHEPREFIX=/tmp/pycache` |
| ssh exit 255 被记成作业退出码 | 传输错误与作业码混在一起 | `leased_runtime`：255 = 传输错误，作业重试 |
| `DeleteInstance` 在 `SSL: UNEXPECTED_EOF` 上失败，机器还在跑、租约却写"已释放" | 释放不能撒谎 | `RunLease`：失败记 `release_failed`；`EcsClient` 每个动作传输错误重试 4 次；`release` 子命令先 `DescribeInstances` 再删（`test_leased_port.py`）。每次跑完核对 `lease.json.released_at` 与控制台无 `p2c-*` |
| requirements 镜像构建失败（ssh 断） | 同上 | best-effort：失败落 `image.json status=failed` 并用基础镜像；`environment_run` 记 `requirements_installed`（D0） |
| `uv sync` 在 Intel Mac 上 `onnxruntime` 无 wheel | `pymupdf4llm` 拉进来的 | `uv sync --group dev --no-install-package onnxruntime`；基线运行的 `cryptography` 同理 `--only-binary cryptography` |

## E. 判分侧（PaperBench）

| 现象 | 原因 | 本线怎么处理 / 旋钮 |
| --- | --- | --- |
| 只判了 1 份，其余无声忽略 | 每个 task 实例只 `pop()` 一份提交 | vendored `run_grade.sh` 自动数目录设 `paperbench.n_tries` |
| 配置校验阶段直接失败 | `~/pb_submissions/` 下有非 paper-id 目录 | `submit` 只往 `<dest>/<paper>/<trial>/` 摆，`paper` 必须是单段；归档放 `~/pb_submissions_archive/` |
| 判分中途余额耗尽，分数被压低但看似正常 | 150+ 叶无效仍出总分 | `num_invalid_leaf_nodes ≤ 2` 才有效（`run_grade.sh` 核验）；判前探针 |
| 裁判"没看到文件"给 0 且 `valid_score=True` | 模型省掉树根 `submission/`，上游原样拼路径 | vendored PaperBench 补丁：精确解析（允许带/不带唯一顶层目录）+ 重问一次 + 仍空记无效叶；bam 三份因此重判 |
| Flash 当裁判：叶子成片 `Grading leaf … failed`，二级解析器收到前缀重复的坏 JSON（`{"valid{"valid_score…`） | Paratera 的 `DeepSeek-V4-Flash` 在带 `response_format` 的请求上返回打乱的正文；自由文本的判分正文没事 | 裁判可以是 Flash，`PB_STRUCTURED_PARSER_MODEL` 保持 `DeepSeek-V4-Pro`；本线 provider 从不发 `response_format`，不受影响 |
| 非 OpenAI 裁判模型直接 `ValueError` / 178 条判词全灭 | 上下文长度表与结构化解析器写死 OpenAI 模型名 | 补丁：`utils.py` 登记模型名（换裁判要再加），`PB_STRUCTURED_PARSER_MODEL`，`PB_JUDGE_CONCURRENCY=20`（100 会被 Paratera 打 429） |
| macOS 上判分沙箱起不来 | Docker Desktop 的 socket 不在 /var/run | 补丁 `is_docker_running` 走 `docker.from_env()`；`run_grade.sh` 设 `DOCKER_HOST` |
| 判新论文时 `paper_split` 不认 | 上游是硬编码枚举 | `nano/eval.py` Literal 加 id + `experiments/splits/<id>.txt`（补丁里有四篇的写法） |
| 提示词里一句 "Graders assign separate credit…"，两轮整体作废 | 评分元知识进流水线 | fix-①② 本线不带、基线默认关（PLAN-2 §1）；rubric 物理不进工作区（`criteria` 阶段只记 `rubric_passthrough`，不读）；验证仓库 `ci/check_no_rubric_leak.sh` |
| 开图跑（`--figures on`）的记录像开了图，实际一张都没描述：`01_intake.json` 的 figures 里 `skipped` 全是 `lfs_pointer` | PaperBench 的 `assets/*.jpg` 是 git LFS 指针（百来字节文本），图描述步按文件头判不是图就跳过，跑下去就成了贴着开图标签的关图运行 | 09-19 起 `figures=on` 且 `figures_found>0, described==0`（或图步 failed / unsupported）→ intake 失败（`phases.figures_on_but_undescribed`，事件 `figures refused`）；`auto` / `off` 不拦。补图：按 HANDOFF-STAGE9 §3 从 HF 下载并按 oid 核对，再 `rerun --phase intake` |
| 每组 2 轮就下结论 | 组内摆动 0.13~0.16，组间 0.01~0.02 | n ≥ 5 才说"优于"；本批只做两边并排、不判分 |

## F. 运维与本机

| 现象 | 原因 | 本线怎么处理 / 旋钮 |
| --- | --- | --- |
| 排查验证仓库脚本用了 `bash -x`，脚本 `source` 了 `paratera.env`，API key 明文进了会话输出（2026-09-18 20:42） | `-x` 回显 `source` 的每一行赋值 | **不要对读 env 文件的脚本用 `bash -x`**；要跟踪就先 `export ENV_FILE=/dev/null` 或在脚本里 `set +x` 包住 `source`；事后换 key |
| 看 `~/.codex/config.toml` 时用 `sed` 遮值，只遮了 `base_url` / `env_key` / `api_key`，`experimental_bearer_token = "sk-…"` 整行打进了会话输出（2026-09-19） | 遮值靠黑名单，漏一个键名就漏一把 key | 读任何可能含 token 的配置只 `grep -o '^[a-z_]*'` 打键名，或 `grep -v -i 'token\|key\|secret'`（白名单/整行丢弃），不做逐键替换；漏了就当泄露，让 owner 轮换 |
| `DEEPCODE_WORKSPACE=<路径>` 一设，DeepCode 加载配置就报 `error parsing value for field "workspace"` | pydantic-settings 以 `DEEPCODE_` 为前缀读环境变量，同名变量被当成配置对象 | 基线运行不设它，工作区用 cwd 默认；本线 `DEEPCODE_*` 只设 PLAN §6 那些引擎旋钮（`apply_env_defaults`） |
| DeepCode 的 `filesystem` / `fetch` MCP 一连就 `Connection closed` | `npx -y` 首次解析包 20 s+ 撞连接超时；`uvx mcp-server-fetch` 重编 cryptography；filesystem 服务器要求允许目录已存在 | 基线 `setup.sh` 装成固定路径（`.mcp-node/`、`DeepCode/.venv/bin/mcp-server-fetch`）并先建 `deepcode_lab/`；本线不起 MCP 进程 |
| MCP 客户端日志里冒出 7 条 "Failed to parse JSONRPC message"（内容是 npm 安装输出） | 某个 stdio 服务器的 stdout 混进了 npm 文本，客户端跳过 | 无害；见 HANDOFF sapg 基线运行记录 |
| `pkill -f "xxx"` 把自己杀了（exit 144）；改运行中的脚本错位执行 | 匹配到自己；bash 逐行读脚本 | `pkill -f "xx[x]"`；运行中的脚本不改 |
| 后台等待用 `sleep` 被工具拦；`grep RUN_EXIT` 误匹配 `RERUN_EXIT` | 工具限制；子串 | Monitor / `until` 循环；用 `^FINAL...` 锚点 |
| 另一个 Claude 会话在同一 worktree 上 `release`、重折提交 | 并行会话 | 动手前 `git log`，不假设记忆里的 sha 还在；不用裸 `git stash` |
| 老任务目录混入新轮 / 拿错论文摆卷 | `deepcode_lab/tasks` 未清、交接文件跨论文 stale | 本线一 run 一目录（`run.json` 冻结 `paper_sha256`，`intake` 校验）；基线 `run_trial.sh` 开跑前归档全部 `paper_*`、按论文分交接文件 + 标题核验 |
| 判分池残留已判副本，重判白花钱 | 判完不清池 | 判完即归档，`~/pb_submissions/<paper>/` 保持空 |
| 三方输入不对标（DeepCode 缺 addendum、本线只有 pdf、裸跑提示带偏） | 各拿各的 | 输入标准（验证仓库 `docs/INPUT_STANDARD.md`）：五样材料字节一致；本线 `intake.compose_input` 与 `run_trial.sh` 的并稿逐字节相同（sapg 两边 sha256 `04790c3f…`） |
| 本地 Ollama（qwen3:4b）默认 `num_ctx=4096`，系统提示被截、模型无限绕圈 | 无 GPU 机器的默认值 | `OLLAMA_CONTEXT_LENGTH=16384`，超时放宽到 15–30 分钟（`deepcode-env.sh`）；只作备用 |
| 容器里 `pip install torch` 554 MB 走了 25 分钟（0.25 MB/s），此前 CPU torch 也是 1–2 MB/s；同一容器 `curl` 同一 URL 14 MB/s | `mirrors.aliyun.com` 对 **HTTP/1.1**（pip 只会这个）限速到 ~0.25 MB/s，HTTP/2（curl 默认）才快；cn-hongkong 机器实测 2026-09-18；tuna 与 download.pytorch.org 的 HTTP/1.1 都是 12 MB/s | `Dockerfile.setupx-base` 与 `job_executor.PIP_INDEX_URL` 换成 `mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`，GPU 镜像重烘；SetupX 事实清单写明索引与测速；排查法：`curl --http1.1 -w %{speed_download}` 对比默认 |
| 试跑失败明明是代码（`ModuleNotFoundError: sapg.networks`、`ValueError`、CLI 参数错），调度器却把下一轮交给 SetupX，两轮后签名重复停机 | 产物测试的尾巴带 `FileNotFoundError …/checkpoint_final.pt`（跑挂了自然没产物），命中"资产缺"正则；编译器还会把产物路径写到 `/workspace/repo/figures/…` 之类，输出目录前缀过滤不到 | `failure_text` 只拿非产物测试的尾巴做归因；`classify` 本仓模块缺 → 代码；`is_output_path` 认 `workspace/out` 两种写法；编译器把输出目录当资产 → G1 前丢掉、试跑前 `mkdir -p` |
| 修复轮之后 SetupX 报"container is not running"、试跑 `GRADE_ERROR: not a git working tree`、checkpoint "No such container" | RSA 的 `rollback_to_checkpoint` 是 `docker rm` 旧容器 + 从快照 `run` 新容器 → 容器 id 变了，调度器还拿着旧 id；回滚失败时它只返回 False、不留原因、backend 里没容器 | `RepairResult.container_id` 回传新 id；回滚失败记 `repair.rollback_failed`（带 daemon 视图）并按 `container_lost` 重建环境；`GRADE_ERROR` 一律重建环境 |
| 修复 agent 一轮 40 次调用花了 28–34 次在探针上，第二轮一个文件都没写 | 探针便宜、写文件贵，模型倾向"再看一眼" | `MAX_PROBES=15`：超过即拒并提示"写文件然后 finish"；提示词写明预算 |
| S9 六个修复轮：`main.py` 每轮整文件重写 2–4 次（20–30k 字符），`algorithm.py` 被覆写成 2k 片段（−388 行）、下一轮再补 845 行；树里多出 `patch.py`、`main.py.flatpatch`、`_patch_import.txt` 并进了产物 | agent 只有整文件 `write_file`，没有编辑工具，于是自造补丁机制；截在中途的覆写让试跑报它自己的 `NameError` | T2：`edit_file(old_string→new_string)` 唯一精确替换（编辑前必须读过，`read_text_file` 带行号）；`write_file` 只建新文件、扩展名白名单；`.py` 写入先 `ast.parse`，不通过则一字不动 |
| S9 六轮没有一轮调用 `finish`，全在第 40 次调用被截断，试跑验的是 agent 停笔那一刻的半成品 | 模型不数调用次数；循环到预算即停并提交当前树 | T2：第 30 次调用的工具返回值带"还剩 10 次，现在 finish"；round 记录 `finished`、`probes_after_write` |
| 修复轮开场连打 15–17 次探针（前 15 次成功、后 2 次被拒）才读第一个文件；改完后没有探针可验 | 提示只给失败尾巴，模型先用探针"看现场" | T2：提示改为读→改→验的顺序、留 3 次探针给验证；提示里直接附回溯各帧的源码 ±20 行（`traceback_excerpts`） |
| 判据命令带代码没有的参数（`--pbt_interval`、`--num-blocks`），两次 S9 的第一轮修复都花在 `unrecognized arguments` 上 | RSA 编译器只看目标句，入口的参数表它看不到，就按经验编；伪证器只校验路径不校验参数 | T2：本线用 `ast` 静态抽入口的 `add_argument` / `add_parser`，目标句写明"只有这些参数，拼写照抄，不要发明"（`experiment_step.entry_flags`） |
| 参数表进目标句之后，编译器改编**值**：`--task dummy`（`choices=sorted(TASK_CONFIGS.keys())` 里没有 dummy），SetupX 跑判据命令直接 `invalid choice`（T2 第一次真机，11:40，杀掉重来花了 35 分钟机器） | 编译器看不见 `choices` | `entry_flags` 静态解析 `choices=`（本文件字面量，或指向本模块/被 import 模块的模块级 dict/list 字面量，穿过 `sorted/list/tuple/.keys()`），目标句里 `--task {regrasping,throw,…}`，"括号里是可选值，只能取其一" |
| SetupX 装完依赖后花 20 分钟改代码：容器里用 python heredoc 改 `main.py`/`rollout.py`/`algorithm.py`、`git stash`/`pop`（T2 真机 11:57–12:14，66 步） | 判据命令失败在代码 bug，SetupX 不知道有修复 agent；它的改动在试跑前被 checkout 到冻结提交丢掉（验证过：round 1 试跑的失败与 SetupX 第一次看到的相同） | 白花的是时间不是正确性。**owner 09-18 下午：SetupX 不改、addendum 不加**（黑盒保持黑盒）；两段式算力让这段时间落在 1.39 元/h 的机器上 |
| 修复 agent 第 30 次调用收到"现在 finish"仍继续读改，到第 40 次被截断（T2 真机第 2、3 轮） | 模型在调查中途不会为了提醒停手 | 第 40 次调用只允许 `finish`（其他工具拒绝且不计数，runner 的迭代上限兜底） |
| T2 真机 46 分钟全程在 9.53 元/h 的 T4 上，判据命令却是 `--device cpu`，GPU 一秒没用；环境轮 19 分钟里 8 分钟下 3 GB CUDA 轮子 | compute 阶段按 spec 的 `needs_gpu`（论文全量实验的需要）默认 GPU 档 | 两段式算力（owner 09-18 下午）：默认最便宜 CPU 档；试跑/SetupX/修复 agent 给出"需要 GPU"的证据时（`repair.GPU` 类失败、SetupX FINISH 说 GPU-only、agent 归因环境且说需要 GPU）控制器以 `gpu required` 停机，阶段释放 CPU 机、租 `escalation_type`（最小 T4）、在当前提交上重开搭建环境（`phases._escalate_to_gpu`、`10_environment_run.escalation.json`）；GPU 机上同一信号归代码 |
| 复制运行目录做实验（`cp -R sapg-s9-off sapg-t3-esc`），跑的却是原目录的树 | `phases/*.json`、`dir_info.json`、`status.json` 里的路径是绝对的，副本照用原目录 | 09-19 起 `paper2code_canary.py relocate --run-dir <副本>`：从 `dir_info.json` 读出旧根，改写 13 个文件（phases、dir_info、status、environment*、submission、任务目录 dir_info）；`rerun --phase environment_run` 顺带把树重置到 round 0（`--keep-tree` 不重置），第 10 步就能在副本上重验 |
| 为验证升级边在代码里加 `args.device="cuda"`，CPU 上照跑 | 生成代码有 `if torch.cuda.is_available() else "cpu"` 回退 | 要触发 CUDA 类失败得无条件调用（`torch.zeros(1, device="cuda")`）或 import 时的门 |
| 修复 agent 两次把无条件的 CUDA 门删掉（summary："removed the spurious CUDA hard-gate"），升级边没机会触发 | agent 在探针里先看到门，判断为多余；提示词里"CUDA-only build → finish(environment)"的规则它没用 | 升级边只在**试跑**先暴露 CUDA 失败时成立；agent 先看到就会当 bug 修——真需要 GPU 的论文可能被改成 CPU 版，记入 T3c 讨论 |
| G0 的"声明的 import"（`experiments`）从仓库根解析，代码在 `sapg/experiments/`；有的轮 SetupX 用 `SET_ENV PYTHONPATH=/workspace/repo/sapg` 盖过去，有的轮没有，G0 就挂在这里，G2 永远跑不到（T3b 第四、五、六次） | RSA 编译器按仓库根列 import，不看嵌套布局；SetupX 是否补 PYTHONPATH 随机 | 未修：候选是目标句加一条静态事实"顶层可导入包 = …；`sapg/` 下的模块以 `sapg.<x>` 导入"（与 `entry_flags` 同类），记 T3d |
| CPU 段之后同进程的 GPU 段，SetupX 读到"Accelerator: none"，在 T4 上装了 CPU torch | 机器事实 addendum 每进程只写一次（有标记就跳过），且 SetupX 的 `llm_engine` 模块已在 `sys.modules` 里 | `append_setupx_addendum` 事实不同就重写文件**并**替换在内存里的类属性（`fd797457a`，用例钉住） |
| G0 同时报 `experiments` 缺失和 `CUDA unavailable`，被归为"代码"送去 CPU 上修 | `classify` 把"自己模块缺失"排在 CUDA 之前 | CUDA 类失败现在排在自己模块缺失之前（`324234fe1`）：机器不对时别的错在 GPU 上还得再修一遍 |
| T4 预装 torch 后，pinn 两条本线的判据被 RSA 伪证器拒绝两次（"G0 passes in the bare environment"），第 10 步开不起来 | 伪证第 1 点 bare = 判据在**未配置**的 `setupx-base` 容器里跑一遍，过了就判"没量到环境"；编译器给 G0 的 import 清单是从入口 import 推的（`torch` + 自身包），不看目标句；torch 进了镜像，两项裸容器都能 import | T4 改 opt-in、换回无 torch 镜像；正解是 T4b wheelhouse（轮子在镜像里、不安装）；目标句里的 requirements 清单留着但对编译器无效 |
| 修复轮之后把代码传到机器失败：`GitDaemonError: 把代码传到机器上失败：connection lost while uploading …/repo.bundle`（pinn-on，22:20），控制器以"repair worker failed"停机 | 本机到 ECS 的 SSH 传输中断（同一时间 `pinn-off` 正常），本机网络抖动 | 09-19 修：`experiment_step.default_serve_repo` 用 `with_retries`（3 次，间隔 5 s / 20 s，每次新建 runtime）包住 bundle 上传 + daemon 起动；仍失败才抛。`_push_worktree`（探针前的小文件同步）走容器 exec，不在这条路径上 |
| 生成树里有语法错误（fre-t14 `prior.py:503` `torch.rand(..., device="cpu", device=device)`；`trainer.py:420` 调了不存在的方法），裁判逐叶扣 | 写码 agent 每个文件写一次、从不运行；重复关键字这类错 `ast.parse` 过得去、只有 `compile()` 拦得住 | 09-20 晚起 `syntax_check.py`：implement 后 `compile()` 全树（不执行），≤ 2 轮 `edit_file` 修复（无容器），`DEEPCODE_SYNTAX_CHECK=0` 关；看 `implement.syntax` 事件。属性不存在 / 签名不对这类不执行看不出来 |
| 让 planner 在蓝图里逐字抄论文（引文 / 锚点 / 义务表）并由 host 校验（ADR 0003，09-19 夜 → 09-20 夜） | 50 篇只跑 plan：第一批 12/12 失败，全是标签层；放宽后仍有定位、目录和 YAML 结构问题；这一层判断既不稳定，也没有增加可验证的科学事实 | ADR 0004：蓝图只写文件路径和 `Source: §x.y`，host 读回；写前逐节、逐页读原文。用 Read receipt 证明暴露过原文，不要求 planner 重抄论文 |
| 写码 agent 写出的公式和论文对不上（SA-Bench 09-19：deepcode 对 basic 的差距全在公式维度；robotic-world-model 把 L1 写成 L2、ma-rlhf 逐 token 比率、wdno 幅值分布丢失） | 蓝图第 2 节被压成摘要，写码 agent只看蓝图；原有 `read_document_segments` 没有接到写码阶段 | T14 / ADR 0004：`read_paper(file_path)` 自动列出该文件所有绑定章节的未读页；写文件前必须读完，审计记录覆盖率。看 `plan.source_manifest`、`implement.paper_reads` 和 `implement.paper_fidelity` |
| pinn 三份树（基线也是）`opt_for_pinns/src/pdes.py` 0 字节，G2 一开始就 `ImportError` | DeepCode 写码循环把内容写到了顶层 `src/pdes.py`，包内同名文件留空；`implementation_status` 只数文件数 | T13 已做：`repair.empty_files` 列出 0 字节源文件 + 同名非空孪生；`implementation_status.detail.empty_files` 记录（不判失败，事件 `implement.empty_files`），修复轮提示里点名。判 stage-9 树前先看这一项 |
| 想用阿里云 1/12 卡（`sgn7i-vws` 2.51 元/h）省钱 | 分片 GPU 只在内地 region 售卖（香港整个 region 不售），且要 GRID vGPU 驱动，与本线镜像不兼容；内地机器访问 GitHub/HF 另有网络问题 | 记为 T4b，不做；省钱大头是 CPU 先跑 + 镜像预装 torch |
| G2 通过靠的是修复 agent 把产物**同时**写到仓库相对路径 `/workspace/repo/workspace/out/smoke/…`（T2 第 4 轮，main.py +47/−21，summary 写明白了） | 编译器把产物写成仓库相对路径（T3 的事），代码按 `--output-dir` 写绝对路径，判据找不到 | T3（09-18 下午）：冻结前本线把 ladder 里 `workspace/out/…` 相对产物改成绝对路径、删掉与绝对产物同名的相对副本（`repair_loop.normalise_ladder_artifacts`，运行期替换 `rsa.pipeline.Freezer`，RSA 不动） |

## 09-22 新增

| 坑 | 现象 | 处理 |
| --- | --- | --- |
| 官方 paper.md 缺章（robust-clip） | md 在 §1 句中截断，§2–§4 开头没有；planner 14 条 `Source:` 全指 `§Addendum`，写码只读 addendum，同口径 0.642 vs Codex 0.906 | 从 arXiv LaTeX 源补全（`paperbench/robust-clip/hydration/`），补全后 1.000。跑一篇新论文前先 `grep -n '^\\section' paper.md` 看顶层编号是否连续 |
| 参考仓库预筛退化成全量分析 | adaptive-pruning 的 LoRA 仓库 930 文件，index 阶段 3.5 h（每分钟 3 个文件），拖住整波 | `index.prefilter_fallback` 事件会标；大仓库多的论文单独起、不放进波次；或 `--skip index` |
| 并发判分时 `run_grade.sh` [4/4] 拷错组 | 取 `ls -t runs/ | head -1` = 任何论文最新的运行组 → 别的论文刚起的空组 → "NO grade.json" 而分其实在 | 已改成取本论文最新组（`5890c13`）；夜间脚本 v4 直接从 nanoeval 运行目录按时间标记收分 |
| 一次只能放一棵树进池 | 判分按 `PAPER` 扫整个池；同论文两臂同时在池里则 run_id ↔ 臂只能靠 tar 内容反推 | 每棵树单独放、判完挪走（`grade_night_v4.sh` 的 staging / graded 目录） |
