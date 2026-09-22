# Paper2Code 线嵌入 DeepEvol：实施步骤

分支 `Paper_repro_0916`，从 `main` `bf1dbae8` 起。本文是施工清单：每一步做什么、从哪拷到哪、改什么、怎么验、提交边界在哪。决策全部来自 2026-09-16 的两轮访谈，不再重议；有争议的点在第 9 节单独列出。

本文进分支后放在 `DeepEvol1.0/apps/v2/agent/paper2code/PLAN.md`。

## 0. 决策摘要

| 项 | 决定 |
| --- | --- |
| 分支 / 工作树 | `Paper_repro_0916` ← `main@bf1dbae8`，第三个 worktree `search/DeepEvol-Paper_repro_0916` |
| 产物线名 | `paper2code` |
| kernel 入库方式 | vendor 拷贝到 `apps/v2/agent_engine/paper2code/`，记录来源提交；`search/paper2code-kernel` 归档 |
| 范围 | 第 1 步 DeepEvol 侧做；kernel 覆盖 3 到 8 加验证；2 透传不用；9、11 桩；10 只有验证 |
| 不做 | 前端、Product 表与迁移、判分、Gateway 与计费准入、A3 自建循环、嫁接 reproduction-c |
| 工具 | 进程内，七个服务器全部不起子进程 |
| 模型 | httpx 直连 Paratera，`DeepSeek-V4-Flash`，`thinking: {"type": "disabled"}` 固定发出，`reasoning_tokens` 必须为 0 |
| 循环 | INTEGRATION.md 的参考循环，不用 LangGraph |
| 执行 | `execute_python` / `execute_bash` / 验证走 `remote_relay.Runtime`；每 run 租一台 Aliyun，首次执行时租、run 结束必释放 |
| 工作区 | 文件工具写本地任务目录；执行前同步 `generate_code/` 到远端作业目录，一次性容器断网跑，产出同步回 |
| 驱动 | file-backed，每个表格阶段一个 phase，可单独重跑；四道闸门全搬 |
| 补丁 | 叠加验证仓库补丁的"带走"子集，默认值取 `run_trial.sh` |
| 验收 | EMA-Detect 冒烟，然后 sapg 走完 1 到 8 加验证；判分推迟 |
| 词汇 | DeepEvol 根 `CONTEXT.md` 加 Paper2Code line 与十一个 phase；一条 ADR |

## 1. 事实基线

| 事实 | 值 |
| --- | --- |
| DeepEvol main HEAD | `bf1dbae8`，2026-09-11 |
| kernel 来源 | `search/paper2code-kernel` @ `c821130`，上游 HKUDS/DeepCode `21ebc57f` |
| 验证过的 DeepCode 基线 | HKUDS `e0767d0` + `patches/deepcode_local_changes.patch`（`2UBBISH/deepcode-paperbench-validation`）；`e0767d0` 是 `21ebc57f` 的祖先，隔 84 个提交 |
| 补丁碰过、上游后来又改的文件 | `tools/code_implementation_server.py`、`tools/command_executor.py`、`workflows/code_implementation_workflow.py` |
| 主干可复用 | `apps/v2/remote_compute/providers.py::AliyunControlClient`、`apps/v2/remote_compute/secrets.py::FileRemoteComputeSecretStore`、`apps/v2/reliability/ssh_channel.py` |
| reproduction-c 可搬零件 | `apps/common/reproduction_paper_bundle.py`、`Agent/remote_relay/`（vendored，`asyncssh`）、`paper_reproduction_agent/canary/{aliyun_lease,remote_daemon}.py`、`Agent/DeepEvol/experiment/{lease,release_policy,lease_api}.py`、`tests/v2_reproduction/test_footprint.py` + `footprint.yaml`、`canary_gateway.py::SiliconFlowChat`（只作参考） |
| reproduction-c 不可直接搬 | `aliyun_lease.py` 依赖已删除的 `apps/api/app/services/aliyun_ecs.py`（1492 行）；`docker_canary_executor.py` 依赖十几个 `reproduction_*` 契约 |
| 思考开关实测 | Paratera + V4-Flash：只有 `thinking: {"type": "disabled"}` 有效（reasoning_tokens 0）；`enable_thinking: false` 被无视 |
| 主干缺的依赖 | `loguru`、`mcp`、`aiohttp`、`aiofiles`、`asyncssh`；已有 `httpx`、`paramiko`、`pyyaml`、`pypdf` |
| Python | 主干 `>=3.11`；kernel 按 3.12 写，C1 要在主干解释器下验一次 |

## 2. 目标布局

```
DeepEvol1.0/
├── apps/v2/agent_engine/paper2code/          引擎（vendor，除第 4 节的补丁外不改）
│   ├── VENDOR.md                              来源提交、导入重写、补丁清单
│   ├── workflows/  tools/  prompts/  utils/   kernel 四目录
│   ├── seams/  support/                       kernel 的接缝契约与支撑代码
│   └── CONTEXT.md  docs/                      kernel 自带文档，只管引擎内部
├── apps/v2/agent/paper2code/                  这条线的一切实现
│   ├── PLAN.md                                本文
│   ├── adr/0001-separate-engine-not-a-graft.md
│   ├── footprint.yaml
│   ├── config.py                              KernelConfig 装配、run.json 冻结
│   ├── provider.py                            httpx 直连 provider（接缝 LLMProvider）
│   ├── runner.py                              AgentRunner.run 参考循环
│   ├── agent.py                               Agent.__aenter__/__aexit__/attach_llm、AugmentedLLM.generate
│   ├── tools/                                 进程内工具注册
│   │   ├── registry.py                        server 名 → Tool 列表，名字消毒
│   │   ├── kernel_servers.py                  包装 tools/*.py 的函数
│   │   ├── filesystem.py  fetch.py            两个外部服务器的薄实现
│   │   └── execute.py                         execute_python / execute_bash → 执行端口
│   ├── execution/
│   │   ├── port.py                            ExecutionPort 协议、作业记录
│   │   ├── remote_relay/                      vendored（来自 reproduction-c，含 DEEPEVOL_VENDOR.md）
│   │   ├── job_executor.py                    同步工作区 → docker run → 收产出
│   │   ├── lease.py  release_policy.py        搬自 experiment/
│   │   ├── aliyun_lease.py  remote_daemon.py  搬自 canary/，改用主干的 AliyunControlClient
│   │   └── leased_runtime.py                  首次 exec 时租、退出时释放
│   ├── intake.py                              论文包 → paper.md + Addendum、黑名单
│   ├── gates.py                               四道闸门
│   ├── phases.py                              十一 phase 与 kernel 步骤函数的映射
│   ├── driver.py                              run 目录、phase 记录、重跑
│   └── verification_hook.py                   验证命令改走执行端口
├── scripts/paper2code_canary.py               CLI
└── tests/v2_paper2code/
```

run 目录（`runs/<name>/`）：

```
run.json            冻结配置：论文包路径与 sha256、模型、provider、算力档、小时上限
status.json         当前 phase、各 phase 状态、闸门结果
events.jsonl        一行一个事件
input/paper.md      paper.md + "# Addendum" 节（sha256 记入 run.json）
workspace/          kernel 的 workspace_root；任务目录在 workspace/.deepcode/workflows/tasks/paper_<run-id>/
phases/<nn>_<name>.json   每次 phase 尝试：开始/结束、输入 sha、kernel 返回值、错误
llm/<seq>.json      每次模型调用：请求（无密钥）、响应、usage、reasoning_tokens
jobs/<seq>/         每次远端作业：命令、退出码、stdout/stderr 尾、同步清单
lease.json          机器：实例 id、规格、租/停/释放时间线
canary.log
```

## 3. 分步实施

每步一个提交。步内先列"做什么"，再列"验证"。除非注明，路径都相对 `DeepEvol1.0/`。

### C0 · 分支、工作树、词汇、ADR、本文

做什么：

1. `cd search/DeepEvol && git worktree add -b Paper_repro_0916 ../DeepEvol-Paper_repro_0916 main`。
2. 根 `CONTEXT.md` 的 "Artifact lines" 节加一条 **Paper2Code line**（以 DeepCode 引擎复现论文的产物线；与 reproduction 线并列，不共享阶段词汇），再加一段 "Paper2Code phases"：`intake` `criteria` `plan` `plan_review` `references` `acquire` `index` `implement` `compute` `environment_run` `optimize`，每个一句话。写明：kernel 自带的 `CONTEXT.md` 只管引擎内部，"步骤"在本仓语境下叫 phase。
3. `apps/v2/agent/paper2code/adr/0001-separate-engine-not-a-graft.md`：决定 = Paper2Code 线是独立引擎加薄适配器，不嫁接进 reproduction 线；备选 = 嫁接进 reproduction-c 的 Stage 5/8/9 执行器、或作为它的第二个 program adapter；理由 = 那条线"模型不持工具、不本地执行"的硬规则与 DeepCode 循环互斥，且它的迁移号与 main 冲突、尚未跑通真论文；后果 = 仓库里并存两条复现线，各自评分，谁吸收谁等分数出来再定。
4. 本文拷到 `apps/v2/agent/paper2code/PLAN.md`。

验证：`git worktree list` 三行；`git log -1` 在新分支。

### C1 · vendor 引擎

做什么：

1. `cp -r search/paper2code-kernel/{workflows,tools,prompts,utils,seams,support,CONTEXT.md,docs,README.md,UPSTREAM_COMMIT} apps/v2/agent_engine/paper2code/`，删 `.git`、`.gitignore`、`__pycache__`。
2. 导入重写（照 PaperOrchestra 先例，不用 `sys.path`）：把六个顶层包的绝对导入改为全路径。

   ```bash
   cd apps/v2/agent_engine/paper2code
   P=apps.v2.agent_engine.paper2code
   grep -rlE "^\s*(from|import) (workflows|tools|prompts|utils|seams|support)\b" --include='*.py' . \
     | xargs sed -i '' -E "s/^(\s*)from (workflows|tools|prompts|utils|seams|support)\b/\1from $P.\2/; s/^(\s*)import (workflows|tools|prompts|utils|seams|support)\b/\1import $P.\2/"
   ```

   缩进的懒导入（`environment.py`、`agent_orchestration_engine.py`、`llm_utils.py` 里的 `from seams...`）靠 `^\s*` 一起覆盖。`seams/mcp_servers.json` 里的 `{KERNEL_ROOT}` 占位不动。
3. 依赖：`pyproject.toml` 主依赖加 `loguru>=0.7`、`mcp>=1.29,<2`、`aiohttp>=3.9`、`aiofiles>=23`、`asyncssh>=2.14,<2.22`（remote_relay 用；上界照 reproduction-c 的说明，`cryptography` 的 macOS wheel 问题）；`tiktoken` 不加，kernel 缺失时退化。`uv lock`。`mcp` 只作为库导入，不起进程，V2 的"无 MCP 进程"约束不受影响，在 VENDOR.md 写明。
4. `VENDOR.md`：来源提交 `c821130`（上游 `21ebc57f`）、导入重写命令、依赖、后续补丁的逐条清单（C2、C5 追加）。
5. kernel 的 `seams/__init__.py` 文档字符串里的"十个契约方法"清单原样保留，本线的实现位置在 C3 到 C5 注明。

验证：

```bash
# 逐模块导入，不得加载任何 DeepCode core.* 或本仓 Agent.* 模块
python - <<'EOF'
import importlib, pathlib, sys
root = pathlib.Path("apps/v2/agent_engine/paper2code")
for p in sorted(root.rglob("*.py")):
    m = ".".join(p.with_suffix("").parts).removesuffix(".__init__")
    importlib.import_module(m)
bad = [n for n in sys.modules if n == "core" or n.startswith(("core.", "Agent."))]
assert not bad, bad
print("ok")
EOF
```

在主干解释器（`>=3.11`）下跑；kernel 用到 `match`（3.10）与 `X | None`，无 3.12 专属语法，若报错在此修。

### C2 · 叠加验证仓库补丁的"带走"子集

来源：`2UBBISH/deepcode-paperbench-validation` `patches/deepcode_local_changes.patch`（基于 `e0767d0`）。逐 hunk 取舍：

| 补丁 hunk | 取舍 | 落到哪 |
| --- | --- | --- |
| `core/agent_runtime/tools/mcp.py`：URL 黑名单 `DEEPCODE_URL_DENYLIST`、重复抓取账本、工具名 `-`→`_` 消毒 | 带走 | 本线 `tools/registry.py`（消毒）、`tools/fetch.py` 与 `kernel_servers.py` 的 clone 包装（黑名单、账本）。kernel 无此文件 |
| `core/compat/agent.py`：tool_filter 按消毒后前缀匹配 | 带走 | `seams/compat.py::apply_tool_filter`（引擎内，一处 `.replace("-", "_")`） |
| `core/compat/request_params.py`、`core/providers/base.py`：`DEEPCODE_LLM_RETRY_MODE`、`DEEPCODE_CHAT_RETRY_DELAYS`、`DEEPCODE_PERSISTENT_MAX_DELAY`、`DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT` | 带走（语义） | 本线 `provider.py` 的重试策略（C4） |
| `tools/code_indexer.py`：预筛 `DEEPCODE_PREFILTER_MAX_TOKENS`；分析/关系 `DEEPCODE_ANALYSIS_MAX_TOKENS`、`DEEPCODE_RELATIONSHIP_MAX_TOKENS` | 带走 | 引擎 `tools/code_indexer.py` |
| `utils/loop_detector.py`：连续 `write_file` 不算循环 | 已被 kernel 的"写工具按参数分键"补丁覆盖 | 不动 |
| `workflows/agent_orchestration_engine.py`：参考挖掘 `DEEPCODE_REFERENCE_MAX_TOKENS`、下载 `DEEPCODE_DOWNLOAD_MAX_TOKENS`、下载 agent 换成工具优先的提示 + `tool_filter={"github-downloader": {"git_clone"}}` + 一次纠正重试 + 零工具调用即失败 | 带走 | 引擎，`max_iterations` 改为 env（`DEEPCODE_REFERENCE_MAX_ITERATIONS` 默认 8、`DEEPCODE_DOWNLOAD_MAX_ITERATIONS` 默认 8），不写死 40/80 |
| 同文件 fix-①：`DEEPCODE_PLAN_COVERAGE_CHECK` | 不带（提示词含评分元知识） | — |
| `workflows/agents/document_segmentation_agent.py`：相信产物不相信 agent 的话 | 带走 | 引擎 |
| `workflows/agents/memory_agent_concise.py` fix-②：`DEEPCODE_ALLOW_PLAN_EXTENSION` | 不带 | — |
| `workflows/code_implementation_workflow.py`：`DEEPCODE_MAX_WALL_SECONDS`、`DEEPCODE_STALL_THRESHOLD` env 化 | 带走 | 引擎；**上游此文件在 `e0767d0` 后有 11 行改动，手工合** |
| 同文件 fix-③：`DEEPCODE_POSTWRITE_COMPILE` | 不带（会编译到源文件） | — |
| `workflows/codebase_index_workflow.py`：f-string 反斜杠兼容 | 带走 | 引擎 |
| `tools/code_implementation_server.py`、`tools/command_executor.py` | 补丁未碰这两个文件；上游后改的是沙箱与命令筛查，kernel 已含 | 不动 |

默认值取 `run_trial.sh` 的注入值，写进 `config.py` 的常量表（第 8 节附表），环境变量仍可覆盖。

验证：`tests/v2_paper2code/test_engine_patches.py`：黑名单命中拒绝、工具名消毒、下载 agent 零工具调用抛错（用假 provider）。

### C3 · 进程内工具

做什么：

1. `tools/kernel_servers.py`：对每个引擎服务器文件，import 模块后把 `@mcp.tool()` 函数（FastMCP 的装饰器返回原函数）包成 `seams.agent_runtime.Tool`：`name` 取消毒后的 `mcp_<server>_<tool>`（`code-implementation` → `mcp_code_implementation_read_file`），`description` 取 docstring，`parameters` 由函数签名生成 JSON Schema（`inspect.signature` + 注解，`List[str]` → array/string），`execute` 直接 `await fn(**kwargs)`。`command_executor.py` 用的是低层 `Server`，包 `handle_call_tool("execute_commands", args)`。工作目录：`code_implementation_server.WORKSPACE_DIR` 由 kernel 在 `_initialize_mcp_agent` 里调 `set_workspace` 设置，进程内同样生效；`document_segmentation_server.DOCUMENT_INDEXES` 是模块级缓存，一 run 一进程无冲突。
2. `tools/filesystem.py`：`read_text_file(path)`、`list_directory(path)`，根目录 = run 的 workspace，越界拒绝。
3. `tools/fetch.py`：`fetch(url, max_length=...)`：复用引擎 `tools/pdf_downloader.py` 的 `_validate_public_url` / `_SafeResolver` / 重定向策略，加黑名单与重复抓取账本，HTML 转文本用 `tools/document_conversion._html_to_markdown`，正文上限 100 KiB。
4. `tools/execute.py`：`execute_python(code, timeout)`、`execute_bash(command, timeout)` 注册为 `mcp_code_implementation_execute_python` / `_execute_bash`，实现走 `execution.port.ExecutionPort`（C5）；命令先过引擎 `support/command_guard.screen_command`。
5. `tools/registry.py`：`build_registry(server_names, *, port, denylist, workspace) -> ToolRegistry`，服务器名映射表：

   | server | 来源 |
   | --- | --- |
   | code-implementation | kernel 函数（除两个 execute 换成 `tools/execute.py`） |
   | command-executor | kernel `handle_call_tool` |
   | document-segmentation | kernel 函数 |
   | code-reference-indexer | kernel 函数 |
   | github-downloader | kernel 函数，`git_clone` / `download_github_repo` 外加黑名单检查 |
   | filesystem | `tools/filesystem.py` |
   | fetch | `tools/fetch.py` |

   未知服务器名抛错，不再静默跳过。
6. `agent.py`：`Agent.__aenter__` = `build_registry(self.server_names, ...)` 后逐个 `register_tool`；`__aexit__` 清空注册表；`attach_llm` = `get_runtime().provider_for(phase=...)` → `PaperAugmentedLLM`；`AugmentedLLM.generate` = `build_run_spec` + `PaperAgentRunner(provider).run`。

验证：`test_tools_registry.py`：七个服务器全部可构建、名字全部匹配 `^[a-zA-Z0-9_]+$`、`build_aliased_registry` 按裸名找得到 `write_file` / `read_code_mem` / `execute_python`；`test_fetch_policy.py`：私网 / 黑名单 / 重复抓取；`test_kernel_servers.py`：`write_file` 后 `read_code_mem` 能切出摘要（用临时工作区）。

### C4 · 模型接缝与循环

做什么：

1. `provider.py::ParateraProvider(LLMProvider)`：
   - httpx 客户端，`POST {base_url}/chat/completions`，OpenAI 兼容；消息、`tools`（函数 schema）、`tool_choice`、`max_tokens`、`temperature` 原样；**固定 `thinking: {"type": "disabled"}`**；`stream` 可选（Paratera 会截断长回复时开，默认按 `run.json` 的 `provider_stream`）。
   - 解析：`choices[0].message.content` / `tool_calls`（arguments JSON 解析，失败→`finish_reason="error"`）/ `finish_reason`（`length` 原样）；`usage` 含 `completion_tokens_details.reasoning_tokens`；**非零即抛 `ThinkingNotDisabled`**，run 作废。
   - 重试：`retry_mode="standard"` = 延迟 `(1, 2, 4)`；`"persistent"` = `DEEPCODE_CHAT_RETRY_DELAYS`（默认 `10,30,60,180,300`）循环到 `DEEPCODE_PERSISTENT_MAX_DELAY`（900）封顶，连续相同错误 `DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT`（30）次收手；请求超时 `DEEPCODE_OPENAI_REQUEST_TIMEOUT_S`（600）；重试只针对网络错误、5xx、429、空回复；`on_retry_wait` 回调透传。
   - 每次调用落盘 `llm/<seq>.json`（请求头不含密钥），并 `events.jsonl` 记 `llm.call`。
   - 密钥只从 `run.json.provider_key_env` 指定的环境变量读，脚本用 `--env-file` 注入；不进 run 目录。
2. `runner.py::PaperAgentRunner(AgentRunner)`：按 INTEGRATION.md §4 逐条实现：`should_stop_callback` 每轮前、`injection_callback` 无工具调用时、hook 顺序、权限 → 审批、结果截断 `max_tool_result_chars`（头尾保留）、`max_iterations` 耗尽先试注入、`finish_reason == "error"` 结束、空回复重试两次、`length` 续写最多三次。**`after_iteration` 之后必须用 `context.messages` 当前对象**（代码记忆原地替换）。
3. `config.py`：`build_kernel_config(run) -> KernelConfig`：`agents.defaults.model` = run 的模型，`planning` / `implementation` 不覆盖（起跑自检要求同模型）；`max_tokens = 32768`；`workspace.root` = run 的 workspace；`tools.mcp_servers` 留空（进程内不用）；`security.permission_mode = "full_auto"`。`PaperKernelRuntime(KernelRuntime).provider_for` 返回同一个 `ParateraProvider` 实例（按 phase 只改 `generation` 的 max_tokens / temperature）。`build_permission_engine` → `FullAutoEngine`；`TerminalApprover.__call__` → 抛错（本线无交互）。
4. Gateway 与计费：不做。在 `provider.py` 顶部注释与第 9 节记债：产品化时实现 `GatewayProvider(LLMProvider)`，走 `apps/v2/agent/gateway_clients.GatewayInvocationHttpClient`，接缝不变。

验证：`test_provider.py`（`httpx.MockTransport`）：请求体含 `thinking.disabled`；`reasoning_tokens=3` → 抛；429 两次后成功且延迟序列正确；`length` 原样透传；工具调用解析。`test_runner.py`：用假 provider 编排"两轮工具 + 注入一次 + 停止回调"，断言 hook 调用顺序与 `stop_reason`；代码记忆钩子替换 `messages[:]` 后下一轮请求用的是新列表。

### C5 · 执行端口与远程算力

做什么：

1. `execution/remote_relay/`：从 `search/DeepEvol-reproduction-c/DeepEvol1.0/Agent/remote_relay/` 整目录 vendor（内部是相对导入，不用改），保留 `DEEPEVOL_VENDOR.md`，追加一行"2026-09 vendored again under apps/v2/agent/paper2code/execution"。
2. `execution/port.py`：

   ```python
   class ExecutionPort(Protocol):
       async def run(self, job: Job) -> JobResult: ...   # job: 工作区路径、命令、超时、镜像
       async def close(self) -> None: ...
   ```

   `Job` 含 `workspace`（本地 `generate_code/<项目>`）、`argv`、`timeout_s`、`cwd_in_workspace`；`JobResult` 含退出码、stdout/stderr 尾 64 KiB、耗时、`machine`、作业目录路径。每次 `run` 写 `jobs/<seq>/`。
3. `execution/job_executor.py::RemoteDockerExecutor(port)`：
   - `runtime: remote_relay.Runtime`（远端）。
   - 同步：`rsync`（或 relay 的 `fs.transfer`）本地工作区 → 远端 `jobs/<seq>/workspace/`，排除 `.git`、`__pycache__`、`.venv`；跑完把 `workspace/` 中新增或修改的文件同步回本地（按 mtime + sha 清单），不同步删除。
   - 执行：`docker run --rm --network none --cpus <n> --memory <m> -v <远端作业目录>/workspace:/workspace -w /workspace <image> bash -lc <cmd>`，`timeout` 由 `Job.timeout_s`，超时 `docker kill`。
   - 镜像：`python:3.11-slim` 为底，run 首次执行时若工作区有 `requirements.txt` 则 `pip install -r` 后 `docker commit` 成 `paper2code-run-<id>`，之后复用；装不上只记警告不阻断（本分支不做环境整备）。`pytest` 总是预装。
4. `execution/lease.py`、`release_policy.py`：从 `Agent/DeepEvol/experiment/` 搬（零依赖）。`experiment/lease_api.py` 不搬（它调的是 Product 内部 API）。
5. `execution/aliyun_lease.py`：从 `canary/aliyun_lease.py` 搬，两处改：`ManifestViolation` → 本地 `LeaseError`；ECS 客户端从 `apps.api...aliyun_ecs.AliyunECSClient` 改为主干 `apps/v2/remote_compute/providers.py::AliyunControlClient`。**先核对主干客户端是否覆盖 `RunInstances` / `DescribeInstances` / `StopInstance(StopCharging)` / `StartInstance` / `DeleteInstance` 与镜像选择（`ALIYUN_IMAGE_ID`）；缺的从分支 `aliyun_ecs.py` 摘对应方法（不整搬 1492 行）。** 凭据经 `FileRemoteComputeSecretStore` 或 `ALIYUN_ACCESS_KEY_ID` / `_SECRET` 环境变量（`--env-file`）。机器规格：本分支只用 CPU 档，`--compute-tier enough|comfortable` 映射到两个实例类型常量；镜像用 reproduction-c `deploy/experiment-images/bootstrap.sh` 烤过的 Docker-ready 镜像 id（`ALIYUN_IMAGE_ID`），没有就在 `lease.json` 记 `bootstrap_required` 并在首次连上后跑一次精简版 bootstrap（只装 Docker）。
6. `execution/remote_daemon.py`：从 `canary/remote_daemon.py` 搬（只改异常类）。它把远端 `docker.sock` 隧道到本地 Unix socket 并设 `DOCKER_HOST`，`job_executor` 的 `docker` 命令因此在本地执行、作用于远端守护。
7. `execution/leased_runtime.py::LeasedExecutionPort`：包 `RemoteDockerExecutor`，首次 `run` 时 `aliyun_lease.acquire` → `remote_daemon.start`，`close()` 时 `release`（`DeleteInstance`），硬顶 `--run-hours` 到点强制释放并让当前作业失败。驱动器在 `finally` 里必调 `close()`，另加 `release` 子命令兜底。
8. `verification_hook.py`：引擎 `code_implementation_workflow._verify_generated_code` 调的是 `support.verification.run_verification`（本地 subprocess）。在引擎加一个可注入点：`CodeImplementationWorkflow.__init__` 增加 `verification_runner: Callable | None`，默认仍是 `run_verification`；本线传入"走执行端口"的实现（命令、超时、64 KiB 尾一致）。这是对引擎的一处小改，记进 `VENDOR.md`。
9. `LocalRuntime` 只在测试里用（`RemoteDockerExecutor` 接 `LocalRuntime` + 本机 Docker 即可离线跑通执行路径）。

验证：`test_job_executor.py`：`LocalRuntime` + 本机 Docker，工作区里一个 `test_x.py`，`pytest -q` 退出码与尾输出正确，产出文件同步回本地；`test_leased_port.py`：假 ECS（照 reproduction-c `tests/v2_reproduction/fake_aliyun.py` 写一个 60 行的替身）验证"首次 run 才租、close 必释放、超时强制释放"；`test_verification_hook.py`：引擎验证走端口。

### C6 · 输入与闸门

做什么：

1. `intake.py`：
   - 接受 PaperBench 论文目录（`paper.pdf` `paper.md` `addendum.md` `blacklist.txt` `assets/`）。`paper.md` 必须存在（不做 PDF 转换；这条线以 PaperBench 口径为准）。
   - 生成 `input/paper.md` = `paper.md` + `\n\n# Addendum\n\nClarifications provided with the paper by the benchmark authors (in scope; follow them):\n\n` + `addendum.md`（无 addendum 则不加），与 `run_trial.sh` 逐字一致；sha256 记入 `run.json`。
   - `blacklist.txt` → `run.json.denylist`（逐行去注释去空行），注入 `tools/fetch.py` 与 clone 包装。
   - **`rubric.json` / `config.yaml` 不拷贝、不读取**；`run.json` 只记 `paper_dir` 路径。`criteria` phase 只写一行 "rubric passthrough: <路径存在与否>"。
   - 可选借用 `apps/common/reproduction_paper_bundle.py` 读 zip 形态的论文包；本分支先只支持目录。
2. `gates.py`（照 `run_trial.sh`）：
   - 起跑自检 `preflight`：`planning` / `implementation` 无模型覆盖；`max_tokens >= 32768`；论文包有 `blacklist.txt` 时 `denylist` 非空；`git config --global --get-regexp insteadof` 命中黑名单仓库否则 **警告**（进程内工具已强制黑名单，git 封锁是第二道）；provider 探活一次（`reasoning_tokens == 0`）。
   - 假计划闸 `plan_source`：`planning_result_meta.json.source == "generated"`（`strict_outcomes=True` 已让规划失败直接抛，此闸兜底）。
   - 状态闸 `implementation_status`：`inner_status == "completed"`。
   - 归属闸 `ownership`：任务目录 `paper.md` 的 sha256 == `input/paper.md` 的 sha256；`generate_code/` 下文件数 ≥ 5。
   任一闸失败 → phase 记 `failed`，`status.json` 写原因，不进入下一 phase。

验证：`test_intake.py`（拼接结果与 `run_trial.sh` 的 shell 拼接逐字节比对；黑名单解析）；`test_gates.py`（四闸各一个正例一个反例）。

### C7 · 驱动器与 CLI

做什么：

1. `phases.py`：phase → 引擎函数：

   | phase | 调什么 | 产物 |
   | --- | --- | --- |
   | `intake` | `intake.prepare(run)` → `prepare_workflow_environment(raw_input=input/paper.md, task_kind="paper2code", task_id=<run-id>, workspace_root=workspace)` → `acquire_input_artifact(ctx)` → `synthesize_workspace_infrastructure_agent(ctx)` | 任务目录、`dir_info`（存 `phases/01_intake.json`） |
   | `criteria` | 记录 rubric 路径存在与否 | 一行 |
   | `plan` | `orchestrate_document_preprocessing_agent` → `orchestrate_code_planning_agent(strict_plan_validation=True)` → 假计划闸 | `initial_plan.txt` 等 |
   | `plan_review` | `run_plan_review_gate(callback=auto_approve)`；`--ask` 时改为从 `phases/04_plan_review.decision.json` 读决定 | `plan_versions/`、历史 |
   | `references` | `orchestrate_reference_intelligence_agent` | `reference.txt` |
   | `acquire` | `automate_repository_acquisition_agent` | `code_base/` |
   | `index` | `orchestrate_codebase_intelligence_agent`；`--skip index` 时写 skipped 报告 | `indexes/` |
   | `implement` | `synthesize_code_implementation_agent(enable_indexing=True, require_verification=True)` + 状态闸 + 归属闸 | `generate_code/`、`verification` |
   | `compute` | 桩：把 `lease.json` 摘要（实例、规格、租用时长）写进 phase 记录 | — |
   | `environment_run` | 本分支：把 `implement` 的 `verification` 列表与 `jobs/` 索引汇总为 `phases/10_environment_run.json` | — |
   | `optimize` | 桩 | — |

   `dir_info` 在 phase 之间通过 `phases/01_intake.json` 持久化，重跑任一 phase 时从文件重建，不依赖内存。
2. `driver.py`：`init`（写 `run.json`、`input/`、`status.json`）、`step --phase <name>`（单跑，前置 phase 必须 `completed`）、`run --until <phase>`（顺序跑，遇 `failed` 停）、`status`、`release`（强制释放机器）、`rerun --phase`（把该 phase 及之后标 superseded 后重跑）。每个 phase 尝试写一个 `phases/<nn>_<name>.json`，含开始/结束时间、输入 sha、引擎返回值、闸门结果、错误。`events.jsonl` 记 `phase.started/finished/failed`、`llm.call`、`job.run`、`lease.*`。`finally` 里 `port.close()`。
3. `scripts/paper2code_canary.py`：参数照 reproduction-c 的 canary：`--run-dir`、`--paper-dir`、`--model`（默认 `DeepSeek-V4-Flash`）、`--provider-base-url`（默认 `https://llmapi.paratera.com/v1`）、`--provider-key-env`（默认 `PARATERA_API_KEY`）、`--provider-stream`、`--compute aliyun|local`（`local` 仅测试）、`--compute-tier`、`--run-hours`、`--env-file`（可重复）、`--skip index`、`--ask`。日志到 `canary.log`。
4. `footprint.yaml` + `tests/v2_paper2code/test_footprint.py`：从 reproduction-c 抄，扫描词改 `paper2code`，登记 `pyproject.toml`（依赖）、`CONTEXT.md`（词条）、`scripts/paper2code_canary.py`。

验证：`test_driver_offline.py`：假 provider（脚本化回复）+ `LocalRuntime` + EMA-Detect 合成论文，`run --until implement` 全绿，四闸全过，`phases/` 十一个文件齐；`rerun --phase plan` 后 `plan_versions` 出现新版本；`test_footprint.py` 绿。

### C8 · 冒烟：EMA-Detect，真模型，真机器

做什么：

```bash
cd search/DeepEvol-Paper_repro_0916/DeepEvol1.0
PY=<主干 venv 的 python>
$PY scripts/paper2code_canary.py init --run-dir runs/ema --paper-dir <EMA-Detect 目录，含 paper.md> \
  --model DeepSeek-V4-Flash --provider-base-url https://llmapi.paratera.com/v1 --provider-key-env PARATERA_API_KEY \
  --compute aliyun --compute-tier enough --run-hours 2 \
  --env-file ~/Documents/env/paratera.env --env-file ~/Documents/env/aliyun.env
nohup $PY scripts/paper2code_canary.py run --run-dir runs/ema --until environment_run \
  --env-file ~/Documents/env/paratera.env --env-file ~/Documents/env/aliyun.env > runs/ema/canary.log 2>&1 &
```

EMA-Detect 目录要补一个空 `blacklist.txt` 和 `addendum.md`（可无）。

通过标准：`status.json` 到 `environment_run` 全 `completed`；`llm/` 每条 `reasoning_tokens == 0`；`jobs/` 至少一条 `pytest` 作业且 `machine` 是 Aliyun 实例；`lease.json` 有 `released_at`；Aliyun 控制台无残留实例。

修的问题回填到对应提交，不单独立提交。

### C9 · 验收：sapg

同 C8，`--paper-dir <paperbench>/sapg`（55k 字符，会走分段模式），`--run-hours 6`。通过标准同上，外加：`initial_plan.txt` `source == generated`；`code_base/` 至少一个仓库、`indexes/` 至少一个索引；`generate_code/` 文件数 ≥ 5。

跑完写 `apps/v2/agent/paper2code/HANDOFF.md`：run 目录位置、耗时、token、遇到的坑、下一步。

### C10 · 收尾

- `VENDOR.md` 补齐 C2、C5 的引擎改动清单。
- `README.md`（本线）：一页说明 + 命令。
- 全部离线测试：`python -m pytest tests/v2_paper2code -q`。
- 主干既有测试跑一遍 `tests/v2_agent tests/data_contracts`，确认没碰坏（已知红的四个套件按 reproduction-c 的说明排除）。

## 4. 对引擎的改动清单（写进 VENDOR.md）

1. 导入路径重写（C1）。
2. 验证仓库补丁"带走"子集（C2 表）。
3. `seams/compat.py::apply_tool_filter` 按消毒后前缀匹配（C2）。
4. `CodeImplementationWorkflow.__init__` 增加 `verification_runner` 注入点（C5）。

其余文件与 `search/paper2code-kernel@c821130` 逐字一致。

## 5. 接缝实现对照

| 接缝 | 实现 |
| --- | --- |
| `KernelRuntime.provider_for` | `config.py::PaperKernelRuntime` |
| `LLMProvider.chat_with_retry` / `get_default_model` | `provider.py::ParateraProvider` |
| `AgentRunner.run` | `runner.py::PaperAgentRunner` |
| `Agent.__aenter__` / `__aexit__` / `attach_llm` | `agent.py::PaperAgent` |
| `AugmentedLLM.generate` | `agent.py::PaperAugmentedLLM` |
| `build_permission_engine` | 返回 `FullAutoEngine` |
| `TerminalApprover.__call__` | 抛 `RuntimeError`（本线无交互） |

## 6. 环境变量与默认值

| 变量 | 默认 | 来源 |
| --- | --- | --- |
| `DEEPCODE_LLM_RETRY_MODE` | `persistent` | run_trial.sh |
| `DEEPCODE_CHAT_RETRY_DELAYS` | `10,30,60,180,300` | run_trial.sh |
| `DEEPCODE_PERSISTENT_MAX_DELAY` | `900` | run_trial.sh |
| `DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT` | `30` | run_trial.sh |
| `DEEPCODE_OPENAI_REQUEST_TIMEOUT_S` | `600` | run_trial.sh |
| `DEEPCODE_PREFILTER_MAX_TOKENS` | `32000` | run_trial.sh |
| `DEEPCODE_ANALYSIS_MAX_TOKENS` | `16000` | run_trial.sh |
| `DEEPCODE_RELATIONSHIP_MAX_TOKENS` | `16000` | run_trial.sh |
| `DEEPCODE_CODE_ANALYZER_TIMEOUT_S` | `600` | run_trial.sh（上游已有旋钮） |
| `DEEPCODE_REFERENCE_MAX_TOKENS` | `32768` | run_trial.sh |
| `DEEPCODE_DOWNLOAD_MAX_TOKENS` | `16384` | run_trial.sh |
| `DEEPCODE_REFERENCE_MAX_ITERATIONS` | `40` | 本线新增，不写死 80；引擎自身缺省 8，两次真机 8 都不够出报告（§10 第 11 条） |
| `DEEPCODE_DOWNLOAD_MAX_ITERATIONS` | `12` | 本线新增；引擎自身缺省 8（§10 第 11 条） |
| `DEEPCODE_STALL_THRESHOLD` | `7200` | run_trial.sh |
| `DEEPCODE_MAX_WALL_SECONDS` | `21600` | run_trial.sh |
| `DEEPCODE_URL_DENYLIST` | 由 `run.json.denylist` 生成 | run_trial.sh |
| `PARATERA_API_KEY` | 必需 | `--env-file` |
| `ALIYUN_ACCESS_KEY_ID` / `ALIYUN_ACCESS_KEY_SECRET` / `ALIYUN_REGION_ID` / `ALIYUN_IMAGE_ID` | 必需（`--compute aliyun`） | `--env-file` |

思考关闭不是环境变量，是 provider 常量；`DEEPCODE_THINKING` 不实现。

## 7. 已知风险

1. **主干 `AliyunControlClient` 覆盖面**：若缺 `RunInstances` / 镜像选择，要从分支 `aliyun_ecs.py` 摘方法，C5 第 5 项已写核对动作。
2. **远端镜像**：`ALIYUN_IMAGE_ID` 必须是 Docker-ready 镜像，否则首次连上要装 Docker（境内机器走 Aliyun 源，`bootstrap.sh` 有现成命令）。
3. **`asyncssh` 与 `cryptography`**：主干钉 `cryptography>=46,<47`，`asyncssh<2.22` 应兼容；`uv lock` 时若冲突，改用 `apps/v2/reliability/ssh_channel.py`（paramiko）重写 relay 的 transport，工作量约一天。
4. **kernel 在 3.11 下**：无已知 3.12 专属语法，C1 验证覆盖。
5. **`mcp` 库导入**：`STRICT_V2_OMITTED_LEGACY_FEATURES` 的测试检查的是进程与适配器，不是 import；若 `test_strict_v2_sqlite_boundary` 报警，把工具模块的 FastMCP 装饰器换成本地空装饰器（引擎改动第 5 条）。
6. **执行前同步的粒度**：kernel 循环里 `execute_*` 调用很少（fre / rice 实测 0 次），验证一次；同步成本可接受。若某篇论文模型频繁 `execute_python`，每次同步几十 MB 会慢，届时改成远端增量。
7. **口径变化**：所有测试改 V4-Flash 关思考后，验证仓库的 0.8367 不可比；比分前原装 DeepCode 要用同口径重跑一次。

## 8. 推迟项（债）

| 项 | 挂点 |
| --- | --- |
| Gateway 与计费准入 | `provider.py` 增加 `GatewayProvider`，接缝不变 |
| Product 表、迁移、`/api/v1/paper2code/*` | 参考 reproduction-c 的 `reproduction.*` 骨架，角色词汇四处闭合 |
| 判分与摆卷 | `run_trial.sh` 的摆卷段 + 官方 `run_grade.sh` |
| 环境整备（第 10 步搭环境） | `job_executor.py` 的镜像构建从"best-effort pip"升级为 recipe |
| 闭环修复（第 10、11 步） | `_run_kernel_implementation` 的 `should_stop` / `inject_followups` |
| A3 自建循环 | 替换 `implement` phase 的引擎调用，phase 边界不变 |
| 双 agent 提取步骤 | `plan` 前加 `extract` phase，产物 `algorithm_extraction.md` |

## 9. 已确认（2026-09-16）

- 执行镜像：`python:3.11-slim` 预装 `pytest`，run 首次执行时 best-effort `pip install -r requirements.txt` 后 `docker commit` 复用。装不上记警告不阻断。
- 计划审阅：默认自动批准；`--ask` 时驱动器在 `plan_review` 停下，把决定写到 `phases/04_plan_review.decision.json`（`action: approve|modify|replace|cancel`，`feedback` / `plan` 随附）后 `step --phase plan_review` 继续。不做交互式输入。
- 依赖：`loguru` `mcp` `aiohttp` `aiofiles` `asyncssh` 加在 `pyproject.toml` 主依赖。


## 10. 实施偏差（2026-09-17 review 通过）

计划正文保持 C0 时的原样；实际做法与计划不同之处列在这里，细节和原因在 `HANDOFF.md`。

| # | 计划条目 | 实际做法 | 原因 |
| --- | --- | --- | --- |
| 1 | C5-6 docker.sock 隧道到本地、docker 命令本地执行 | docker 命令经 SSH 在机器上执行（`SshDockerHost` / `SshOnlyDaemon`），隧道 vendor 但不启用 | 隧道下 `docker pull` 挂死，直连 7 秒 |
| 2 | C5-3 镜像策略 | 基础镜像先 `docker pull`，失败换 docker.1ms.run / daocloud / dockerproxy；容器一律 `docker run -d` + 短轮询取日志 | 长命令随 SSH 会话一起死；HK 机器拉 Docker Hub 超时 |
| 3 | C5-7 `release` 子命令兜底 | 增加 `release_failed` 状态、ECS API 传输错误重试 4 次、`release` 先查 DescribeInstances 再删 | DeleteInstance 遇 SSL EOF 曾留下计费实例 |
| 4 | C6 状态闸 `inner_status == completed` | 也接受 `unverified` / `no_tests_discovered` 且全部文件写完；`test_failed` 仍失败 | 无测试的仓库永远拿不到 `completed` |
| 5 | C6 预检"有 blacklist.txt 则 denylist 非空" | "blacklist.txt 的每一条都在 denylist 里"（空文件合法） | EMA-Detect 的黑名单为空 |
| 6 | C7 `environment_run` 只汇总验证与 jobs | 固定加一条 `python -m compileall` 远端作业，字节码写到 `PYTHONPYCACHEPREFIX` | 无测试的仓库也要走一次租机、镜像、同步 |
| 7 | C7 `references` / `acquire` 直接透传引擎 | 报告是迭代耗尽套话或为空时 `references` 判失败；报告不含仓库时 `acquire` 记 skipped；识别 `Repository: owner/repo` 简写 | 退化结果曾静默通过 |
| 8 | C3 github-downloader 直接包装 | `git_clone` 目标强制落在任务 `code_base/`；`GIT_TERMINAL_PROMPT=0` | 模型传空路径把仓库克隆进了仓库根目录 |
| 9 | C3 fetch 复用引擎解析器 | DNS 落在 198.18/15 时回退系统解析 | 开发机代理的 fake-IP DNS |
| 10 | C4 `reasoning_tokens != 0` 即作废 | 只看 token；`reasoning_content` 记长度，`PAPER2CODE_STRICT_REASONING_CONTENT=1` 恢复严格；`llm/<seq>` 跨进程接续编号 | Paratera 的 V4-Flash 会给零 token 的 reasoning_content |
| 11 | §6 `DEEPCODE_REFERENCE_MAX_ITERATIONS=8`、`DOWNLOAD=8` | 40 / 12 | GLM 与 V4-Flash 两次真机 8 都不够出报告；D0 已回写 §6 与 `config.py` |
| 12 | （未预见）V2 出网守卫 | 直连 Paratera 的两个符号钉死豁免在 `scripts/data_contracts/check_provider_egress.py`，Gateway 落地时删除 | 仓库规则：V2 代码不得直连 provider |
| 13 | §0 验收在 V4-Flash 上 | EMA 冒烟与 sapg 预演在 GLM-4.5-Flash 上，C9 本身在 V4-Flash（`paratera_backup.env`） | `paratera.env` 的 key 对 DeepSeek 403 |
| 14 | §2 布局 | 多出 `plan_review.py`；`execution/port.py` 在 C3 建 | `--ask` 循环独立成文件；工具层先于执行器需要类型 |
| 15 | 计划表第 7 阶段建议索引上限 | 不加上限 | 2026-09-17 决定 |
