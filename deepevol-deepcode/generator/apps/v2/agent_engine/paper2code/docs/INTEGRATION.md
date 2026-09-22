# 接入文档

本目录不能直接运行。它是从 DeepCode 切下来的论文复现业务层，模型循环、MCP 连接、权限策略三样东西被留成了接缝，由接入方提供。本文说明接缝在哪、每个接缝要做什么、循环必须遵守的行为约定，以及怎么把管线跑起来。

词汇按 [CONTEXT.md](../CONTEXT.md)。

## 1. 目录与导入

```
paper2code-kernel/
├── workflows/   复现管线本体（步骤编排、规划、实现、审阅）
├── tools/       七个 MCP 服务器中的五个，以及文档转换、下载、索引器
├── prompts/     全部系统提示词
├── utils/       文件处理、配置读取、循环检测
├── seams/       接缝：接入方要实现的十个方法，以及它们的数据类型
├── support/     从 DeepCode 拷来的无依赖真代码，不是接缝
└── docs/
```

把本目录加进 `sys.path`，导入路径就是 `from workflows.agent_orchestration_engine import ...`。不是 pip 包，没有 `setup.py`。

第三方依赖（业务层直接 import 的）：`loguru`、`aiohttp`、`aiofiles`、`pypdf`、`PyYAML`、`mcp`（五个服务器文件用 `FastMCP` 和 `Server`）、`tiktoken`（可选，缺失时退化）、`docling`（可选，缺失时用内置转换器）。Python 3.12。

## 2. 接缝清单

十个方法，全部在 `seams/` 下，未实现时抛 `NotImplementedError`。

| 接缝 | 文件 | 要做什么 |
| --- | --- | --- |
| `KernelRuntime.provider_for` | `seams/config.py` | 按阶段和模型名返回一个 `LLMProvider` |
| `LLMProvider.chat_with_retry` | `seams/llm_runtime.py` | 一次带重试的对话补全 |
| `LLMProvider.get_default_model` | `seams/llm_runtime.py` | 这个 provider 实例的模型 id |
| `AgentRunner.run` | `seams/agent_runtime.py` | 工具调用循环，见第 4 节 |
| `Agent.__aenter__` | `seams/compat.py` | 连接 `server_names` 里的 MCP 服务器，注册工具 |
| `Agent.__aexit__` | `seams/compat.py` | 断开 |
| `Agent.attach_llm` | `seams/compat.py` | 解析 provider，返回 `AugmentedLLM` |
| `AugmentedLLM.generate` | `seams/compat.py` | 用 `build_run_spec` 造规格，交给 `AgentRunner` |
| `build_permission_engine` | `seams/harness.py` | 返回有 `.mode` 和 `.evaluate()` 的对象 |
| `TerminalApprover.__call__` | `seams/harness.py` | 询问人类是否放行一个工具调用 |

其余在 `seams/` 里的东西都是真代码：`KernelConfig` 及其嵌套数据类、`RequestParams`、`AgentRunSpec` / `AgentRunResult` / `AgentHook` / `AgentHookContext`、`Tool` / `ToolRegistry` / `AliasedTool`、`LLMResponse` / `ToolCallRequest`、`PermissionMode` 枚举、`FullAutoEngine`、`observability` 和 `sessions` 两个空操作模块。接入方不需要动它们。

每个接缝文件的模块文档字符串都标了"谁在用它"。

## 3. 配置

业务层通过 `get_runtime().config` 读六组配置，全部在 `seams.config.KernelConfig` 里：

| 字段 | 谁读 | 作用 |
| --- | --- | --- |
| `workspace.root` | `workflows/environment.py` | `workflow_root` 参数缺省时的工作区根，默认 `./deepcode_lab` |
| `workspace.max_input_mb` | 同上 | 本地输入文件大小上限，默认 100 |
| `tools.default_search_server` | `agent_orchestration_engine.py` | 参考挖掘用的搜索服务器名，默认 `filesystem` |
| `tools.mcp_servers` | `Agent.__aenter__`（接入方）、`environment.py` | 服务器名到启动方式的表；`environment.py` 会往 `filesystem` 的 `args` 里追加工作区路径 |
| `security` | `code_implementation_workflow.py` | 原样传给 `build_permission_engine` |
| `agents.defaults.{max_tokens, base_max_tokens, retry_max_tokens}` | `utils/llm_utils.get_token_limits` | 规划器输出上限和重试时的上限 |
| `agents.{defaults,planning,implementation}.model` | `utils/llm_utils.get_default_models` | 各阶段模型名，通过 `resolve_phase` 合并 |
| `document_segmentation.{enabled, size_threshold_chars}` | `utils/llm_utils.should_use_document_segmentation` | 是否分段、超过多少字符才分段，默认 50000 |

安装方式：

```python
from seams.config import KernelConfig, KernelRuntime, MCPServerConfig, set_runtime

cfg = KernelConfig()
cfg.agents.defaults.model = "your-model"
cfg.agents.defaults.max_tokens = 8192
cfg.tools.mcp_servers["code-implementation"] = MCPServerConfig(
    name="code-implementation",
    type="stdio",
    command="/path/to/python",
    args=["/path/to/paper2code-kernel/tools/code_implementation_server.py"],
)
# ... 其余六个服务器见 seams/mcp_servers.json

class MyRuntime(KernelRuntime):
    def provider_for(self, *, provider_name=None, connection_id=None,
                     phase="default", model=None, execution_profile=None):
        settings = self.config.resolve_phase(phase)
        return MyProvider(model=model or settings.model, settings=settings)

set_runtime(MyRuntime(cfg))
```

`get_runtime()` 在 `set_runtime` 之前调用会抛 `RuntimeNotConfigured`，不会静默用默认值。

`phase` 取值只有 `"default"`、`"planning"`、`"implementation"` 三个。规划器、参考挖掘、仓库下载、计划修订用 `planning`；实现循环、代码记忆摘要用 `implementation`；索引器用 `default`。

## 4. 循环的行为约定

`AgentRunner.run(spec)` 是最重要的接缝。业务层不只是调用它，还通过 `AgentHook`、`should_stop_callback`、`injection_callback` 三个口子往循环里插逻辑，代码记忆的清空策略就挂在 `after_iteration` 上。循环若不按下面的约定走，实现步骤会静默出错。

以下约定逐条从 DeepCode 的 `core/agent_runtime/runner.py` 提取。

### 4.1 输入与输出

- 从 `spec.initial_messages` 开始，这是一个可变的 `list[dict]`，`role` 为 `system` / `user` / `assistant` / `tool`。循环在原地追加。
- 返回 `AgentRunResult`。`final_content` 是模型最后一段文本；`stop_reason` 取 `completed`、`max_iterations`、`callback_stop`、`tool_error`、`error`、`empty_final_response` 之一；`error` 在出错时填错误文本；`tools_used` 是调过的工具名列表；`usage` 是累计 token。

### 4.2 每轮的顺序

一轮 = 一次模型请求。每轮按这个顺序：

1. 若 `max_iterations` 已耗尽：先试 `injection_callback`，有内容就重置计数继续；没有就以 `stop_reason="max_iterations"` 结束，`final_content` 用 `spec.max_iterations_message` 或 `DEFAULT_MAX_ITERATIONS_MESSAGE`。
2. 若 `spec.should_stop_callback` 存在，`await` 它。返回非空字符串就以 `stop_reason="callback_stop"` 结束，那个字符串作为 `final_content`。返回 `None` 继续。回调抛异常按继续处理，记日志。
3. 构造 `AgentHookContext(iteration=n, messages=messages)`，依次 `await hook.before_iteration(ctx)`、`await hook.before_model_request(ctx)`。
4. 请求模型：`provider.chat_with_retry(messages, tools=spec.tool_definitions(), model=spec.model, max_tokens=spec.max_tokens, temperature=spec.temperature, reasoning_effort=spec.reasoning_effort, retry_mode=spec.provider_retry_mode, on_retry_wait=spec.retry_wait_callback)`。把响应放进 `ctx.response`，`await hook.on_model_response(ctx)`。
5. 若响应 `finish_reason == "error"`：以 `stop_reason="error"` 结束，`error` 填 `response.content`，`final_content` 用 `spec.error_message`。
6. 若响应带工具调用（`response.should_execute_tools`）：
   - 把 assistant 消息（含 `tool_calls`）追加到 `messages`。
   - `ctx.tool_calls = response.tool_calls`，`await hook.before_execute_tools(ctx)`。
   - 逐个执行工具（见 4.3），每个结果作为 `{"role": "tool", "tool_call_id": id, "content": text}` 追加到 `messages`，同时收进 `ctx.tool_results`。
   - `await hook.after_iteration(ctx)`。**这一步之后 `messages` 可能已被钩子原地替换**，下一轮必须用替换后的列表。
   - 回到第 1 步。
7. 若响应没有工具调用：
   - `clean = hook.finalize_content(ctx, response.content)`。
   - 若 `clean` 为空且 `finish_reason != "error"`：最多重试两次（追加一条要求给出最终回复的 user 消息再请求）；仍为空就以 `stop_reason="empty_final_response"` 结束。
   - 把 assistant 消息追加到 `messages`。
   - `await hook.after_iteration(ctx)`。
   - `await spec.injection_callback()`。返回的 `{"role","content"}` 列表非空：逐条追加到 `messages`，重置迭代计数，回到第 1 步。返回空：以 `stop_reason="completed"` 结束，`final_content = clean`。

`finish_reason == "length"` 时 DeepCode 会追加一条续写提示再请求，最多三次。这是可选优化，不做也不影响业务层。

### 4.3 单个工具调用

对每个 `ToolCallRequest`：

1. 权限：若 `spec.permission_checker` 存在，调 `checker(name, arguments)` 得 `(decision, reason)`。`decision` 是 `"allow"` / `"ask"` / `"deny"` 字符串或带 `.value` 的枚举。`allow` 放行；`deny` 不执行，结果为 `"Error: permission denied: <reason>"`；`ask` 转给 `spec.approval_callback(name, arguments, reason)`，`await` 后为真放行，为假或没有回调则拒绝。检查器或审批回调抛异常一律按拒绝处理。
2. 执行：`await spec.tools.execute(name, arguments)`。`ToolRegistry.execute` 自己处理未知工具、参数校验失败、工具抛异常，都返回以 `"Error"` 开头的字符串加一句提示，不会向上抛。
3. 截断：结果文本超过 `spec.max_tool_result_chars` 就截断，DeepCode 保留头尾、中间换成省略标记。
4. 若 `spec.fail_on_tool_error` 为真且结果以 `"Error"` 开头：以 `stop_reason="tool_error"` 结束。业务层没有开这个开关。

`spec.concurrent_tools` 为真时可并行执行只读工具，业务层传的是假，顺序执行即可。

### 4.4 一个够用的参考循环

```python
async def run(self, spec):
    messages = spec.initial_messages
    hook = spec.hook or AgentHook()
    remaining = spec.max_iterations
    tools_used, usage, iteration = [], {}, 0

    async def drain():
        if spec.injection_callback is None:
            return False
        items = await spec.injection_callback() or []
        for m in items:
            messages.append({"role": "user", "content": m["content"]})
        return bool(items)

    while True:
        if remaining is not None and remaining <= 0:
            if await drain():
                remaining = spec.max_iterations
            else:
                msg = (spec.max_iterations_message or DEFAULT_MAX_ITERATIONS_MESSAGE)
                return AgentRunResult(final_content=msg.format(max_iterations=spec.max_iterations),
                                      messages=messages, tools_used=tools_used, usage=usage,
                                      stop_reason="max_iterations")
        if spec.should_stop_callback is not None:
            reason = await spec.should_stop_callback()
            if reason:
                return AgentRunResult(final_content=reason, messages=messages, tools_used=tools_used,
                                      usage=usage, stop_reason="callback_stop")
        if remaining is not None:
            remaining -= 1
        iteration += 1

        ctx = AgentHookContext(iteration=iteration, messages=messages)
        await hook.before_iteration(ctx)
        await hook.before_model_request(ctx)
        response = await self.provider.chat_with_retry(
            messages, tools=spec.tool_definitions(), model=spec.model,
            max_tokens=spec.max_tokens, temperature=spec.temperature,
            reasoning_effort=spec.reasoning_effort, retry_mode=spec.provider_retry_mode,
            on_retry_wait=spec.retry_wait_callback)
        ctx.response = response
        accumulate(usage, response.usage)
        await hook.on_model_response(ctx)

        if response.finish_reason == "error":
            return AgentRunResult(final_content=spec.error_message, messages=messages,
                                  tools_used=tools_used, usage=usage,
                                  stop_reason="error", error=response.content)

        if response.should_execute_tools:
            messages.append(assistant_message_with_tool_calls(response))
            ctx.tool_calls = list(response.tool_calls)
            await hook.before_execute_tools(ctx)
            for call in response.tool_calls:
                denial = await check_permission(spec, call)          # 4.3 第 1 条
                text = denial or await spec.tools.execute(call.name, call.arguments)
                text = truncate(str(text), spec.max_tool_result_chars)
                tools_used.append(call.name)
                ctx.tool_results.append(text)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
            await hook.after_iteration(ctx)                          # 钩子可能替换 messages[:]
            continue

        clean = hook.finalize_content(ctx, response.content)
        if not (clean or "").strip():
            return AgentRunResult(final_content="", messages=messages, tools_used=tools_used,
                                  usage=usage, stop_reason="empty_final_response")
        messages.append({"role": "assistant", "content": clean})
        ctx.final_content = clean
        await hook.after_iteration(ctx)
        if await drain():
            remaining = spec.max_iterations
            continue
        return AgentRunResult(final_content=clean, messages=messages, tools_used=tools_used,
                              usage=usage, stop_reason="completed")
```

四十行左右，省略了空回复重试和续写。`check_permission` 按 4.3 第 1 条写。

### 4.5 实现步骤怎么用这个循环

`workflows/code_implementation_workflow.py` 里 `_run_kernel_implementation` 是唯一直接构造 `AgentRunSpec` 的地方，它插进去的逻辑：

- `hook = _ImplementationHook(state)`：`after_iteration` 里往 `messages` 追加一条引导 user 消息（有工具调用时）、把写完的文件登记进代码记忆、判断是否触发清空并原地替换 `messages[:]`。
- `should_stop_callback`：中止原因已设、超时（默认 7200 秒）、循环检测器报警、所有计划文件已写完，四种情况返回字符串。
- `injection_callback`：模型不调工具就停了但还有文件没写，注入一条"继续实现"的 user 消息。
- `permission_checker=engine.evaluate`，`approval_callback` 仅在非 `FULL_AUTO` 模式下有。
- `max_iterations=800`，`max_tool_result_chars=60000`。

`AugmentedLLM.generate` 的参考实现就是 `spec = self.build_run_spec(message, params)` 然后 `await YourRunner(self.provider).run(spec)`。`build_run_spec` 是真代码，负责把 `RequestParams` 映射到规格，规则和 DeepCode 一致。

## 5. Agent 与 MCP

`Agent(name, instruction, server_names)` 的容器部分是真代码。接入方实现三个方法：

**`__aenter__`**：对 `server_names` 里每个名字，查 `get_runtime().config.mcp_servers[name]` 拿到启动方式，连上，把它的每个工具包装成 `seams.agent_runtime.Tool` 的实例（`name` 为 `mcp_<server>_<tool>`，`parameters` 为该工具的 JSON Schema，`execute` 转发调用），`self.register_tool(...)` 注册。

DeepCode 在服务器缺失或启动失败时只打 warning 然后继续，管线会在没有工具的情况下跑完并产出垃圾。建议接入方抛异常。

**`__aexit__`**：断开。业务层在 `finally` 里调它，不期望异常。

**`attach_llm`**：`provider = get_runtime().provider_for(provider_name=..., phase=phase, model=model)`，返回 `YourAugmentedLLM(agent=self, provider=provider, provider_name=名字, phase=phase)`。

七个服务器和它们的启动方式在 [seams/mcp_servers.json](../seams/mcp_servers.json)，占位符换成本地路径。哪个步骤需要哪个：

| 服务器 | 来源 | 谁用 |
| --- | --- | --- |
| `filesystem` | npm `@modelcontextprotocol/server-filesystem` | 参考挖掘、仓库下载 |
| `fetch` | pip / uvx `mcp-server-fetch` | 参考挖掘 |
| `github-downloader` | `tools/git_command.py` | 仓库下载 |
| `document-segmentation` | `tools/document_segmentation_server.py` | 分段、索引模式实现循环 |
| `code-implementation` | `tools/code_implementation_server.py` | 实现循环 |
| `code-reference-indexer` | `tools/code_reference_indexer.py` | 索引模式实现循环 |
| `command-executor` | `tools/command_executor.py` | 建文件树 |

快速模式只需要 `document-segmentation`、`code-implementation`、`command-executor` 三个。规划器不连任何服务器，论文内容或排名靠前的分段直接放在消息里。

## 6. 权限

`build_permission_engine(security, cwd=..., default_mode=...)` 返回的对象要有 `.mode`（`PermissionMode`）和 `.evaluate(name, args) -> (PermissionDecision, reason)`。`seams.harness.FullAutoEngine` 是合规的最小实现，全部放行。无人值守跑就用它。

`TerminalApprover` 只在 `mode` 不是 `FULL_AUTO` 时被构造，`as_async()` 是真代码，包装 `__call__` 到线程里。

## 7. Logger

管线接一个 `logger` 参数。约定是 loguru 风格：`logger.info("x={}", value)`。传 loguru 的 logger 直接用；传 stdlib `logging.Logger` 也行，`workflows/environment.py` 会用 `_BraceLoggerAdapter` 把 `{}` 先格式化。引擎里还有大量 `print()`，原样保留。

## 8. 调用管线

```python
from workflows.agent_orchestration_engine import execute_multi_agent_research_pipeline

result = await execute_multi_agent_research_pipeline(
    input_source="/path/to/paper.pdf",   # 或 URL
    logger=logger,
    progress_callback=on_progress,       # 可选
    enable_indexing=True,                # False = 快速模式
    task_id=None,                        # 给定则复用同名任务目录（resume）
    plan_review_callback=on_review,      # 可选，None = 不审阅
    workflow_root="/path/to/workspace",  # 缺省用 config.workspace.root
    strict_outcomes=False,               # True = 严格模式
)
```

返回字典的形状见 [ARTIFACTS.md](ARTIFACTS.md) 第 3 节。

**`progress_callback(percent: int, message: str, *extra)`**：同步函数。`percent` 是硬编码的步骤进度（1、4、25、40、50、65、66、70、75、80、85、100），失败时是 `0` 且第三个参数是错误文本。实现循环内会以 85 反复报告文件进度。

**`plan_review_callback(request) -> decision`**：异步。`request` 是一个字典：

```python
{
  "interaction_type": "plan_review",
  "title": "Review Implementation Plan",
  "description": "...",
  "required": False,
  "timeout_seconds": 1800,
  "data": {
    "plan": "<完整蓝图文本>",
    "plan_preview": "<前 80 行>",
    "plan_path": "...", "paper_dir": "...",
    "modification_round": 0, "max_rounds": 3,
    "plan_validation": {"valid": True, "missing_sections": [], ...},
    "last_error": None,
  },
  "options": {"confirm": "...", "modify": "...", "replace": "...", "cancel": "..."},
}
```

返回字典的 `action` 取 `approve`（别名 `confirm`、`approved`、`continue`）、`modify`（附 `feedback` 文本，管线会调模型修订蓝图再问一次）、`replace`（附 `plan` 文本，直接替换）、`cancel`（管线抛 `PlanReviewCancelled`）、`skip`（或 `skipped: True`，视为批准）。`data` 子字典里的同名键也认。

## 9. 已知的接入陷阱

- **`Agent.__aenter__` 的 anyio 取消作用域**：DeepCode 用一个独立的 supervisor task 开关 MCP stdio 会话，因为 `mcp` SDK 的 `stdio_client` 在哪个 task 里进入就必须在哪个 task 里退出，否则关闭时会向调用方注入一个 `CancelledError`。`core/compat/agent.py` 里 `_close_registry_quietly` 和 `_supervise_mcp` 是解决办法，接入方若用同一个 SDK 会遇到同样的问题。
- **代码记忆钩子替换 `messages`**：`_apply_memory_optimization` 做的是 `context.messages[:] = ...`。循环若在每轮开始时复制了一份消息列表，替换就丢了。
- **`filesystem` 服务器的根目录**：`environment.py` 会把工作区路径追加进 `config.mcp_servers["filesystem"].args`，这依赖 `KernelRuntime` 持有的是同一个 dict 对象。`KernelRuntime.__init__` 已经保证这一点，不要在 `provider_for` 或 `__aenter__` 里复制配置。
- **`code-implementation` 服务器的工作目录**：`code_implementation_workflow._initialize_mcp_agent` 在连上后先调 `set_workspace` 工具把服务器切到生成代码目录。若接入方的工具包装改了参数名，这一步会失败但不会报错。
- **闭环钩子挂点**：验证失败的结果目前不回灌。要做闭环，位置在 `_run_kernel_implementation` 的 `should_stop` 和 `inject_followups` 两个闭包：前者改成"验证全过才返回完成"，后者在验证失败时注入失败输出。详见 [LOGIC.md](LOGIC.md) 已知缺口一节。
