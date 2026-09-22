# DeepCode：论文提交时的版本（v1.0.7）与本仓库基线（21ebc57f）的差异

写于 2026-09-18。对象是 Paper2Code 业务层（`workflows/ tools/ prompts/ utils/`），数据来自
`~/Documents/search/DeepCode`（HKUDS/DeepCode 的克隆）的 `git diff v1.0.7 21ebc57f`。

| | v1.0.7 | 21ebc57f（本仓库基线 / DeepEvol 本线 vendored） |
| --- | --- | --- |
| 日期 | 2025-11-18（"release the new version of deepcode 1.0.7"，论文提交期） | 2026-09-16，之后 403 个提交、业务层 42 文件 +6823/−9697 |
| Agent 底座 | `mcp_agent`（ParallelLLM、RequestParams、MCP 工具） | 自研 AgentRunner kernel（`df4127d4`，2026-07）；模型目录钳制 `max_tokens`（deepseek 裸名 → 8192，本仓库补丁 `manualModels` 32768） |

## 逐阶段

| 阶段 | v1.0.7 | 21ebc57f | 影响 |
| --- | --- | --- | --- |
| 输入获取 | 两个 LLM agent（research analyzer + resource processor）解析输入、下载、转 markdown | 确定性 `acquire_input_artifact`（`710c5e5e`，2026-04-20） | PaperBench 直接给 `paper.md`，两边等价 |
| 分段 | `document_segmentation_server`，同一套启发式（max_segments 3/次） | 基本相同 | — |
| **规划** | **fan-out**：Concept + Algorithm 两个分析 agent 并行，`ParallelLLM` 汇入 planner；**整篇论文原文预载进提示**（`paper_content = f.read()`，不截断）；Algorithm agent 带 brave 网页搜索；`max_iterations` 5 | **单 agent**（`c9090c1a`，2026-04-21，"avoids fan-out deadlocks"），无工具无搜索；分段模式下只预载 `_load_document_segments_context(max_segments=8, max_chars=24000)` —— sapg 59.5k 字符只进了 4 段/24k | **这是回退**。本线 VENDOR 11 恢复了 fan-out 并按上下文窗口给预算（整篇进），S9 两边分数从 0.34/0.32 升到 0.64–0.72，Figure 7/8 子树从 0 变 1 —— 基本就是把规划器的输入退回到 v1.0.7 的量 |
| 规划审阅 | 无 | `workflows/interactions/plan_review.py`，无人值守默认通过 | 中性 |
| 参考仓库挖掘 / 下载 / 索引 | 同一套 prompt；`code_indexer.pre_filter_files` 已存在 | 同；并发索引改真 asyncio task（`a9252a84`） | 本线 VENDOR 10 修的预筛截断在两个版本都会出现 |
| **写码** | `code_implementation_workflow_index.py`（索引模式）：工具面 **仅 `write_file` + `search_code_references`**，800 次迭代，无墙钟；`memory_agent_concise` | 合并为一个 `CodeImplementationWorkflow(enable_indexing=True)`：**同样的工具面、同样 800 次**；新增 2 h 墙钟、`LoopDetector`、stall 预算、写完后可选本地验证（`require_verification`，跑发现的测试）、命令走沙箱 | agent 文件 `code_implementation_agent.py` 1117 行两边逐字相同；prompt 清单相同（只删了输入分析/下载两个 prompt）。写码能力本身没变 |
| 单次输出上限 | `get_token_limits()` 读配置（qwen-max 时代默认） | 同函数改读 defaults + 目录钳制 | 本仓库补丁固定 32768（VENDOR 12：8192 会截断长 `write_file`） |

## 结论

1. 论文时的 DeepCode 与今天的 DeepCode，**写码循环一样**，差在**规划器看到多少论文**：v1.0.7 整篇 + 双 agent + 搜索，21ebc57f 只给 24k 字符、单 agent。本线/本仓库的 16 文件补丁（VENDOR 11）把这一点补回去了，基线 `vexp1/vexp2`（0.7156 / 0.6910）是补丁开着跑的。
2. 因此"和论文提交时的版本比"在规划输入这一维已经基本对齐；没对齐的是 v1.0.7 的 **brave 搜索**（本仓库口径不联网搜索，两边都关）和 `mcp_agent` 底座的调用细节（重试、工具格式）。
3. 若要一个真正的 v1.0.7 数字：在本仓库另起 `DeepCode-v1.0.7` 目录按同口径跑一次 sapg（需要 `mcp_agent` 依赖与旧配置格式，模型 Vision-Exp，判分 Flash），≈ 40 min + ¥5–10 判分；它和 `vexp1/vexp2` 的差在 ±0.03 内就说明补丁已经把回退补平。
