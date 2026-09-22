# 输出文档

复现管线的全部产物都落在任务目录里，管线本身只返回一个字典。本文按文件逐个说明：谁写、谁读、什么格式。示例片段取自一次真实运行：输入是一篇 2.7k 字符的合成论文 EMA-Detect，快速模式，无索引。

词汇按 [CONTEXT.md](../CONTEXT.md)。回调的形状在 [INTEGRATION.md](INTEGRATION.md) 第 8 节。

## 1. 任务目录

```
<工作区>/.deepcode/workflows/tasks/paper_<task_id>/
├── paper.<原扩展名>              获取：输入源的副本（本地文件复制，URL 下载）
├── paper.md                      获取：转换后的 Markdown，后续所有步骤的唯一输入
├── document_segments/            分段（仅当文档超过阈值）
│   ├── document_index.json       分段索引
│   └── <id>.md                   每个分段一个文件
├── planning_checkpoint.json      规划：规划器的最新检查点，成功后删除
├── planning_attempts.jsonl       规划：每次尝试一行
├── planning_result_meta.json     规划：最终结果元数据，计划审阅后再更新一次
├── initial_plan.txt              规划：蓝图
├── plan_versions/                计划审阅
│   └── initial_plan.v00.generated.txt   每个版本一个文件
├── plan_review_history.jsonl     计划审阅：每个事件一行
├── reference.txt                 参考挖掘：模型的自由文本分析（快速模式下是一句跳过说明）
├── github_download.txt           仓库获取：模型的自由文本报告（同上）
├── code_base/                    仓库获取：克隆下来的参考仓库，每个一个子目录
├── indexes/                      代码检索
│   └── <repo>_index.json         每个参考仓库一个索引
├── codebase_index_report.txt     代码检索：结果字典的 str()
├── generate_code/                实现循环：生成代码目录
│   └── <项目名>/                 规划器决定的顶层目录名
├── implement_code_summary.md     代码记忆
├── code_implementation_report.txt 实现循环：结果字典的 str()
└── logs/
    ├── llm.jsonl                 每次模型调用一行
    ├── mcp.jsonl                 每次 MCP 工具调用一行
    └── mcp_server_<name>.log     每个服务器的 stderr
```

`task_id` 缺省是 8 位十六进制随机串；Desktop 传的是 `wfr_<uuid>`。传入已存在的 `task_id` 触发 resume：跳过获取，若 `initial_plan.txt` 已存在且校验通过则跳过规划。

`.txt` 结尾的三个报告文件内容是 Python 字典的 `str()`，不是 JSON。读取要用 `ast.literal_eval`。

## 2. 逐文件说明

### paper.md

获取步骤写，工作区合成、分段、规划读。转换路径见 [LOGIC.md](LOGIC.md) 第 3 节。PDF 经 pypdf 逐页抽文本时格式是：

```
# Extracted from paper.pdf

*Total pages: 12*

---

## Page 1

<正文>

## Page 2
...
```

Markdown、TXT、HTML、DOCX 经内置转换器则保留原有标题结构。管线要求这个文件必须存在，否则在获取步骤就抛错。

### document_segments/document_index.json

分段步骤写，规划步骤读（`_load_document_segments_context` 取 `relevance_scores.code_planning` 最高的前 8 段、总量不超过 24000 字符）。

```json
{
  "document_path": "...",
  "document_type": "research_paper",
  "segmentation_strategy": "semantic_research_focused",
  "total_segments": 14,
  "total_chars": 61230,
  "created_at": "...",
  "segments": [
    {
      "id": "seg_003",
      "title": "3.1 Exponential moving statistics",
      "content": "...",
      "content_type": "methodology",
      "keywords": ["ema", "variance", "streaming"],
      "char_start": 4100, "char_end": 6820, "char_count": 2720,
      "relevance_scores": {"concept_analysis": 0.6, "algorithm_extraction": 0.9, "code_planning": 0.85},
      "section_path": "3.1"
    }
  ]
}
```

`document_type` 取 `research_paper`、`technical_doc`、`algorithm_focused`、`general`。`segmentation_strategy` 取 `semantic_research_focused`、`algorithm_preserve_integrity`、`concept_implementation_hybrid`、`semantic_chunking_enhanced`、`content_aware_segmentation`。`content_type` 是 `introduction`、`methodology`、`algorithm`、`results` 等自由标签。三个相关性分数对应三种查询类型，当前只有 `code_planning` 被用到。

EMA-Detect 那次没有这个目录：文档 2734 字符，低于 50000 阈值，走全文模式。

### planning_attempts.jsonl

规划步骤写，每次尝试追加一行。真实记录：

```json
{"attempt": 1, "max_retries": 3, "mode": "traditional", "segmentation": false, "segmented_context": false, "max_iterations": 2, "max_tokens": 8192, "temperature": 0.3, "status": "success", "result_chars": 10137, "tools_used": [], "usage": {"prompt_tokens": 2435, "completion_tokens": 2599, "total_tokens": 5034, "cached_tokens": 2432}, "runner_error": null, "completeness_score": 1.0, "plan_validation": {"yaml_valid": true, "yaml_error": null, "required_sections": ["file_structure", "implementation_components", "validation_approach", "environment_setup", "implementation_strategy"], "missing_sections": [], "sections_found": 5, "valid": true}, "updated_at": "2026-09-16T12:05:09.192466+00:00"}
```

`status` 取 `success`、`incomplete`、`timeout`、`error`。`mode` 是 `segmented` 或 `traditional`。`completeness_score` 是 0 到 1 的启发式分数，低于 0.8 视为截断并重试。

### planning_result_meta.json

规划步骤写一次，计划审阅步骤再更新一次加上 `plan_review` 子字典。`status` 为 `success` 时 `source` 是 `generated`、`existing`（resume 复用）或 `coerced_from_freeform`（非严格模式下模型没给出合法 YAML，被机械转成最小蓝图）；严格模式下这种情况直接抛异常。

### initial_plan.txt

规划步骤写，计划审阅可能改写，实现循环和代码检索读。整个文件是一个 YAML 代码块，顶层键 `complete_reproduction_plan`，五个必需段落。真实开头：

```yaml
complete_reproduction_plan:
  paper_info:
    title: "EMA-Detect: A Minimal Streaming Anomaly Detector Based on Exponential Moving Averages"
    core_contribution: "A lightweight streaming anomaly detector using exponential moving average and variance with O(1) time/memory complexity"

  # SECTION 1: File Structure Design
  file_structure: |
    ema-detect/
    ├── ema_detect/
    │   ├── __init__.py          # Package initialization, exports main classes
    │   ├── detector.py          # Core EMA-Detect algorithm implementation
    │   ├── statistics.py        # Exponential moving statistics (mean, variance)
    │   └── baseline.py          # Global z-score baseline for comparison
    ├── experiments/
    ...

  # SECTION 2: Implementation Components
  implementation_components: |
    **Component 1: Exponential Moving Statistics (statistics.py)**
    - Purpose: Maintain EMA mean and variance per Equations 1 and 2
    - Class: `EMAStatistics`
      - Methods:
        - `update(self, x: float)`: Apply Eq. 1 and Eq. 2:
          ```
          mu_new = alpha * x + (1 - alpha) * self.mu
          ...
```

五段的键名固定：`file_structure`、`implementation_components`、`validation_approach`、`environment_setup`、`implementation_strategy`，缺一个校验就失败。`file_structure` 是块标量里的树形文本，代码记忆和代码检索都从它解析文件清单。注意块标量里可以嵌套代码栏，这是补丁修过的解析陷阱。

### plan_versions/ 与 plan_review_history.jsonl

计划审阅步骤写。每个版本文件名 `initial_plan.v<两位序号>.<来源>.txt`，来源是 `generated`（规划器产出）、`ai`（按反馈修订）、`manual`（接入方替换）。历史文件每行一个事件，`event` 取 `review_started`、`review_requested`、`review_response`、`review_approved`、`review_auto_approved`、`review_cancelled`、`revision_attempt`、`revision_failed`、`replacement_rejected`。真实片段：

```json
{"timestamp": "2026-09-16T12:05:09.214156+00:00", "event": "review_requested", "interaction": 1, "round": 0, "validation": {"valid": true, ...}}
{"timestamp": "2026-09-16T12:05:25.081264+00:00", "event": "review_response", "interaction": 1, "round": 0, "action": "approve", "skipped": false}
```

### reference.txt 与 github_download.txt

参考挖掘和仓库获取两步写，都是模型的自由文本，没有固定格式。下一步把前者原样作为消息喂给下载 Agent。快速模式下各写一句：

```
Reference intelligence analysis skipped - fast mode enabled for optimized processing
```

### indexes/<repo>_index.json

代码检索步骤写，实现循环通过 `search_code_references` 工具读。每个参考仓库一个文件：

```json
{
  "repo_name": "some-repo",
  "total_files": 42,
  "file_summaries": [
    {"file_path": "src/ema.py", "file_type": "python", "main_functions": ["update", "score"],
     "key_concepts": ["exponential moving average"], "dependencies": ["numpy"],
     "summary": "...", "lines_of_code": 120, "last_modified": "..."}
  ],
  "relationships": [
    {"repo_file_path": "src/ema.py", "target_file_path": "ema_detect/statistics.py",
     "relationship_type": "direct_match", "confidence_score": 0.85,
     "helpful_aspects": ["..."], "potential_contributions": ["..."],
     "usage_suggestions": "..."}
  ],
  "analysis_metadata": {...}
}
```

`relationship_type` 只有 `direct_match`、`partial_match`、`reference`、`utility` 四种，权重在 `tools/indexer_config.yaml`。`target_file_path` 必须是蓝图 `file_structure` 里的路径。

### codebase_index_report.txt

代码检索步骤写，字典的 `str()`。`status` 取 `success`、`skipped`、`warning`、`error`。真实内容（快速模式）：

```
{'status': 'skipped', 'reason': 'fast_mode_enabled', 'message': 'Codebase intelligence orchestration skipped for optimized processing'}
```

### generate_code/

实现循环写。第一步由 `command-executor` 按蓝图 `file_structure` 用 shell 命令建好空文件树，之后模型逐个填内容。顶层目录名来自蓝图（EMA-Detect 那次是 `ema-detect/`），这就是验证步骤需要 `resolve_project_root` 下探一层的原因。

### implement_code_summary.md

代码记忆。实现循环每写完一个文件追加一条，`read_code_mem` 工具按文件路径切出对应段落返回给模型。真实的一条：

```markdown
## IMPLEMENTATION File ema-detect/ema_detect/statistics.py; ROUND 1
================================================================================

# Code Implementation Summary
**Generated**: 2026-09-16 20:06:06
**File Implemented**: ema-detect/ema_detect/statistics.py

**Core Purpose**:
- Implements exponential moving statistics (EMA mean and variance) per Equations 1 and 2 ...

**Public Interface**:
- Class `EMAStatistics`: Maintains exponential moving average mean and variance | Key methods: `__init__`, `initialize`, `update`, `get_mu`, `get_var`, `get_std` | Constructor params: `alpha: float`
  ...

**Internal Dependencies**:
- External packages: None (pure Python standard library only)

- Architecture decisions: ...
- Cross-File Relationships: `detector.py` will instantiate `EMAStatistics` ...

---
*Auto-generated by Memory Agent*
```

段落标题 `## IMPLEMENTATION File <路径>; ROUND <n>` 是 `read_code_mem` 的定位锚。四个字段对应论文 CodeMem 条目的核心目的、公共接口、依赖边；`Next Steps` 段在提示词里要求但写入文件前被剥掉，单独存在内存里。

### code_implementation_report.txt

实现循环写，字典的 `str()`，和管线返回值里的 `implementation` 子字典同源。真实内容（截断）：

```
{'status': 'incomplete', 'inner_status': 'unverified', 'generation_status': 'completed',
 'abort_reason': 'no_tests_discovered', 'files_completed': 15, 'total_files': 15,
 'unimplemented_files': [], 'iterations': 13, 'elapsed_seconds': 151.73,
 'plan_file': '.../initial_plan.txt', 'target_directory': '.../paper_wfr_...',
 'code_directory': '.../generate_code', 'results': {'file_tree': '...', 'code_implementation': '...'},
 'verification': [], 'mcp_architecture': 'standard'}
```

这一条正好展示了未打补丁前的验证根目录问题：15 个文件全写完，`generate_code/ema-detect/tests/` 下有三个测试文件，但验证只看了 `generate_code/`，报 `no_tests_discovered`。

### logs/

`llm.jsonl` 每行的键：`timestamp`、`provider`、`model`、`duration_ms`、`status`、`finish_reason`、`request_preview`、`response_preview`、`error`、`response_sha256`。`mcp.jsonl` 每行的键：`timestamp`、`task_id`、`server`、`tool`、`duration_ms`、`status`、`arguments_preview`、`result_preview`。这两个文件由 DeepCode 的观测层写，本目录的 `seams/observability.py` 是空操作，接入方不接就没有。

## 3. 管线返回值

`execute_multi_agent_research_pipeline` 返回：

```python
{
  "status": "completed" | "incomplete" | "completed_with_warnings" | "error",
  "summary": "<多行人类可读摘要>",
  "paper_dir": "<任务目录>",
  "implementation": {
    "status": "success" | "incomplete" | "warning" | "error",
    "inner_status": ...,
    "generation_status": ...,
    "abort_reason": ...,
    "files_completed": int,
    "total_files": int,
    "unimplemented_files": [str],
    "code_directory": str,
    "verification": [ {...}, ... ],
  },
}
```

计划审阅返回 `cancel` 时不返回字典，抛 `PlanReviewCancelled`。其他异常原样向上抛，`progress_callback(0, "Pipeline failed", 错误文本)` 先被调一次。

### 3.1 顶层 status

| 值 | 条件 |
| --- | --- |
| `completed` | `implementation.inner_status == "completed"` |
| `incomplete` | `implementation.status == "incomplete"`；或 `warning` 且严格模式 |
| `completed_with_warnings` | `implementation.status == "warning"` 且非严格模式 |
| `error` | 其余 |

### 3.2 implementation.status

| 值 | 条件 |
| --- | --- |
| `success` | 生成完成，且（非严格模式）或（严格模式下发现了测试且全部通过） |
| `incomplete` | 生成未完成；或严格模式下没发现测试或有测试失败 |
| `error` | 实现步骤抛异常 |

### 3.3 implementation.inner_status 与 abort_reason

`inner_status` 先取实现循环的 `generation_status`，生成完成后再按验证结果覆盖：

| inner_status | abort_reason | 含义 |
| --- | --- | --- |
| `completed` | `None` 或 `all planned files implemented` | 全部文件写完；严格模式下还表示测试全过 |
| `unverified` | `no_tests_discovered` | 严格模式，写完了但没找到测试命令 |
| `test_failed` | `generated_tests_failed` | 严格模式，至少一个测试失败 |
| `incomplete` | `model stopped while planned files remain unimplemented` 或 `LLM request failed ...` | 模型停了但还有文件没写 |
| `max_iterations` | `reached max_iterations=800 without completion` | 800 轮用完 |
| `max_time` | `wall-clock budget exhausted after Ns (limit 7200s)` | 两小时用完 |
| `aborted` | `loop_detector ...: <原因>` | 循环检测器判定卡死 |

### 3.4 implementation.verification

列表，每个发现的测试命令一项：

```python
{"command_id": "pytest", "command": ["python3", "-m", "pytest", "-q"],
 "passed": bool, "exit_code": int, "timed_out": bool, "duration_ms": int,
 "stdout": "<末尾 64 KiB>", "stderr": "<末尾 64 KiB>", "output_truncated": bool}
```

非严格模式下永远是空列表，因为不跑验证。`command_id` 取 `pytest`、`unittest`、`npm-test`、`cargo-test`。

## 4. 进度回调看到的序列

正常一次完整运行（有索引）的 `percent` 序列：

```
1  4  25  40  50  65  [66]  70  75  80  85  85  85 ...  100
```

66 只在有计划审阅回调时出现。85 在实现循环里每写完一个文件重复一次，`message` 形如 `Code implementation: 7/15 planned files completed`。快速模式下 70、75、80 仍会报告，只是消息说跳过。
