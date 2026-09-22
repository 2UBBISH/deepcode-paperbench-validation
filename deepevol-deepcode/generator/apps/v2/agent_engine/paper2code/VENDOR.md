# VENDOR.md — Paper2Code engine

This directory is a vendored copy of the DeepCode Paper2Code business layer.
It is an engine: the line that drives it lives in `apps/v2/agent/paper2code/`.
Nothing here may import from `apps/v2/agent/`, from the retired `Agent/`
tree, or from DeepCode's `core/`.

## Source

| item | value |
| --- | --- |
| copied from | `search/paper2code-kernel` @ `c821130` (2026-09-16) |
| that tree's upstream | HKUDS/DeepCode `21ebc57fbcab3e3a238976771a0e06115aa30da6` (also in `UPSTREAM_COMMIT`) |
| copied | `workflows/ tools/ prompts/ utils/ seams/ support/ CONTEXT.md docs/ README.md UPSTREAM_COMMIT` |
| dropped | `.git`, `.gitignore`, `__pycache__` |
| added | this file, an empty `__init__.py` |

The kernel's own `CONTEXT.md` and `docs/` describe the engine from the
inside (its 步骤, its seams, its artifacts). They are authoritative for the
engine and say nothing about this repository's phases or asset roles.

## Import rewrite (C1)

The kernel's six top-level packages (`workflows tools prompts utils seams
support`) were imported by bare name. Every such import, including the
indented lazy ones, now uses the full package path, following the
PaperOrchestra precedent (no `sys.path` games):

```
^(\s*)(from|import) (workflows|tools|prompts|utils|seams|support)(?=[\s.])
  -> \1\2 apps.v2.agent_engine.paper2code.\3
```

28 files changed. Three `sys.path.insert/append` lines that existed only to
make those bare imports work were removed at the same time
(`workflows/codebase_index_workflow.py`,
`workflows/code_implementation_workflow.py`,
`workflows/agents/code_implementation_agent.py`).

`seams/mcp_servers.json` keeps its `{KERNEL_ROOT}` placeholders: it
documents how the upstream stdio servers would be launched. This line never
launches them (see below).

## Dependencies (C1)

Added to `pyproject.toml` main dependencies: `loguru`, `mcp`, `aiohttp`,
`aiofiles`, `asyncssh` (the last one is for the vendored `remote_relay`
under the line, not for the engine).

`mcp` is imported for its `FastMCP` decorator and its `Tool` types only.
No MCP server process is started anywhere in the V2 runtime: the line
wraps the decorated functions in-process (`apps/v2/agent/paper2code/tools/`).
The V2 "no MCP" boundary is about processes and adapters and is unaffected.

`tiktoken` is deliberately not added; the engine degrades to a character
estimate when it is missing.

## Patches applied on top of `c821130`

Each entry names the file, the change, and the step that made it. Files not
listed are byte-identical to the source tree except for the import rewrite.

| # | file | change | step |
| --- | --- | --- | --- |
| 1 | all `.py` | import rewrite and `sys.path` removal (above) | C1 |
| 2 | `seams/compat.py` | `apply_tool_filter` matches `tool_filter` servers against the sanitized prefix (`-` → `_`), because the line's registry exposes sanitized names | C2 |
| 3 | `tools/code_indexer.py` | pre-filter / analysis / relationship `max_tokens` read `DEEPCODE_PREFILTER_MAX_TOKENS`, `DEEPCODE_ANALYSIS_MAX_TOKENS`, `DEEPCODE_RELATIONSHIP_MAX_TOKENS` (upstream defaults kept) | C2 |
| 4 | `workflows/agent_orchestration_engine.py` | `github_repo_download`: tool-first instruction, `server_names=["github-downloader"]`, `tool_filter={"github-downloader": {"git_clone"}}`, framed report message, one corrective retry when nothing was cloned, `DEEPCODE_DOWNLOAD_MAX_TOKENS` / `DEEPCODE_DOWNLOAD_MAX_ITERATIONS`. `paper_reference_analyzer`: `DEEPCODE_REFERENCE_MAX_TOKENS` / `DEEPCODE_REFERENCE_MAX_ITERATIONS`. `automate_repository_acquisition_agent`: empty or missing `code_base/` raises instead of printing | C2 |
| 5 | `workflows/agents/document_segmentation_agent.py` | the segmentation prompt demands a real tool call; the result is an error unless `document_segments/document_index.json` exists | C2 |
| 6 | `workflows/code_implementation_workflow.py` | `_max_wall_seconds()` reads `DEEPCODE_MAX_WALL_SECONDS` at call time; `LoopDetector` takes `DEEPCODE_STALL_THRESHOLD` when set | C2 |
| 7 | `workflows/codebase_index_workflow.py` | hoist a `split("\n")` out of an f-string expression (Python 3.11 rejects the backslash) | C2 |

| 8 | `workflows/code_implementation_workflow.py` | `CodeImplementationWorkflow.__init__` takes `verification_runner` (default: module-level `VERIFICATION_RUNNER`, then `support.verification.run_verification`); `_verify_generated_code` awaits an async runner or threads a sync one. The line points it at the execution port | C5 |
| 9 | `seams/observability.py` | `set_task_dir(task_id, task_dir)` — the no-op stub took one argument while `workflows/environment.py` passes two | C7 |
| 10 | `tools/code_indexer.py` | pre-filter asks for `file_path` + `confidence` only (the two long fields were never read; a 263-file repository overran the output budget and the parse failure silently indexed every file), the two "recommendation systems / GNN / diffusion" sentences replaced by domain-neutral wording (a leftover of the authors' example project), the log reports the count actually selected, and `_call_llm` treats `finish_reason == "length"` as a failure that goes through the retry loop. Same hunk in the validation repository's patch (PLAN-3 item 1, 2026-09-17) | PLAN-3 |
| 11 | `workflows/agent_orchestration_engine.py`, `tools/document_segmentation_server.py` | planning, two things (PLAN-3 items 7 / 7b, 2026-09-18): (7) the planning fan-out upstream removed in `c9090c1a` — `_generate_plan_with_fanout` runs the Concept and Algorithm analysis agents (prompts unchanged in `prompts/code_prompts.py`) on the planner's message and appends their outputs under `# Worker outputs`, as the legacy `ParallelLLM` did; behind `DEEPCODE_PLANNING_FANOUT=1`, default off. (7b) `_load_document_segments_context` takes `budget_chars` from `_planner_segment_budget_chars` = (`DEEPCODE_PLANNER_CONTEXT_WINDOW` − max_tokens − 12k prompt reserve) × 0.85 × 3 chars/token: when the whole paper fits, every segment in document order; otherwise upstream's relevance ranking without the 8-segment cap; unset = upstream's 8 / 24 000. The segmenter gives an appendix after the references (`\section*{A. …}` / `# Appendix`) its own `appendix` segment with high `code_planning` relevance — unconditional (sapg: 9 → 10 segments, appendix 0.86). The line sets both from `run.json` (`planning_fanout`, `context_window`) in `apply_env_defaults`. Same hunks in the validation repository's patch (its README §5.1 A) | PLAN-3 |
| 12 | `workflows/code_implementation_workflow.py` | `_implement_max_tokens()`: the implementation loop's per-call `max_tokens` reads `DEEPCODE_IMPLEMENT_MAX_TOKENS` (upstream's 8192 kept as the default). Upstream's value cuts a `write_file` call whose file runs past ~8k tokens — the tool-call JSON arrives truncated, the provider reports an error and the whole implementation stops with files unwritten (sapg on DeepSeek-V4-Flash-Vision-Exp, 2026-09-18, a 30 KB `train_baselines.py`). The line's `ENV_DEFAULTS` and the validation repository's `run_trial.sh` both set 32768 | PLAN-3 S9 |
| 13 | `workflows/paper_readback.py` (new), `workflows/agent_orchestration_engine.py`, `workflows/code_implementation_workflow.py`, `workflows/agents/memory_agent_concise.py` | **Paper fidelity** (PLAN-3 T14, 2026-09-19; SA-Bench found the whole deepcode-vs-basic gap in the formula dimension: the blueprint compresses or loses equations and the coding agent never reads the paper again). Behind `DEEPCODE_PAPER_FIDELITY=1`, default off (upstream byte-identical when unset; the line's `ENV_DEFAULTS` sets 1): (a) `_planner_instruction` and the fan-out `AlgorithmAnalysisAgent` get an addendum — every formula / loss / update rule / hyper-parameter quoted verbatim (the LaTeX as in `paper.md`) with `Source: §<number> <heading>`, section-2 budget lifted for the quotes; (b) the coding loop gets `read_paper(section=…, query=…, part=…)` — a deterministic index over the paper's own headings (`\section*{}` / `#`) returning one section's text with its display equations, paged at 3 000 chars, registered next to the MCP-aliased tools (not routed through the legacy tracker, not recorded into the memory agent: the clean slate after `write_file` drops it); the system prompt, the per-round knowledge-base message and the success guidance tell the agent to call it for a file's `Source` sections before `write_file`; `_RunState.paper_reads` counts calls and lands in the run result (`paper_reads`). `plan_fidelity_stats()` counts equations / `Source` refs in a plan against the paper (the line records it as `plan.fidelity`). Tests: `test_engine_patches.py` VENDOR 13 block | PLAN-3 T14 |
| 14 | `workflows/source_fidelity.py` (new), `workflows/paper_readback.py`, `workflows/code_implementation_workflow.py`, `workflows/agent_orchestration_engine.py` | **ADR 0004 (2026-09-20/21, supersedes the verbatim-quote form of 13 and the whole of ADR 0003)**: the planner writes file path + `Source: §x.y` per Section 2 paragraph and does *not* copy formulas; `compile_manifest` reads the pointers back mechanically (file → sections, glue, unmatched) into `source_manifest.json`; `FidelitySession` records read receipts in `source_trace.json`; `read_paper(file_path=…)` walks the file's unread (section, page) list; `write_file` is refused (`SOURCE_READ_REQUIRED`) until every page of every bound section was read in an earlier model turn; `audit()` replays the trace and is recorded, never a gate. `DEEPCODE_PAPER_FIDELITY` unset = upstream byte-identical. Tests: `test_source_fidelity.py`, `test_engine_patches.py` | ADR 0004 |

### Validation-repo hunks deliberately not taken (C2)

Source: `2UBBISH/deepcode-paperbench-validation`, `patches/deepcode_local_changes.patch` (against upstream `e0767d0`).

| hunk | why not |
| --- | --- |
| `core/agent_runtime/tools/mcp.py` (URL denylist, repeat-fetch ledger, name sanitizing) | the engine has no such file; the line implements all three in `apps/v2/agent/paper2code/tools/` |
| `core/compat/request_params.py`, `core/providers/base.py` (retry mode and delays) | provider concerns; the line's `provider.py` implements the same semantics |
| `utils/loop_detector.py` (consecutive `write_file` exempt) | superseded by the kernel's own fix: write tools are keyed by argument digest before reaching the detector |
| `agent_orchestration_engine.py` fix-① `DEEPCODE_PLAN_COVERAGE_CHECK` | the audit prompt carries scoring meta-knowledge; not comparable with upstream |
| `agent_orchestration_engine.py` index reuse on resume | the line's driver decides reruns per phase; a silent skip inside the engine would defeat `rerun --phase index` |
| `memory_agent_concise.py` fix-② `DEEPCODE_ALLOW_PLAN_EXTENSION` | same reason as fix-① |
| `code_implementation_workflow.py` fix-③ `DEEPCODE_POSTWRITE_COMPILE` | compiles into the source tree; verification is the line's job |
