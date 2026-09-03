# 独立代码审查:DeepCode 本地改动(相对上游 HEAD e0767d0)

审查日期:2026-09-03 · 审查方式:只读(`git diff HEAD`、`git show HEAD:<file>` 对照、脚本静态审读、无副作用的路径解析实测、日志取证)。
被审对象:`/home/deepevol/deepevol/DeepCode`(12 个已跟踪文件改动 + 未跟踪 `RICE/`)、`/home/deepevol/deepevol/frontier-evals/project`(3 个改动 + 5 个未跟踪)、`deepcode_test/scripts/` 5 个脚本。

## 总体结论(先说)

- **纪律 A(默认行为与上游完全一致)不成立。** 12 处改动完全没有 env 门控,任何一轮(包括所谓"官方默认"基线轮)都在跑着改过的行为;另有 3 处"env 可调"的默认值本身已经漂移(参考挖掘 maxTokens 4096→8192、写码墙钟 7200→14400、stall 300→1800)。日志取证:repeat-fetch 节流在 15 个 trial 日志里触发 56 次,覆盖 fre trial1/2/3/6 与 rice trial1/2/k1/k2——即"基线轮"跑的并不是上游行为。
- **纪律 B(不注入评分知识)基本成立但有一处越线。** 全部 diff 中没有 rubric/PaperBench/裁判/权重字样,fix-①② 的提示词不含任何论文专有信息;但两处提示词都写了 **"Graders assign separate credit to each (baseline); omitting them forfeits those points"**——这是对 PaperBench 评分结构("每个基线单独计分")的元知识注入,不是工程缺陷修复。它只出现在 env 开关内(默认关),所以只污染 fx 对照臂,不污染基线臂,但 fx 臂的结论必须如实标注这一点。
- **需要工程师立刻处理**:(1) 报告里"官方默认值一字未改"的表述必须改写,列出下文 🔴-1 的 12+3 项非门控差异,并说明所有臂共享;(2) fix-①② 去掉 "Graders…points" 一句,或把它记入实验报告;(3) `refresh_after_fx.sh:18` 明文 API key 立即轮换;(4) fix-③ 的 cwd 候选路径会编译 DeepCode 自己的 `setup.py/__init__.py/deepcode.py`(实测),要删掉该候选。

---

## 🔴 会导致错误结论 / 破坏纪律 A、B

### 🔴-1 纪律 A:12 处非门控行为改动 + 3 处默认值漂移

以下每一项在 env 全部未设置时的行为都与 `git show HEAD:<file>` 不同。

| # | 文件:行 | 上游(HEAD) | 现状(env 未设) | 性质 |
|---|---|---|---|---|
| 1 | `core/agent_runtime/tools/mcp.py:87,165-188,231` | 无重复抓取限制 | `fetch` 同一 URL 成功返回 2 次后第 3 次起被拒,返回 "ALREADY FETCHED…" | **检索方法改变**;且 key 只含 URL 不含 `start_index`,mcp-server-fetch 默认 `max_length=5000`,长页面(README、arXiv HTML)翻到第 3 页即被拒,拒绝文案还谎称"returned no further information"。日志:56 次触发,散布在基线轮 |
| 2 | `core/agent_runtime/tools/mcp.py:90-102` + 所有 `_sanitize_exposed_name` 调用点 | 工具名 `mcp_github-downloader_git_clone` | `mcp_github_downloader_git_clone` | 模型看到的工具名变了(有针对 Kimi 的合理动机,但非门控) |
| 3 | `core/compat/agent.py:155,192` | 前缀 `mcp_{srv}_` | `.replace("-","_")` | 与 #2 配套,非门控 |
| 4 | `utils/loop_detector.py:99` | 同名工具连续 5 次 → `should_stop` | `write_file`/`write_multiple_files` 豁免 | 上游会在连续 5 次 write_file 后中止整轮;现在永不。这是对上游"失控保护"的实质改动 |
| 5 | `workflows/agent_orchestration_engine.py:999-1019` | `instruction="Download github repo to the directory …/code_base"`,`server_names=["filesystem","github-downloader"]` | 全新的 5 条规则提示词;去掉 filesystem | **提示词整体重写,非门控** |
| 6 | 同文件 `:1045,1051` | 无 `max_iterations`(→ DEFAULT 8)、无 tool_filter | `max_iterations=40`,`tool_filter={"github-downloader":{"git_clone"}}` | 非门控 |
| 7 | 同文件 `:1068-1083` | 无 | code_base 为空时**再调一次 LLM** 纠正重试(新增提示词) | 非门控,额外一次模型调用 |
| 8 | 同文件 `:1131` | 参考挖掘 agent 无 `max_iterations`(→ 8) | `max_iterations=80` | 非门控;任务描述说的"40→80"实际是 **8→80** |
| 9 | 同文件 `:1555,1563` | code_base 为空只 `print` 继续跑 | `raise RuntimeError` fail-fast(`except` 块 `raise e` 会传到主流水线 → 整轮 error) | 非门控;改变了流水线终止条件 |
| 10 | 同文件 `:1604-1627` | 每次重建索引 | 若 `indexes/<repo>_index.json` 齐全则跳过 | 非门控(run_trial 每轮清目录故实际未触发,但代码层面违反 A) |
| 11 | `workflows/agents/document_segmentation_agent.py:132` | 无 | 提示词新增一整段 "CRITICAL: You MUST invoke …" | **提示词改动,非门控** |
| 12 | 同文件 `:163-178` | 只信 agent 回话,返回 success | 无 `document_index.json` → 返回 error → 上游走 `fallback_to_traditional`(全文规划) | 非门控;改变了分段失败时的分支(上游会带 `segments_available=True` 继续,规划期再兜底) |
| 13 | `workflows/agent_orchestration_engine.py:1121` | `maxTokens=4096` | env 默认 **8192** | **默认值漂移**。注释自称"默认保持 8192"是错的,上游是 4096 |
| 14 | `workflows/code_implementation_workflow.py:100` | `_MAX_WALL_SECONDS = 7200` | env 默认 **14400** | 默认值漂移 |
| 15 | `workflows/code_implementation_workflow.py:493` | `LoopDetector()` → stall 300s | env 默认 **1800** | 默认值漂移 |

证据(节选):
```
-        self._name = f"mcp_{server_name}_{tool_def.name}"
+        self._name = _sanitize_exposed_name(f"mcp_{server_name}_{tool_def.name}")
-                return {"status": "loop_detected", ...
+                if recent_tools[0] not in ("write_file", "write_multiple_files"):
-            maxTokens=4096,
+            maxTokens=int(os.environ.get("DEEPCODE_REFERENCE_MAX_TOKENS", "8192")),
-_MAX_WALL_SECONDS = 7200
+_MAX_WALL_SECONDS = int(os.environ.get("DEEPCODE_MAX_WALL_SECONDS", "14400"))
-        self.loop_detector = LoopDetector()
+        self.loop_detector = LoopDetector(stall_threshold=int(os.environ.get("DEEPCODE_STALL_THRESHOLD", "1800")))
```
```
$ grep -rl "short-circuited" deepcode_test/*/logs | wc -l   → 15 个日志(含 fre trial1/2/3/6, rice trial1/2/k1/k2)
```

**影响**:实验报告中"官方默认值一字未改"为假。所有臂(基线/fx)共享这 15 项差异,所以 fx-vs-基线的**相对**比较仍然公平,但任何"与上游 DeepCode 官方行为对比"或"与论文报告数字对比"的绝对结论都被这些差异混淆——尤其 #1(检索被节流)、#4(去掉 5 次 write_file 熔断)、#5-#9(下载/参考 agent 全套重写)直接作用在论文声称的"检索/规划"环节。

**建议**:要么把 15 项全部补上 env 门控(默认关),要么在报告里单列一节"所有臂共享的非官方改动清单",把它们从"工程缺陷修复"中区分出来。至少 #13-15 的注释与默认值必须改回上游值(4096/7200/300),由 `run_trial.sh` 显式 export 覆盖。

### 🔴-2 纪律 B:fix-①② 提示词注入评分结构元知识

- `workflows/agent_orchestration_engine.py:796-797`:`"Graders assign separate credit to each of these; a plan that omits them forfeits those points."`
- `workflows/agents/memory_agent_concise.py:1684-1685`:`"Graders assign separate credit to each baseline; omitting them forfeits those points."`

这两句告诉模型:有一个评分者,按基线逐项计分。这正是 PaperBench rubric 的树状叶子结构,不是论文里的信息,也不是工程缺陷。其余措辞("every BASELINE / comparison method the paper evaluates against"、"every experiment in the main body"、"environment/dataset")是泛指论文内容,没有 GC-IQL/GC-BC/OPAL 之类专名(专名只出现在代码注释,不进模型)。

**建议**:删掉两处 "Graders…points" 句,替换为 "The paper's baselines and experiments are part of the reproduction target"。若保留,实验报告必须写明 fx 臂含此句。

### 🔴-3 `refresh_after_fx.sh:18` 明文 API key

```
-H "Authorization: Bearer sk-<REDACTED>"
```
Paratera 的 key 硬编码在脚本里(`deepcode_test/` 虽非 git 仓库,但已进入交接文档目录)。另外 `frontier-evals/project/paperbench/.env.bak_siliconflow_0902` 含 5 个 provider 的 key,**未被 `.gitignore` 覆盖**(`git check-ignore` 无输出),`git status` 里以 `??` 出现,`git add -A` 就会提交。
**建议**:立即轮换 key;脚本改读 `$PB/.env`;`.env.bak_*` 移出仓库或加 ignore 规则。

### 🔴-4 fix-③ 路径解析会编译到 DeepCode 自己的源文件(实测)

`workflows/code_implementation_workflow.py:114` 第一个候选是 `os.path.join(os.getcwd(), rel_or_abs)`;写码时 cwd 是 DeepCode 仓库根(`run_trial.sh` 中 `cd "$REPO/DeepCode"`),而 write_file 的 `file_path` 是**相对 workspace(generate_code)** 的路径(`tools/code_implementation_server.py:399-406`)。用 monkeypatch 的 `py_compile.compile` 实测:
```
'fre/config.py'  -> deepcode_lab/tasks/paper_55339c5d/generate_code/fre/config.py   (对)
'setup.py'       -> /home/deepevol/deepevol/DeepCode/setup.py                        (错:编译了 DeepCode 自己的)
'__init__.py'    -> /home/deepevol/deepevol/DeepCode/__init__.py                     (错)
'deepcode.py'    -> /home/deepevol/deepevol/DeepCode/deepcode.py                     (错)
```
计划里 `setup.py` 极常见;命中时返回 `""`(假"通过"),模型写坏的 `setup.py` 得不到反馈。跨任务目录串扰在 `run_trial.sh` 每轮归档 `paper_*` 的前提下不会发生,但脱离该脚本(多个 `paper_*` 共存)时 glob 按 mtime 取最新文件,同名文件会串。
**建议**:删掉 cwd 候选;直接从 `_RunState`/`code_tracker` 拿 workspace 根拼绝对路径(`set_workspace` 已知),不要 glob。

---

## 🟠 逻辑缺陷 / 副作用

### 🟠-1 fix-① 采纳条件无"超集"保证
`agent_orchestration_engine.py:815-819`:`adopted = revised and completeness>=0.8 and valid`。`validate_plan_text` 只查 5 个 section 存在 + YAML 可解析;`_assess_output_completeness` 只查 section/围栏/末行/长度。一份改坏核心组件、删掉原文件、`chars_after < chars_before` 的修订稿照样采纳,`chars_before/after` 只记录不判断。
**实测取证**(用日志里的原计划 vs 采纳稿做 file_structure 差集):fx1 37→50 文件、fx2 31→39 文件,**均无删除、组件名无变化**,两次都是纯增量——所以这两轮结果没被此缺陷污染,但这是运气不是保证。
另:原计划没有落盘(只在 stdout 日志里,且 stdout 经管道块缓冲,与 stderr 交错),事后审计困难;审计调用的 token 用量没进 `planning_attempts.jsonl`/`usage`,成本被少记一次调用。
**建议**:采纳前校验 `orig.file_structure ⊆ revised.file_structure` 且组件名集合不减;把原稿写成 `initial_plan_pre_audit.txt`;把审计 usage 记进 meta。
其他核对:异常路径 `except Exception` 覆盖 `asyncio.TimeoutError`(3.11 下是 `TimeoutError` 子类),`result` 不会被清空;`coverage_info` 只在 success 分支写 meta,fallback/coerced 分支无该 key——合理,因 fix-① 只在 success 分支运行。

### 🟠-2 fix-② 与既有指令自相矛盾
`memory_agent_concise.py:1705`:同一条消息里先说 `If … "All files implemented!", you MUST reply … Do NOT continue calling tools.`,紧接着 `**PLAN COMPLETENESS (overrides the plan):** … CREATE a new … file with write_file`。当剩余列表为空时 Objective 仍是 "Reply 'All files implemented' to finish"。模型会二选一,行为不稳定。另外新建文件会让 `ProgressTracker.completed_files` 超过 `total_files`(`utils/loop_detector.py:236-240` 不设上限),进度显示 >100%。
**建议**:开关开启时把 IMPORTANT 句改成"仅当计划文件与新增基线文件都完成后再回复",并把 `total_files` 随新增文件递增。

### 🟠-3 URL 黑名单扫描所有字符串参数,包括 `write_file` 的 `content`
`mcp.py:144-163` 对每个 MCP 工具的**全部**字符串实参做子串匹配。`code-implementation` 的 `write_file(file_path, content)` 也是 MCPToolWrapper,若模型在 README/docstring 里引用 `https://github.com/kvfrans/fre`(极常见的"Original implementation:"一行),整次写入被拒并收到 "Accessing it would be cheating" 的文案。日志里目前只见 `mcp_fetch_fetch` 被拦(7 次),尚未真的发生,但 `rice/bare_kimi/README.md` 已经出现过 `chengzelei` 说明模型确实会这么写。
**建议**:只对 `fetch`/`git_clone` 类工具的 URL 形参(`url`/`repo_url`)做匹配。

### 🟠-4 repeat-fetch 计数只在成功返回时累加
`mcp.py:231` 在 `try` 成功后才 `+1`;超时/异常路径提前 `return`,不计数。所以"死 URL 反复超时"这一原始动机场景(fre trial1 的 16×)恰恰**不被拦**,被拦的是成功返回内容的 URL(含分页翻页)。与 🔴-1 #1 合看,该守卫拦错了对象。

### 🟠-5 fix-③ 的重写循环缺少上限
`code_implementation_workflow.py:341`:"Rewrite this file with write_file to fix the error before implementing the next file." 没有次数上限;而 🔴-1 #4 又豁免了 write_file 的连续重复熔断,唯一封底是 800 迭代/墙钟。fx1/fx2 日志中 `SYNTAX CHECK FAILED` 0 次,所以本轮没触发,但风险真实存在。
**建议**:同一文件失败 ≥2 次后改为提示"继续下一个文件,稍后回来修"。

### 🟠-6 `refresh_after_fx.sh` 金丝雀可能读到陈旧判分组
第 30 行调用 `run_grade.sh` 不检查退出码;第 31 行 `G=$(ls -1t runs/ | head -1)` 取最新目录。若 `run_grade.sh` 在创建 run 目录之前失败(Docker、配置校验、`uv run` 报错——`run_grade.sh` 自身 `set -e -o pipefail` 会退出),`G` 就是上一次(可能是老的、合格的)判分组 → 最大无效叶 ≤2 → 金丝雀"通过" → 继续判 rice 批。`mx=-1→999` 只覆盖"目录里一个 grade.json 都没有"的情形。
**建议**:记录调用前的 `runs/` 目录快照,要求 `G` 是新出现的目录;检查 `run_grade.sh` 退出码。

### 🟠-7 `run_trial.sh` 假计划闸在 meta 缺失时静默放行
第 168-177 行:找不到 `planning_result_meta.json` 只打 `⚠️` 继续摆卷。当前代码路径下 meta 一定会写(`write_planning_meta` 在 success/coerced/existing 三条路都调用),所以实际风险低,但闸门语义上应 fail-closed。另注意 `source=="existing"`(断点续跑复用旧计划)也会被判废轮(exit 3)——与 `stage_b_driver.py` 头部注释宣称的"断点续跑省钱"用法冲突。

### 🟠-8 `uv.lock` 手改 `requires-python`,`pyproject.toml` 无对应声明
`uv.lock:3` `>=3.12` → `>=3.11`,`pyproject.toml` 里没有 `requires-python`。venv 实际是 3.11.15,上游按 3.12 写(`codebase_index_workflow.py:225-229` 那处 f-string 就是 3.12-only 语法的证据)。我用 venv 3.11 对全仓 `compileall` 通过(rc=0),所以当前没有第二处语法炸点,但运行时语义差异(如 3.12 的 f-string、`itertools.batched`)不在编译期暴露。
**建议**:要么把 venv 换成 3.12 并回滚 uv.lock 与 f-string 改动,要么在报告里注明"Python 3.11,非上游要求的 3.12"。

### 🟠-9 PaperBench 侧未在任务清单里的改动:`paperbench/judge/simple.py`
- `:115-120` `PB_JUDGE_CONCURRENCY`(默认 100,不变)——无影响。
- `:135-140` `PB_STRUCTURED_PARSER_MODEL`(默认 `gpt-4o-2024-08-06`,不变;`.env` 里设为 `deepseek-ai/DeepSeek-V4-Pro`)——这**改变了裁判的二级结构化解析模型**,属于判分链路的一环。默认值未变,但实验实际跑的是 DeepSeek 解析。任务描述只列了 utils.py/eval.py/rice.txt,漏了这一处;报告需要披露"裁判 = V4-Pro 主判 + V4-Pro 结构化解析"。
其余:`CONTEXT_WINDOW_LENGTHS` 新增两条只影响上下文预算裁剪;`paper_split` Literal 新增 `fre/rice/lite` 只影响 chz 校验;`rice.txt`=`rice`、`fre.txt`=`fre`;`analyze_judge_eval_bias.py` 为离线分析脚本,不碰判分。以上均不触及裁判提示词/评分树/文件选择。✅

---

## 🟡 可改进

- **`run_trial.sh:32` rice 标题关键词 `"rice"` 太弱**:`grep -qi rice` 会匹配 "price"、"matrices"。实测 fre 论文前 4000 字符不含 "rice",当前不误判,但换论文就靠不住。建议用 `"breaking through the training bottlenecks"`。
- **`run_trial.sh:192` / `run_grade.sh:38` 的文件计数把 `.pyc` 算进去**(fx1 提交 78 个文件里 36 个是 `.pyc`),"≥5 文件"/"≥3 文件"闸门被虚增。建议 `find … -type f ! -name '*.pyc' ! -path '*/__pycache__/*'`。
- **fix-③ 用 3.11 解释器做语法检查**,若产物用了 3.12-only 语法(嵌套同引号 f-string)会被误报为错误,诱导模型改写正确代码。建议 `sys.executable` 显式记录版本或用 `ast.parse` 并注明。
- **`core/skills/capabilities.py:185`** 仍用未 sanitize 的 `f"mcp_{value}_"` 前缀,含 `-` 的 server 名在 capability 探测处会失配(与 🔴-1 #2 不一致)。
- **`_FETCH_COUNTS` 是进程级全局**,同进程内跨 agent/跨阶段共享且不清零;当前每轮独立进程故无害,但若以后在同进程跑多篇论文会串。
- **`provider_retry_mode`/`_CHAT_RETRY_DELAYS` 等在 import/类定义时读 env**,必须在 Python 启动前 export(`run_trial.sh` 满足);程序内 `os.environ[...] = ` 无效,注释里应写明。
- **`run_trial.sh` 对所有臂注入非官方值**:stall 7200、墙钟 21600、预筛 32000、persistent 重试、参考 32768/下载 16384、规划限时 600。这些与 🔴-1 一起构成"共享的非默认配置",报告应整表列出。
- **未跟踪 `DeepCode/RICE/experiments/experiment2_refining.py`**(716 行,rice 论文的实验脚本,来自 08-26 pilot):放在工具仓库根目录,`git status` 显示 `??`,容易被误提交;filesystem MCP 的允许目录是 `deepcode_lab`,当前 agent 读不到它,所以不构成跨轮泄露,但应移走。
- **fix-① 的审计调用复用 `enhanced_params.checkpoint_callback`**,会用审计输出覆盖规划 checkpoint 文件;随后 `clear_planning_checkpoint` 清掉,所以无残留,但若审计中途崩溃、checkpoint 里留下的是审计稿而非原稿。

---

## ✅ 已核实无问题

- **语法/静态**:11 个改动 .py 在系统 3.12 与 venv 3.11 下 `py_compile` 全过;全仓 `compileall`(排除 .venv/deepcode_lab/RICE 等)rc=0。`os`/`re`/`glob`/`py_compile` import 齐全(`mcp.py:16-17`、`request_params.py:18`、`base.py:5`,其余文件原本已有 `import os`)。无未定义名字;f-string 内无嵌套同类引号问题(`agent_orchestration_engine.py:1012` 用单引号包裹双引号,合法)。
- **`_RunState` 的 `@dataclass` 装饰器完好**(`code_implementation_workflow.py:136-137`),字段列表与上游一致,`max_wall_seconds: float = _MAX_WALL_SECONDS` 仍在。
- **env 门控点在 env 未设时的取值**(逐一对照上游):`DEEPCODE_LLM_RETRY_MODE`→"standard"✅;`DEEPCODE_CHAT_RETRY_DELAYS`→(1,2,4)✅;`DEEPCODE_PERSISTENT_MAX_DELAY`→60✅;`DEEPCODE_PERSISTENT_IDENTICAL_ERROR_LIMIT`→10✅;`DEEPCODE_PREFILTER_MAX_TOKENS`→2000✅;`DEEPCODE_DOWNLOAD_MAX_TOKENS`→4096✅;`DEEPCODE_PLAN_COVERAGE_CHECK`/`DEEPCODE_ALLOW_PLAN_EXTENSION`/`DEEPCODE_POSTWRITE_COMPILE` 未设时代码路径与上游逐字相同✅;`DEEPCODE_URL_DENYLIST` 未设时 `_denied_urls()` 为空列表,不进入任何拦截分支✅。`DEEPCODE_OPENAI_REQUEST_TIMEOUT_S`、`DEEPCODE_CODE_ANALYZER_TIMEOUT_S` 确为上游自带旋钮(`git grep HEAD` 命中)✅。
- **`LoopDetector` 构造默认**(`max_repeats=5, timeout=600, stall=300, max_errors=10`)本身未改;漂移的是 `CodeImplementationWorkflow` 调用处的实参(见 🔴-1 #15)。
- **`.pyc` 不影响裁判文件选择**:`paperbench/judge/simple.py:267-269,327` 的 `blacklisted_base_dirs` 含 `__pycache__`,按路径 part 过滤;`.pyc` 也不在白名单扩展名内。所有 12 份提交(含已判的 archive)都带 `.pyc`(基线轮也有,来源是模型自己执行/编译),故 fix-③ 的 `py_compile` 写入 `__pycache__` 不会造成判分差异。但见 🟡 文件计数虚增。
- **`run_trial.sh` 假计划闸**:`planning_result_meta.json` 的 `source` 取值全集为 `generated / coerced_from_freeform / existing`(`agent_orchestration_engine.py:844,946,964,1414,1489`),`coerce_text_to_minimal_plan` 路径写 `coerced_from_freeform` → exit 3 ✅。fix-① 采纳后仍写 `source: generated` 并附 `coverage_check` ✅。
- **状态闸**:`completed_with_warnings` 仅在 `impl_status=="warning"` 且非 strict 时产生;loop-detector/墙钟中止的轮次 `impl_status=="incomplete"` → `pipeline_status="incomplete"` → exit 2,不会摆卷 ✅。
- **`run_grade.sh` n_tries**:`N` = `$SUB_ROOT/*/` 目录数,与 `solver.py` 每个 task 实例 `pop()` 一份的机制匹配;`pb_submissions/` 根下非法 paper id 的预检存在 ✅。
- **`stage_b_driver.py`**:无默认论文(缺 `STAGE_B_INPUT` 直接退出)、交接文件按 slug 分文件、兜底只在本轮 `paper_dir` 内 glob、`sys.path.insert(0, cwd)` 依赖 `run_trial.sh` 的 `cd` ✅。
- **`_apply_tool_filter` 与 sanitize 一致**:两处前缀都做了 `.replace("-","_")`,`bare` 名切分正确;`tool_filter={"github-downloader":{"git_clone"}}` 会匹配 `mcp_github_downloader_git_clone` ✅。
- **索引复用的文件名模式** `{repo_name}_index.json`(`tools/code_indexer.py:219`)与复用判断一致 ✅。
- **document_index.json 路径**与分段服务器写入位置一致(`tools/document_segmentation_server.py:1495`)✅。
- **黑名单内容**:fre=`https://github.com/kvfrans/fre`、rice=`https://github.com/chengzelei/RICE`,小写子串匹配;fx 轮 `github_download.txt` 显示 5 个仓库全部成功且不含黑名单仓库 ✅。
- **fix-① 实证**:fx1/fx2 两轮采纳稿的 file_structure 均为原稿严格超集(37→50、31→39),组件名集合无变化(见 🟠-1)。

---

## 一句话结论

**纪律 A 不成立**(15 项非门控/漂移差异,所有臂共享,报告措辞必须改);**纪律 B 基本成立但 fx 臂提示词含 "Graders … forfeit points" 这一评分结构元知识,需删除或披露**;工程师须立刻处理:改报告表述、删/披露 Graders 句、轮换 `refresh_after_fx.sh` 里的明文 key、修 fix-③ 的 cwd 候选路径、给 fix-① 加超集校验、给金丝雀加"新判分组"校验。
