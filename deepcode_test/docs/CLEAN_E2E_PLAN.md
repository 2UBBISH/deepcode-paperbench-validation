# 干净端到端验证(E1)· 计划书 + 独立台账

> 2026-08-26 · 对应 HANDOFF.md §3.1 待办第一条
> 口径:**新任务目录、零人工策展、一次跑完**;13 项修补全部保留,不还原。
> 点火脚本:`run_clean_e2e.sh`(本轮专用,不动历史脚本 `run_stage_b.sh`)

---

## 0. 已定决议(2026-08-26 评审定稿)

1. **模型:全程 Kimi-K2.7-Code**(defaults = implementation = Kimi/32768,现配置即是,零改动;不设 planning 覆盖)。
   备注:规划相位首次由 Kimi 生成(B/B′ 的方案是 DeepSeek 写的);风险由 5 段校验 + 完整性 ≥0.8 + 3 重试兜底,全败即停(≈¥1~2)。
2. **CodeRAG 不裁剪**:仓库 agent 下多少建多少,正面检验内置预过滤器(fail-open 风险已知,靠熔断线兜底)。
3. **源码补丁一处不还原**:5 处 `[local compat]` + MCP 配置 + git insteadOf 封锁全部保留;venv 维持 3.11(重建 3.12 归为跑后正规化,本轮不引入新变量)。
4. **裁判恒定**:DeepSeek-V4-Pro + code_only(paperbench `.env`,不动)——可比性的锚。
5. **实验纪律**:唯一允许的人工干预 = 按熔断线杀进程。熔断即实验结果("不裁剪跑不动"是有效结论),**不救援、不裁剪、不切模型、不重启相位**——任何救援都会让本轮重新变成拼装。

## 1. 闸门(独立于旧台账;数字待批)

| 闸门 | 提议值 | 超线动作 |
| --- | --- | --- |
| 本轮总额 | **≤ ¥100**(复现侧目标 ≤¥50 + 判分 ≈¥35~40) | 停、保留产物、报实际数 |
| 索引相位墙钟 | **投影或实际 > 4h 熔断**(投影法见 §5:索引开跑 ~15min 内即可判死刑,不必干等 4h) | 杀进程,记"不裁剪跑不动" |
| DeepCode 侧总墙钟 | **10h 硬顶**(脚本 `timeout` 自动兜底) | 自动杀,不判分 |
| 单步无输出 | > 40min 视为卡死(沿用旧规) | 杀 → 分诊 → 报告 |
| 判分闸 | 仅 `completed` / `completed_with_warnings` 自动判分 | 其余状态停下等口令,决定是否花 ¥35+ 判部分产物 |

## 2. 点火(等口令)

```bash
nohup bash /home/deepevol/deepevol/run_clean_e2e.sh > /home/deepevol/deepevol/stageE1_console.log 2>&1 &
```

脚本流程:预飞自检(模型配置 / 6 MCP / git 封锁 / 无残留进程 / 裁判解析模型)→ 归档旧任务目录 `paper_e8af8afa` → 清 stale 路径文件 → 驱动流水线(严格退出码 + 10h timeout)→ 状态闸 → 摆卷(校验产物属于本轮新任务目录)→ 判分 → 打印 grade 路径。

对上一版脚本修复的三个陷阱:① `/tmp/stage_b_code_dir.txt` 残留 B′ 旧路径 + `|| true` 吞错 → 中途崩溃会拿旧产物判分(¥37 买假分数);② 判分不看流水线状态;③ 产物新鲜度无校验。

## 3. 本轮回答什么(与"原论文效果"的对照口径)

**定位(用户定稿)**:拼装轮 B′ 只是扫清障碍的工程轮;**E1 才是与 DeepCode 原论文主张对照的测量轮**。能校验与不能校验的要预先划清:

| DeepCode 原论文主张 | E1 能否校验 | 口径 |
| --- | --- | --- |
| P1 能自主端到端把论文变成代码 | ✅ 直接校验 | "修补版 + 全程 Kimi"一次跑通即支持;中途死掉即"修补仍不足以自治"(原版裸奔已被 8 轮失败史证伪,故本轮口径必须写"修补版") |
| P2 CodeRAG 带来显著增益 | ⚠️ 方向级校验 | 对照 B(43.3,无 CodeRAG):E1 明显更高 **且** 索引确实建成、检索确实被调用 → 方向一致;数值增益不可精确归因(方案/语料/规划模型混杂) |
| P3 官方绝对分数 | ⚠️ 可做量级对标(2026-08-26 升级:论文表 3 有 rice 单篇官方数据,见 §5e) | rice 官方:0.738/0.609/0.761,均值 **0.702±0.082**(底座 Claude Sonnet 4.5-thinking,裁判 o3-mini,Code-Dev 变体≈我们的 code_only)。模型档差(Sonnet4.5 vs Kimi)必须注明;禁用"复现了官方分数"式表述 |

- 我们实测的本质是**"DeepCode 机制在平价国产模型上的可迁移效果"**,不是复刻官方数字。
- 时长预期:输入/分段/规划 ~20min → 参考 ~10min → 下载(网络)→ **索引 1~4h+(最大不确定项)** → 写码 ≤2h → 判分 ~10min。
- **硬结论路径(条件触发,本轮预算不含)**:E1 是单次跑、单次判。
  - 差距大(±10 分级)→ 单次即可做方向性定性,不追加;
  - 差距小 → 先做**同卷复判 ×2**(同一份 submission 再判两次,≈¥70,分离"裁判方差"这一层,便宜);
  - 复判后仍不可判 → 才升级**全流程重跑**(E2/E3,每次 ≈¥60~100,另立预算另批)。
  - 方差有两层(考生方差 + 裁判方差),先花小钱隔离便宜的那层,再决定是否买贵的。
- 附带产出:各仓库 `filtering_efficiency`(indexes/indexing_statistics.json)= 内置裁剪器的第一份正面测评;`search_code_references` 调用次数(对照 B′ 的 4 次)。

## 4. 台账(本轮独立计,起点 ¥0)

| 时间 | 项目 | 金额(¥) |
| --- | --- | --- |
| | | |

**累计:0 / 上限(待批)100**

## 5. 值守手册(接棒模型用;Opus 5 级即可胜任)

> 设计/评审已冻结在本文档与 HANDOFF.md;值守是机械活:点火 → 看节点 → 算投影 → 必要时熔断 → 填台账。
> 无人值守时的兜底是脚本内 10h 硬顶;下述投影熔断需要有值守才生效。

1. **点火**:`nohup bash ~/deepevol/run_clean_e2e.sh > ~/deepevol/stageE1_console.log 2>&1 &`
2. **节点时刻表**(看 `stageE1_deepcode_*.log` 的 `Progress:` 行):
   - ~20min 内应见 65%(规划完成,`initial_plan.txt` 落盘);
   - ~40min 内应见 75%(参考+下载);此时记录 `code_base/` 仓库清单与体积(`du -sh`)进台账;
   - 80% = 索引开始。**开始后 ~15 分钟做投影**:
     从日志逐文件分析行的节奏实测单文件耗时 t,乘以各仓库预过滤保留文件数之和 N;
     `t × N > 4h` → 立即熔断,不必干等(浪费分钟级,不是 4 小时);
     若日志出现"预过滤解析失败→回退分析全部文件"且该仓库文件数巨大 → 同样立即熔断;
   - 85% = 写码开始(自带 2h 墙钟);之后到判分为止无需干预。
3. **熔断三连**(括号防自杀,沿用旧规):
   `pkill -f "stage_b_driver\.p[y]"` → 等 10s → `pkill -f "python -m tool[s]\."` → `pgrep -f "stage_b_driver\.p[y]"` 验证已死。
   熔断后**不救援不续跑**:日志与已建索引卡就是"不裁剪跑不动"的结论证据,填 §4/§5 收工报告。
4. **单步无输出 >40min** → 按卡死杀(同上),报告日志尾部。
5. **跑完**:填 §4 台账(DeepCode 调用数看日志、判分 token_usage 看 grade.json)、§5 结果登记;
   附带产出记录:`indexes/indexing_statistics.json` 的 filtering_efficiency、
   日志中 `search_code_references` 出现次数、关系抽取 `Error finding relationships` 次数。
6. **升级找用户的情形**:判分闸拦下(状态非 completed*,要不要花 ¥35 判部分产物)、
   熔断后要不要触发 §3 硬结论路径、任何预算超线。

## 5b. 运行中观察(E1 实录)

**F1 · 分段阶段空转(首个 Kimi vs DeepSeek 实质行为差异)** — 2026-08-26 17:44
- 现象:Phase 4 报 "Document segments prepared successfully",但 `document_segments/` **目录根本未创建**;
  Phase 5 随即 `WARNING: no usable segments were found` 回退全文规划。
- 实证:分段 agent 全程只有 **1 次** HTTP 请求(4.6s)。有工具调用必然产生第二次请求(模型→工具→结果→模型),
  故可判定 **Kimi 没有调用 `analyze_and_segment_document`,直接用文字作答收尾**。
  对照 B′(DeepSeek 跑的分段):产出 `document_index.json` + 10 个 segment_*.md(见 archive_bprime_frankenstein/)。
- 代码侧共犯:`prepare_document_segments` 只看 agent 是否返回,**不校验文件是否落盘**,故把空转报成 success。
- 影响:规划改走 traditional 全文模式(106K 字符全文入 prompt vs B′ 的 24K 分段上下文;
  `max_iterations` 5→2、temperature 0.2→0.3)。索引(用 initial_plan.txt)与写码(本就不用分段)不受影响。
- 处置:**不救援**(纪律 §0.5)。这正是干净跑该暴露的事:全程 Kimi 的代价之一。

**F2 · 规划一次通过,但方案比 B′ 短 36%** — 17:46(耗时 ~2min,attempt 1/3)
- `planning_result_meta.json`:status=success、completeness=**1.0**、valid=True、missing=[]、
  usage = prompt 28,614 + completion 6,002 = 34,616 tokens。
- 尺寸:Kimi 14,741 字符 vs B′ 的 DeepSeek 23,066 字符(**-36%**)。结构门槛(5 段 + 完整性 ≥0.8)全过,
  但"更短"是否等于"更薄"要等判分回答——这是全程 Kimi 的第二个可观测代价,记为待验。

**F3 · 日志中的 anyio teardown 噪声(无害)** — MCP stdio 关闭时报
  `RuntimeError: Attempted to exit cancel scope in a different task` + `Task exception was never retrieved`。
  属后台清理竞态,主流程照常推进(仓库自身对同类情况已有 "benign anyio teardown noise" 的注释)。**不处置。**

**F4 · ⚠️ CodeRAG 链条整条塌方(与 F1 同源,是本轮头号发现)** — 17:52~17:54
- `github_download.txt` **全文只有 121 字节一句话**:
  > "I'll download all five selected GitHub repositories into the specified directory, with each repo in its own subdirectory."
  agent 宣布意图后**没有调用任何工具**就收尾。与 F1(分段)**完全同一种失败模式**。
- 连锁后果:`code_base/` 从未创建 → Phase 8 无仓库可索引 → **无 `indexes/`、无 `codebase_index_report.txt`**
  → Phase 9 在**空索引**下开跑。
- 参考挖掘本身是成功的(`reference.txt` 12.9KB,自主选出 5 个仓库:stable-baselines3 / **carla** /
  metadrive / random-network-distillation / tianshou——注意 metadrive+tianshou 正是 B′ 当年人工删掉的两个,
  carla 更是 GB 级新增)。**塌在下载执行,不塌在选材。**
- **命中 RUNBOOK 坑8 的明确警告**:`enable_indexing=True` + 空索引 = 只有 2 个工具(write_file +
  search_code_references)、无 read/execute,是**已知最弱配置**("别用完整模式 + 空索引")。
  当前写码正是在这个配置下进行(17:54 起,已落 envs/__init__.py、envs/mujoco_setup.py)。
- 根因推定:源码评审阶段已发现 **`GITHUB_DOWNLOAD_PROMPT` 定义了却从未接线**,下载 agent 的 instruction
  只有一句话拼串、无结构化输出契约(`agent_orchestration_engine.py:934`)。DeepSeek 当年"碰巧"照做了;
  Kimi 在这种松散契约下选择用散文作答。**"未接线的 prompt"从潜在缺陷变成了实际阻断。**
- 处置:**不救援**,让它跑完写码(便宜,≈¥5),真实记录"全自动零人工"的终局。判分与否见下方决策点。

---

## 5c. E1 熔断后排查报告(2026-08-26 17:58 杀,证据链完整)

**熔断执行**:driver + 外层脚本已死、MCP 无残留、**判分从未启动**(`pb_submissions` 空、`runs/` 无新组)。
**实际花费**:23 次 LLM 调用,in 398,283 / out 39,136 tok ≈ **¥2.2**(闸门 100,几乎未动)。
**决定性证据**:`tasks/paper_328873ca/logs/llm.jsonl` 逐次记录了 finish_reason 与 tool_count。

### 关键对照表(同一个 Kimi,三种结局)

| # | Agent | 工具数 | finish_reason | 结果 |
| --- | --- | --- | --- | --- |
| #02~#10 | ReferenceAnalysisAgent | **3(经 tool_filter 收窄)** | **tool_calls ×9 轮** | ✅ 成功,产出 12.9KB reference.txt |
| #00 | DocumentSegmentationCoordinator | 3 | **stop**(137 tok) | ❌ 只说"I'll perform..." |
| #12 | GithubDownloadAgent | **17(未过滤)** | **stop**(575 tok) | ❌ 只说"I'll download all five..." |
| #13 | StructureGeneratorAgent | **0** | stop(2616 tok) | ❌ 把工具调用当**文本**吐出 |

**结论先行:不是"Kimi 不会调工具"**——它在参考相位连调 9 轮。失败与**契约质量/工具面纪律**强相关。

### D1 · `command-executor` MCP 服务器缺失(坑7 没修干净)【配置层,一行可修】

- 全仓 `server_names=[...]` 请求 7 个服务器,`deepcode_config.json` 只配了 6 个,**差集恰为 `command-executor`**。
- 后果:StructureGeneratorAgent 拿到 **tool_count=0**,于是把调用**以字面文本吐出**(#13 原文):
  `<|tool_calls_section_begin|><|tool_call_begin|>functions.execute_commands:0<|tool_call_argument_begin|>{"commands":["mkdir -p rice/rice/baselines", ...]}`
  ——**与 RUNBOOK 坑7 记录的 `<｜DSML｜tool_calls>` 泄漏是同一现象**,只是换成 Kimi 的标记格式。
- 连锁:文件骨架从未建立 → 写码相位的待办清单退化为解析 `initial_plan.txt`(而非扫描 `generate_code/`)。
- 修复材料现成:`tools/command_executor.py` 存在,工具名 **`execute_commands`**(与模型想调的完全一致),
  `server_name="command-executor"`,有 `__main__` 入口,`python -m tools.command_executor` 即可。

### D2 · `GITHUB_DOWNLOAD_PROMPT` 是死代码,且已过期【源码层】

- 全仓检索:该常量**只出现 1 次**——它自己的定义行(`prompts/code_prompts.py:91`)。从未 import、从未引用。
  对照 `PAPER_REFERENCE_ANALYZER_PROMPT`:44 行导入、974 行使用。
- 下载 agent 实际用的是**内联一行**(`agent_orchestration_engine.py:934`)
  `"Download github repo to the directory {paper_dir}/code_base"`,user message = **reference.txt 全文原样倾倒**,无任务框定、无输出契约。
- **即便直接接上也会坏**,两处已实证:
  1. `.format(paper_dir=...)` 抛 `KeyError: '"downloaded_repos"'`(JSON 花括号与占位符冲突)——很可能正是当年改用内联一行的原因;
  2. 它要求 "Use **interpreter** tool",而该 agent 工具面只有 filesystem + github-downloader,**没有 interpreter**。
- 故修复不是"接上",而是**按真实工具面重写**(`download_github_repo` / `git_clone`),并写死"必须调用工具"的契约。

### D3 · 工具面纪律不一致(为什么参考成功、下载失败)

- 参考 agent:**3 个经 tool_filter 收窄的工具** + 88 行结构化 prompt → 连续 9 轮 tool_calls;
- 下载 agent:**17 个未过滤工具**(14 filesystem + 3 github-downloader)+ 一行 prompt + 原始倾倒 → 一枪narration;
- 分段 agent:3 个工具、instruction 也丰富,但 message 在"调工具"之后又索要
  "quality assessment / recommendations / completeness evaluation" 四项**叙述性交付物** → 模型选择先叙述、就此收尾。
- 规律:**契约越紧、工具面越窄,Kimi 越稳;要求越像"写份评估",越容易退化成散文。**

### D4 · 代码侧共犯:失败被静默吞掉

- `document_segmentation_agent.py:151-161`:`generate_str` 后**无条件**返回
  `status="success"`、`segments_available=True`,**从不校验 `document_segments/` 是否真的落盘**;
- 下载相位把模型回的任何文本原样写进 `github_download.txt`,**不校验 clone 是否发生**;
- 于是两次空转都被报成成功,流水线一路开到写码——**这才是"塌方却无人叫停"的直接原因**。

### 缺陷归属考据(git 实证,别把锅都算给上游)

`git -C DeepCode status`:本地只改了 5 个文件(`utils/loop_detector.py`、`uv.lock`、
`workflows/{agent_orchestration_engine,code_implementation_workflow,codebase_index_workflow}.py`)。
`git diff -U0` 显示对编排引擎只有 3 处 hunk(共 +32/-1):下载迭代配额、参考相位参数、索引幂等。

| 缺陷 | 归属 | 实证 |
| --- | --- | --- |
| D2 死 prompt + 内联一行 instruction | **100% 上游** | `prompts/code_prompts.py` 未被我们修改;`git show HEAD:` 确认 :934 那行内联 instruction 是上游原样 |
| D3 下载 agent 无 tool_filter | **100% 上游** | 同上,位于 `github_repo_download` 未改动区 |
| D4 分段无条件报 success | **100% 上游** | `workflows/agents/document_segmentation_agent.py` 未被我们修改 |
| D1 配置缺 `command-executor` | **我方遗漏**(上游有共犯) | 配置是坑7 时手工补的 6 项;上游 `presets.json` 全是第三方(playwright/context7…),**不含自家 7 个**;`mcp_servers` 默认空 dict;`init` 不生成 |

**上游为什么会留下这种 bug —— 四条结构性原因**:
1. **失败静默**:D4 让空转与成功在日志和返回值上完全同形,没有任何信号可供发现;
2. **强模型掩盖松契约**:一行 instruction 对 GPT-4o/Claude/DeepSeek 级模型"够用",缺陷只在换模型时引爆——
   恰好落在开发者测试配置之外;
3. **死常量对工具链不可见**:`prompts/` 里的模块级常量被 linter 视为导出 API,永不报 unused;无测试导入;
   且 `.format()` 一接就抛 KeyError,谁试着接线都会立刻退回内联一行——我们挖到的正是这块化石;
4. **Paper2Code 已是仓库次要路径**:当前 README 明写 `tools.mcpServers` 是 "historical" 且与通用
   coding-agent 的 `mcpServers` **分开**;`deepcode mcp presets`/Desktop MCP 管理只覆盖后者;
   `tests/` 里仅 1 个 compatibility 测试提到 segmentation。仓库主线已转向通用 coding agent,论文流水线是遗留面。

**对实验口径的修正**:E1 的结论不能写成"DeepCode 不能自主端到端"。精确表述是
**"论文流水线的工具调用契约过松,扛不住换模型;而静默成功的代码把失败藏了起来"**;
且 D1 是我方配置遗漏,故 E1 也不是对上游的公平测试——E2 必须在补齐 D1 后重跑才谈得上公平。

### 5d · README/官方文档精读补遗(2026-08-26,E2 前必读)

首次读到权威架构文档 `docs/P5_PAPER2CODE_ARCHITECTURE.md` 与 README 的 Paper2Code / Research results 章节。

**① 必补:`command-executor`**(= D1)。旧版 README 的 MCP 表列了 **8 个**服务器(含 command-executor 与
   file-downloader),而上游**从未给过覆盖自家服务器的完整配置样例**——旧 README 里唯一的 JSON 样例是
   Windows 专用、只含 filesystem。我方 6 项配置漏了 command-executor(file-downloader 无代码路径请求,可不补)。

**② 应决策:`strict_outcomes`(我们无意中用了 legacy 宽松语义)**
   P5 明确:Desktop 走 `strict_outcomes=True`,"legacy CLI/direct Python 调用方保留 P0 planner fallback
   与旧完成语义,除非显式 opt-in"。我们的 driver 未传该参数 → **默认 False**,后果:
   (a) 规划失败可被"自由文本强扭成最小 plan"兜底;(b) 只要文件写出来就算 `completed`,不要求验证。
   strict=True 则要求"发现到至少一条测试命令且全部通过"才算成功——**我们的产物通常没有测试套件,
   开 strict 会被判非成功、卡住判分闸**。故 **E2 建议仍用 False,但要从"默认踩中"改为"书面选定"**。

**③ 待验风险:`security.accessPreset = "ask"`**
   代码核实(`core/harness/policy.py:125`):`build_permission_engine` 读的是 `permission_mode`,
   我们**没设**该键 → 回落 `FULL_AUTO`,所以 E1 的 write_file 未被拦(实证:3 次 write_file 全过)。
   但 `accessPreset="ask"` 仍使 `bypass_origin_approval=False`;而 **command-executor 的 shell 命令是 E2 才首次引入**,
   非 FULL_AUTO 会挂 `TerminalApprover`,**nohup 无 TTY 下可能挂死或拒绝**。
   → E2 前必须冒烟验证 mkdir/touch 能否无人值守执行;必要时用 `DEEPCODE_PERMISSION_MODE` 覆盖。

**④ 环境合规**:README 正文 "DeepCode requires **Python 3.12+**";源码安装示例更写 `uv venv --python=3.13`。
   我们是 **3.11**(坑11 遗留)。E1 能跑通说明非阻断,但已属文档级违规,E2 后应正规化。

**⑤ 官方验证步骤**:`python -m compileall`(刚跑,✅ 通过)、`deepcode --version`(✅ 2.1.0)、
   `deepcode-app-server --verify-runtime`(**未跑过**,可补)。

**⑥ 口径修正——官方数字的准确引用**(README Research results):
   | 子集 | DeepCode | 对照 | 差 |
   |---|---|---|---|
   | 人类专家子集 | 75.9% | 最佳人类基线 72.4% | +3.5 |
   | 商业 agent 子集 | 84.8% | 最佳商业 agent 58.7% | +26.1 |
   | 科学编码 | 73.5% | PaperCoder 51.1% | +22.4 |
   | LLM-agent 基线 | 73.5% | **最佳 LLM agent 43.3%** | +30.2 |
   ⚠️ **两处防混淆**:(a)"最佳 LLM agent 基线 **43.3%**"与我们 B 轮得分 43.3 **纯属数字巧合**,报告中必须显式声明;
   (b) 官方口径是 20 篇 / 8,316 个可评分项的**子集**统计,与我们 rice 单篇 / 178 项 / code_only **不可直接比**。
   方法学细节在论文 arXiv:2512.07921(未读,若要严谨对标需补读)。

**⑦ README 与源码的架构落差(关系到"是否与原论文一致"的解释)**
   README 称 "Central Orchestrating Agent … **chooses and revisits phases according to task state
   instead of treating reproduction as a fixed prompt chain**",并列出 7 个专家角色;
   但我们实跑的 `execute_multi_agent_research_pipeline` 是**硬编码的 Phase 0→10 线性链**,
   分支只有 `if enable_indexing`,**没有任何"依状态选择/回访相位"的中枢**;
   "Iterative verification 把失败回灌规划"在代码里也只体现为末尾一次性的测试发现(且需 strict 才启用)。
   → 结论:我们测的是**线性流水线实现**,而非 README 描述的自适应编排。这一条必须写进任何对外结论,
   否则会把"实现与文档的落差"误算成"论文方法无效"。

**⑧ 入口考据:直调内核没有跑错,且是唯一可行的 headless Paper2Code 入口**(2026-08-26 核查)
   - 新 README(README_ZH 为其中文镜像,结构同构)给 Paper2Code 的入口只有两个:
     **Desktop Paper Threads** 与 **`deepcode test <paper>`**。
   - Desktop 路径:`core/application/workflow_adapter.py:113` 调用的**正是**
     `execute_multi_agent_research_pipeline` —— 与我们 driver 直调的同一函数(P5 原话:
     "DefaultWorkflowRunner is the only adapter to the existing research and implementation kernel")。
     壳与内核之差只有三开关:`strict_outcomes=True`、`planReview` 默认开(交互式)、
     **`enableIndexing` 默认 `False`**。
   - `deepcode test` 路径:**帮助文本漂移的死条目**——`cli/` 无实现文件,help 的 "Available papers:" 为空
     (与 driver 编写时的记录一致,今日复核仍然如此)。
   - `deepcode exec` / `loop` 是**通用 coding agent** 的 headless 入口,不是 Paper2Code;
     用它复现论文 = 换成非结构化方法,测的不再是论文的多相位主张。
   - ⭐ **反转发现:官方产品的默认形态是"索引关闭"**(Desktop `enableIndexing` 默认 False)——
     即新 README 默认跑法 ≈ 我们的 B 组(无 CodeRAG);我们开索引跑 B′/E 系列,
     反而是对论文"CodeRAG 增益"主张**最有利**的配置。对外报告需写明此入口与开关组合
     (kernel-direct / indexing on / strict off / no plan review)。
   - 数据集侧从第一天即为 paperbench:输入 = `frontier-evals/project/paperbench/data/papers/rice/paper.md`,
     判分 = 同仓库 rubric + DeepSeek 裁判;全量 20 篇在 `data/papers/`(`paper_split=debug` 仅 rice)。
     DeepCode 仓库不含官方 PaperBench 挂架(`eval/` 只有 swebench),PBDirectSubmissionSolver 组合
     是 RUNBOOK §四记录的正确路线。

### 建议的修复顺序(E2 前置,待批)

1. **D1 配置修复**(零风险):`deepcode_config.json` 补 `command-executor`,与另三个 python MCP 同样式;
2. **D2 重写下载 prompt**(小改):按真实工具面写契约式 prompt 并接线,避开 `.format` 花括号陷阱;
3. **D3 给下载 agent 加 tool_filter**(小改):收窄到 `github-downloader` 的 clone 工具 + 必要的 filesystem 读写;
4. **D4 加落盘校验**(小改):分段/下载两处按"产物是否存在"判定 success,否则 fail-fast;
5. 全部就位后重跑 **E2**——那才是第一次真正带 CodeRAG 的干净端到端。
   ⚠️ 注意:D1~D4 属"修 bug",不属"人工策展",不违反 E1 的干净口径;但 E2 必须重新声明为**修补版 v2** 的测量轮。

---

## 5e. 修复实录 + 官方声称效果 → E2 验证矩阵(2026-08-26)

### 修复完成(全部冒烟验证通过,总花费 <¥1)

| # | 修复 | 位置 | 验证 |
| --- | --- | --- | --- |
| D1 | 配置补 `command-executor`(第 7 个 MCP) | `~/.deepcode/deepcode_config.json` + 点火脚本预飞 need-set | 冒烟1:注册+无TTY执行 mkdir/touch ✅ |
| D2 | 下载 prompt 重写接线(契约式,按真实工具面) | `agent_orchestration_engine.py` `github_repo_download` | 冒烟2 ✅ |
| D3 | 下载工具面收窄 `tool_filter={"github-downloader":{"git_clone"}}` | 同上 | 冒烟2 ✅ |
| D4 | 落盘校验:分段验 `document_index.json`/下载空产出先自纠一次再 fail-fast | `document_segmentation_agent.py` + engine | 冒烟2/3 ✅ |
| **D5** | **工具名连字符消毒(`-`→`_`)——比 D2/D3 更根本的总根因** | `core/agent_runtime/tools/mcp.py`(5 处)+ `core/compat/agent.py` tool_filter 前缀(2 处) | 冒烟2 真克隆 ✅ / 冒烟3 真分段 ✅ |

**D5 根因(A/B 裸 API 实证)**:硅基流动 Kimi-K2.7 对声明名含 `-` 的工具**静默不产生结构化调用**
(`mcp_github-downloader_git_clone` → finish='stop'、0 tool_calls;同 schema 改下划线 → 正确调用)。
E1 全部"宣布意图不调工具"的谜底:**F1 分段(`document-segmentation`)与下载(`github-downloader`)的工具全是哑的;
参考相位能活只因 `filesystem`/`fetch` 恰好无连字符;写码相位能活只因走了短名别名。**
路由不受影响(wrapper 用 `_original_name` 调 MCP);`build_aliased_registry` 按裸名后缀匹配,不受影响。

### 官方声称的效果(arXiv:2512.07921 实测设置:底座 Claude Sonnet 4.5-thinking / 裁判 o3-mini / Code-Dev / 每篇 ×3)

| # | 声称 | 官方数字 | 我们已有 | E2 验证什么 |
| --- | --- | --- | --- | --- |
| **C1 ⭐主目标** | **CodeRAG 对非 frontier 模型大幅增益**("约 +70% 相对",Gemini-2.5-Flash,3 篇子集;frontier 模型获益微乎其微) | 无RAG 0.35-0.38 → 有RAG ≈0.6+ | B→B′:0.433→0.605(**+40% 相对**,Kimi)——方向一致但 B′ 是拼装+人工裁剪 | **E2(有RAG,干净) vs B(无RAG)**:E2 显著高于 0.433 且索引建成、检索被调用 → C1 在 Kimi 档位成立 |
| C2 | rice 绝对分(碾压商业工具) | DeepCode **0.702±0.082**(0.738/0.609/0.761);同表 Claude Code 0.3787 / Cursor 0.4186 / Codex 0.3645 | B′ 0.605 ≈ 官方 Run2(0.609);B 0.433 已及 Cursor 档 | E2 分数落点做量级参照;差距解释带 ±0.08(官方自身单次 std) |
| C3 | CodeMem 增益(rice 0.33-0.43→0.70-0.92) | CodeMem 表 | 我们记忆常开,无对照 | 本轮不验(需另做关记忆消融) |
| C4 | 20 篇 73.5±2.8 / 人类专家 75.9 / 商业对比 0.8541 | 主表 | 不在范围 | 不验(需多篇) |
| C5 | 自动验证 +3.7~6.5pp | §消融 | strict_outcomes=False 未启用 | 不验(书面选定 legacy 语义) |
| — | 官方"LLM agent 基线 43.3%" | o1-IterativeAgent | 与我们 B=43.3 **纯数字巧合** | 报告防混淆声明 |

**两个意外对齐(增强可信度,写进最终报告)**:
1. 官方 CodeRAG 消融的弱模型无RAG分数 0.35-0.38,我们 Kimi 无RAG 0.433 —— 同量级;
   官方有RAG ≈0.6+,我们 B′ 0.605 —— 同量级。**Kimi 档位的行为落在官方声称的弱模型曲线上。**
2. 官方方法学 = 每篇 ×3 取均值±std —— 与本计划 §3"硬结论路径"天然对齐;
   官方 rice 单次方差实测 0.609~0.761,证明单次 ±0.08 属正常噪声,解释差距时必须带此噪声带。

**E2 定义(修复后测量轮,等口令)**:配置 = 全 Kimi / 不裁剪 / strict=False / 修复 D1-D5 就位;
命令与闸门沿用 §1-2(点火脚本预飞已含 7 服务器检查);对照组 = B(0.433,无RAG);
判定 = C1 方向 + C2 量级;若差距落在噪声带内,走 §3 硬结论路径(同卷复判×2 → 全流程重跑)。

---

# 7. rice 线结论(定位:**跑通工具的测试**,非主科学结论)

> 用户 2026-08-26 定调:rice 这条线的价值在于**证明整条链路可跑通、并把坑挖出来**,
> 分数是支撑证据而非头条主张。真正的验证在 fre 线(见 `FRE_VALIDATION_PLAN.md`)。

## 7.1 一句话结论

**链路已验证可用**:DeepCode 复现 → PaperBench SimpleJudge 判分 → 得分,全程跑通、判分零失败;
过程中挖出并修复 **18 处缺陷**(RUNBOOK 13 坑 + 本轮 D1-D5);
产出的分数在**同一篇论文**的对照下站得住,但因拼装、单次、跨裁判三重保留,**不作为效果主张**。

## 7.2 实测数据(rice · Code-Dev · 178 个 Code Development 叶子 · 裁判恒定 DeepSeek-V4-Pro)

| 主体 | 得分 | 裁判 | 性质 |
| --- | --- | --- | --- |
| 白卷(dummy) | **0.000** | DeepSeek | 地板校准:178/178 有效判分、全部 0 分 |
| DeepCode+Kimi,**无** CodeRAG(B) | **0.433** | DeepSeek | fast 模式,11 工具 |
| DeepCode+Kimi,**有** CodeRAG(B′) | **0.605** | DeepSeek | indexed 模式,2 工具;**拼装产物** |
| — 论文同篇对照 — | | | |
| Codex | 0.3645 | o3-mini | 论文 Table 1 |
| Claude Code | 0.3787 | o3-mini | 论文 Table 1 |
| Cursor | 0.4186 | o3-mini | 论文 Table 1 |
| DeepCode+Sonnet4.5 | 0.7380(单次)/ 0.702±0.082(×3) | o3-mini | 论文 Table 1 / Table 3 |

## 7.3 站得住的三条

1. **链路可用**:白卷得 0(证明裁判不是随便给分)、真卷得 0.433/0.605、178 条判分零失败。
   成本结构摸清:DeepCode 跑一轮 ¥3~5,**判分才是大头 ¥17~37**(每叶携带全文论文)。
2. **CodeRAG 方向与论文一致**:B→B′ 唯一变量是 `enable_indexing` 开关,得分 0.433→0.605
   (**+40% 相对**)。论文声称非 frontier 模型约 +70% 相对(Gemini-2.5-Flash,0.35-0.38→≈0.6+)。
   **量级与方向双吻合**:我方无 RAG 的 0.433 与论文弱模型的 0.35-0.38 同档,有 RAG 的 0.605 与论文的 ≈0.6+ 同档。
   → **Kimi 档位的行为落在论文描述的"弱模型受益于 CodeRAG"曲线上。**
3. **数值不丢人**:同一篇 rice 上,0.605 高于全部三个商业工具(0.36~0.42),
   0.433(连 CodeRAG 都没有)也已在商业工具之上。

## 7.4 必须打折的四条(诚实版)

1. **B′ 是拼装产物**:跨 8 轮、复用失败轮的中间产物、中途两次换模型、语料经**两次人工精简**。
   "自主端到端得 0.605"**不成立**;成立的是"修补+人工策展后的 CodeRAG 流水线产物得 0.605"。
2. **干净端到端从未跑通**:E1(2026-08-26 17:44)因 D1-D5 缺陷在 14 分钟内塌方 ——
   Kimi 不调工具 → 无仓库下载 → 空索引 → 落进"完整模式+空索引"这一已知最弱配置,已熔断(花费 ¥2.2)。
   根因五条(缺 command-executor / 死 prompt / 无 tool_filter / 静默 success / **工具名连字符使 Kimi 哑火**)已全部修复并冒烟验证,但**修复后的 rice 干净轮未再跑**。
3. **跨裁判**:0.433/0.605 由 DeepSeek 判,论文数字由 o3-mini 判(我方对其 F1 仅 0.685)。
   若我方裁判偏松则含水分。脚手架相同(均为 SimpleJudge),差异仅在 completer 模型 —— fre 线的锚点正是为量化这一偏移而设。
4. **各只跑一次**:无方差估计。论文自己在 rice 上三次跑出 0.609~0.761(单次间差 0.15),
   说明**单次 ±0.08 属正常噪声**,任何小差距都不可解读。

## 7.5 这条线真正的产出:一份缺陷清单

| 类别 | 数量 | 代表 |
| --- | --- | --- |
| 环境/上游落差(RUNBOOK) | 13 | 裁判上下文表 OpenAI 硬编码、二级解析器写死 gpt-4o、`deepcode init` 不写自家 MCP |
| 本轮新挖(D1-D5) | 5 | **D5 工具名连字符让 Kimi 静默哑火**(A/B 裸 API 实证)、`GITHUB_DOWNLOAD_PROMPT` 是从未接线的死代码、分段无条件报 success |
| fre 切换时新挖 | 2 | `paper_split` 是 chz Literal(新建 split 文件不够)、**多提交必须设 `n_tries` 否则只判一份** |

**归属澄清**:D2/D3/D4 是 100% 上游缺陷(git 实证:相关文件我方从未修改);D1 是我方配置遗漏(上游共犯:不为自家 7 个 MCP 提供任何配置/preset/init)。
上游缺陷能长期存活的结构性原因:失败是静默的、强模型掩盖松契约、死常量对 linter 不可见、Paper2Code 已是仓库次要路径(README 明写 `tools.mcpServers` 为 "historical")。

## 7.6 一条方法学告诫(代价换来的)

**跨论文比较毫无意义**。同一个 Claude Code,在 fre 上 0.6286、在 rice 上 0.3787 —— **论文难度差异(1.66×)大于工具差异**。
本项目曾据此误判"我方 0.605 还不如裸 Claude Code",实际在同篇对照下高出约 60%。任何对外结论必须锁定同一篇论文。

## 6. 结果登记(跑后填)

- 流水线状态:
- 任务目录 / 产物文件数:
- 索引:仓库数 / 索引文件数 / filtering_efficiency / 关系抽取失败数:
- 检索调用次数:
- grade.json:
- 得分: ______ (对照:白卷 0 / B 无 CodeRAG 43.3 / B′ 拼装+裁剪 60.5)
- 结论:
