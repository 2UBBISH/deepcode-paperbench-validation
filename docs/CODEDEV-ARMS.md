# PaperBench Code-Dev 对比：口径、题面、各臂怎么接（2026-09-19）

一页说清"我们在比什么、拿什么给 agent、裁判看什么、每一臂怎么接进来"。所有引用都指向**本仓库里**的
`frontier-evals/`（openai/frontier-evals 按 `patches/UPSTREAM_BASE.txt` 的 commit vendored 进来，见 §5），
文件名后的行号是那个版本的行号。DeepCode 论文 = arXiv 2512.07921。

## 1. 比什么：PaperBench Code-Dev

| 事实 | 依据 |
| --- | --- |
| DeepCode 论文用的是 **PaperBench Code-Dev** | 论文 §4.1 Datasets："we employ PaperBench Code-Dev"；§4.1 Grading Methodology："follows the PaperBench Code-Dev protocol, which … **does not include post-submission reproduction**" |
| Code-Dev = 只判 Code Development 叶，**不执行代码**，开关 `paperbench.judge.code_only=True` | `frontier-evals/project/paperbench/README.md` §"PaperBench Code-Dev" |
| 论文的对照系统：Cursor 1.7.52、Claude Code 2.0.22（Sonnet 4.5-thinking）、Codex codex-cli 0.47.0（GPT-5 Codex-high，auto approval）；**对它们用了什么提示词一字未提** | 论文 §4.1 Baselines (3) |
| 论文 Table 1 bam 列：DeepCode 0.853 / Claude Code 0.383 / Cursor 0.378 / Codex 0.194（o3-mini 裁判） | 论文 Table 1 |
| 我们唯一有效的对照（09-15，bam，同底座 V4-Pro，同裁判）：DeepCode 0.837 / Codex 裸跑 0.734 | `RESULTS-HISTORY.md` §1 |

结论：**DeepCode 的绝对分复现了，论文的 4.4× 相对优势没复现**（Codex 在同底座 + 官方题面下是 0.73 不是 0.19）。

## 2. 题面（给 agent 的）和答案卡（给裁判的）是两样东西

```
eval.py   PaperBench(PythonCodingEval).get_instances()   列题；每道题 prompt = 题面文件；judge = 裁判配置
task.py   PBTask.setup()                                  把题面写成 /home/instructions.txt，和论文四件一起上传进 agent 容器
judge/    SimpleJudge(rubric, submission)                 裁判：rubric.json 逐叶打分；code_only 时只判 Code Development 叶
```

| | 文件 | 谁拿到 | 依据 |
| --- | --- | --- | --- |
| **题面** | `frontier-evals/project/paperbench/paperbench/instructions/code_only_instructions.txt` | 做题的 agent（写进它容器的 `/home/instructions.txt`） | `paperbench/nano/eval.py:127-130`（`code_only` → 这份；否则 `instructions.txt`，多出 `reproduce.sh` / 7 天 / A10 那段）；`paperbench/nano/task.py:95-108`（上传） |
| **题面附注** | `paperbench/solvers/basicagent/prompts/templates.py:45-54` `additional_notes_template`（Compute / Total Runtime / API keys / root / 用满时间 / 要真做） | 官方参考 agent（BasicAgent / IterativeAgent，论文 [7] 的全部官方数字都是这样喂的） | `paperbench/solvers/basicagent/utils.py:183-204`：读同一份题面后 `instructions += additional_notes_template.format(...)`；无 GPU 填 `no_gpu_template`，无时限填 `no_time_limit_template`，有时限填 `time_limit_template`（小时数） |
| **续跑语** | 同文件第 34 行 `DEFAULT_CONTINUE_MESSAGE` | agent 停下时 harness 回的话 | — |
| **答案卡** | `data/papers/<id>/rubric.json`（+ `judge.addendum.md`） | 只给裁判 | `paperbench/judge/base.py:22-37`（`Judge.__init__(rubric: TaskNode, …)`）；README §Dataset |

PaperBench 的 README 没有一节讲题面（它假定你把 agent 作为 solver 插进 nanoeval，题面自动装进 task）；唯一提到的地方是
README 第 90 行 "the agent is informed of this file in our default instructions for BasicAgent"。所以题面的依据只能是代码。

## 3. 各臂拿什么、怎么接

裁判只看提交目录，**怎么把提交送进去不影响 Code-Dev 分**；差别只在输入和 harness 条件。

**09-20 起的正式三臂是 Codex 桌面版、Claude 桌面版、DeepEvol Paper2Code 线**，接法在分支 `0919-test`（`README.md` 白话版、`desktop/desktop_prep.sh` / `desktop_finish.sh` / `audit_desktop.py`）。三臂共同条件：`deepseek-flash` @ api.deepseek.com、思考开、**不执行代码**（Codex：`~/.codex/rules/paperbench-no-exec.rules`；Claude：工作目录 `.claude/settings.json` 禁 Bash；本线：无执行工具 + `PAPER2CODE_IMPLEMENT_VERIFY` 关）、上下文 1M、同字节输入、3 小时官方时限句。下表的 Codex CLI / Claude Code 两行是 09-19 的 CLI 方案（带关思考代理），**只作参考**；DeepCode 基线行不在本批。

| 臂 | 输入 | 接法 | harness 条件 | 记录 |
| --- | --- | --- | --- | --- |
| **Codex CLI** | 题面 + 官方附注（`deepcode_test/bare/render_prompt.sh` 生成的 `PROMPT.txt`）+ `paper.md` / `addendum.md` / `blacklist.txt`（**不给 PDF 和 assets**，与 DeepCode 臂同一份字节；owner 09-19） | `deepcode_test/bare/run_bare.sh codex <paper>`：`codex exec -C $WS --approve-for-me …`，模型 / provider / base_url 用 `-c` 只对本进程覆盖，经本地代理到 api.deepseek.com（`PROXY_THINKING=disabled` 注入 `reasoning.effort=none` + 换 User-Agent + 丢 `x-codex-*`——DeepSeek 对 Codex 客户端无视 effort=none，见 bare/README §3.1）。论文用的就是 codex-cli，不是桌面版 | 非交互一跑到底，无需续跑语；工作目录里只有 `paper/ submission/ PROMPT.txt agent.env`，无 AGENTS.md | 代理日志：每行 `model=deepseek-flash`、`injected`、`user_agent`、`reasoning_tokens=0`、`reasoning_items=0` |
| **Claude Code** | 同上 | `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` 指到同一个代理（Anthropic Messages 格式，Paratera 直接支持；snse 批就是这样接的），`ANTHROPIC_MODEL` / `ANTHROPIC_DEFAULT_{HAIKU,SONNET}_MODEL` 全指 V4-Flash，`claude -p … --setting-sources project --dangerously-skip-permissions`（不加 `--setting-sources project` 会被 `~/.claude/settings.json` 里 cc-switch 的 env 块劫持到 DeepSeek 官方）；`run_bare.sh claude <paper>` | 无 `~/.claude/CLAUDE.md`、无工作目录 CLAUDE.md、MCP / skills 关；论文的 Claude Code 是 2.0.22 | 代理日志：Anthropic 回包无 `reasoning_tokens`，看 `thinking_blocks=0`（09-19 验过：Claude Code 的关思考开关只是不发字段，DeepSeek 缺省开；注入 disabled 后 0） |
| **DeepCode 基线运行**（本仓库） | 论文（paper.md + addendum 并稿）；**不读题面**——DeepCode 自带任务描述 | `deepcode_test/scripts/run_trial.sh`（原装 21ebc57f + 16 文件补丁，Flash 思考关） | 自己跑到底 | `runs/<paper>/logs/` |
| **DeepEvol Paper2Code 线** | 论文；不读题面 | `scripts/paper2code_canary.py … run --until compute`，第 9 步的树 `submit --dest-root` | 自己跑到底；不租机 | DeepEvol 仓库 `apps/v2/agent/paper2code/README.md` |

官方式接入（写一个 `BasePBSolver` 子类实现 `_run_agent`，`frontier-evals/project/paperbench/paperbench/solvers/base.py:46-70`）比上面多的只有：alcatraz 容器沙箱、`agent.log` 给 monitor 步查黑名单。我们用干净工作目录 + 代理日志 + 事后 `grep` 黑名单代替，判分走官方 `PBDirectSubmissionSolver`（`run_grade.sh`，池子 `~/pb_submissions/<paper>/<trial>/`）。

## 4. 本批（09-20）口径

deepseek-flash（api.deepseek.com）**思考开**；PaperBench 全部 20 篇（robust-clip 的官方 paper.md 缺方法章，照跑单独标）；三臂：Codex 桌面版、Claude 桌面版、DeepEvol Paper2Code 线（第 9 步的树）；**三臂都不执行代码**，桌面臂由 `audit_desktop.py` 核（执行过命令即 `CALIBER_BROKEN` 作废）；全部关图、md-only；裁判 V4-Flash + V4-Pro 解析器；每篇每臂 1 份先看方向，差值 < 0.03 再补；单篇噪声 0.025、历史组内摆动 0.09–0.19，n < 5 不说"优于"。跑法见分支 `0919-test` 的 README。

与 09-19 草案的差别：思考从"关"改"开"（桌面版关不掉，DeepSeek 缺省开，论文对照也是 -thinking）；CLI 臂换成桌面版（论文用的是桌面 agent；CLI 的 UA/x-codex 头会让 DeepSeek 走另一套 profile）；5 篇扩到 20 篇；加了不执行规则（09-19 的 fre / rice Codex 运行各跑了 177 min CPU 实验，作废）；3 小时用官方 `time_limit_template` 句子，不设硬上限。

与 bam 批（`INPUT_STANDARD.md`）的差别：Flash（bam 是 Pro）；题面后接**官方附注** + Execution 一条（bam 接的是我们自写的两句后缀）；续跑语用官方 `DEFAULT_CONTINUE_MESSAGE`。

## 5. `frontier-evals/` 为什么在本仓库里

2026-09-19 起 PaperBench 不再由 `setup.sh` 克隆，而是以普通文件 vendored 进来（`project/paperbench` + `project/common`，上游
commit 见 `patches/UPSTREAM_BASE.txt`；`.venv/`、`runs/`、`nanoeval/records/` 仍在 gitignore）。原因：本文这类说明要能指着仓库里的
文件和行号；同事 clone 即有裁判和题面，不再依赖能不能连上 GitHub / LFS。上游 `.gitattributes` 改名 `.gitattributes.upstream`，
数据以真实字节入库（用过的论文已水合，其余仍是 LFS 指针文本，`setup.sh` 按 `$PAPERS` 补）。

`patches/verify_paperbench.sh` 联网拉上游到临时目录逐文件比，证明 vendored 树 = 上游 + `paperbench_local_changes.patch`（5 文件）
+ `paperbench_changes/` 的新增文件；数据文件要么与上游相同，要么哈希等于上游 LFS 指针里的 oid。09-19 运行结果：`VERIFY_OK`
（384 相同，134 已水合并校验）。
