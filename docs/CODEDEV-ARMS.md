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

## 3. 四臂各拿什么、怎么接

裁判只看提交目录，**怎么把提交送进去不影响 Code-Dev 分**；差别只在输入和 harness 条件。

| 臂 | 输入 | 接法 | harness 条件 | 记录 |
| --- | --- | --- | --- | --- |
| **Codex CLI** | 题面 + 官方附注（`deepcode_test/bare/render_prompt.sh` 生成的 `PROMPT.txt`）+ `paper.md` / `addendum.md` / `blacklist.txt`（**不给 PDF 和 assets**，与 DeepCode 臂同一份字节；owner 09-19） | `deepcode_test/bare/run_bare.sh codex <paper>`：`codex exec -C $WS --approve-for-me …`，模型 / provider / base_url 用 `-c` 只对本进程覆盖，经本地代理到 api.deepseek.com（`PROXY_THINKING=disabled` 注入 `reasoning.effort=none` + 换 User-Agent + 丢 `x-codex-*`——DeepSeek 对 Codex 客户端无视 effort=none，见 bare/README §3.1）。论文用的就是 codex-cli，不是桌面版 | 非交互一跑到底，无需续跑语；工作目录里只有 `paper/ submission/ PROMPT.txt agent.env`，无 AGENTS.md | 代理日志：每行 `model=deepseek-flash`、`injected`、`user_agent`、`reasoning_tokens=0`、`reasoning_items=0` |
| **Claude Code** | 同上 | `ANTHROPIC_BASE_URL=http://127.0.0.1:<port>` 指到同一个代理（Anthropic Messages 格式，Paratera 直接支持；snse 批就是这样接的），`ANTHROPIC_MODEL` / `ANTHROPIC_DEFAULT_{HAIKU,SONNET}_MODEL` 全指 V4-Flash，`claude -p … --setting-sources project --dangerously-skip-permissions`（不加 `--setting-sources project` 会被 `~/.claude/settings.json` 里 cc-switch 的 env 块劫持到 DeepSeek 官方）；`run_bare.sh claude <paper>` | 无 `~/.claude/CLAUDE.md`、无工作目录 CLAUDE.md、MCP / skills 关；论文的 Claude Code 是 2.0.22 | 代理日志：Anthropic 回包无 `reasoning_tokens`，看 `thinking_blocks=0`（09-19 验过：Claude Code 的关思考开关只是不发字段，DeepSeek 缺省开；注入 disabled 后 0） |
| **DeepCode 基线运行**（本仓库） | 论文（paper.md + addendum 并稿）；**不读题面**——DeepCode 自带任务描述 | `deepcode_test/scripts/run_trial.sh`（原装 21ebc57f + 16 文件补丁，Flash 思考关） | 自己跑到底 | `runs/<paper>/logs/` |
| **DeepEvol Paper2Code 线** | 论文；不读题面 | `scripts/paper2code_canary.py … run --until compute`，第 9 步的树 `submit --dest-root` | 自己跑到底；不租机 | DeepEvol 仓库 `apps/v2/agent/paper2code/README.md` |

官方式接入（写一个 `BasePBSolver` 子类实现 `_run_agent`，`frontier-evals/project/paperbench/paperbench/solvers/base.py:46-70`）比上面多的只有：alcatraz 容器沙箱、`agent.log` 给 monitor 步查黑名单。我们用干净工作目录 + 代理日志 + 事后 `grep` 黑名单代替，判分走官方 `PBDirectSubmissionSolver`（`run_grade.sh`，池子 `~/pb_submissions/<paper>/<trial>/`）。

## 3.1 桌面版（本批最终口径，owner 09-19）

论文的三个对照里 Cursor 是桌面 IDE，Claude Code 2.0.22 和 codex-cli 0.47.0 都是终端 CLI；owner 决定本批用 **Codex 桌面版 + Claude 桌面版（Code 标签）** 对 DeepCode，
不做 Cursor、不做论文底座复现。桌面版没有命令行覆盖，模型路由来自各自配置（cc-switch 写），所以用 cc-switch 各建一个指向本机代理的档；
代理钉模型（`PROXY_FORCE_MODEL`，记 `model_in`）、注入思考关、换 UA、丢 `x-codex-*`、带 key、逐请求记录——`desktop_prep.sh` 起代理并出题面，人驱动 app，`desktop_finish.sh` 审计并收进 `results/`。
输入与 CLI 臂完全相同（md + addendum + blacklist，官方题面 + 官方附注，官方续跑语 ≤5 次）。桌面版多出的插件（browser / computer-use / chrome、skills / MCP）关掉并记进 `RUN_NOTES.md`。

## 4. 本批（09-19）口径

deepseek-flash（api.deepseek.com，owner 的 cc-switch 路由；基线仍在 Paratera，见 bare/README §3 末）思考关；5 篇：sapg、pinn、adaptive-pruning、self-expansion、test-time-model-adaptation（Code-Dev 叶 70–130；robust-clip 因官方 paper.md 缺方法章被剔除，`check_paper_md.py`）；两臂裸跑 + DeepCode 基线（+ 本线 stage 9）；全部关图；裁判 V4-Flash + V4-Pro 解析器；每篇每臂 1 份先看方向，差值 < 0.1 再补；单篇噪声 0.025、历史组内摆动 0.09–0.19，n < 5 不说"优于"。跑法见 `deepcode_test/bare/README.md`。

与 bam 批（`INPUT_STANDARD.md`）的三处差别：Flash 关（bam 是 Pro 开）；题面后接**官方附注**（bam 接的是我们自写的两句后缀——owner 09-19 指出时间和运行要求被去掉了，改回基准原文）；续跑语用官方 `DEFAULT_CONTINUE_MESSAGE`（CLI 非交互跑基本用不上）。

## 5. `frontier-evals/` 为什么在本仓库里

2026-09-19 起 PaperBench 不再由 `setup.sh` 克隆，而是以普通文件 vendored 进来（`project/paperbench` + `project/common`，上游
commit 见 `patches/UPSTREAM_BASE.txt`；`.venv/`、`runs/`、`nanoeval/records/` 仍在 gitignore）。原因：本文这类说明要能指着仓库里的
文件和行号；同事 clone 即有裁判和题面，不再依赖能不能连上 GitHub / LFS。上游 `.gitattributes` 改名 `.gitattributes.upstream`，
数据以真实字节入库（用过的论文已水合，其余仍是 LFS 指针文本，`setup.sh` 按 `$PAPERS` 补）。

`patches/verify_paperbench.sh` 联网拉上游到临时目录逐文件比，证明 vendored 树 = 上游 + `paperbench_local_changes.patch`（5 文件）
+ `paperbench_changes/` 的新增文件；数据文件要么与上游相同，要么哈希等于上游 LFS 指针里的 oid。09-19 运行结果：`VERIFY_OK`
（384 相同，134 已水合并校验）。
