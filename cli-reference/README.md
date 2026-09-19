# 裸跑臂（bare）：Codex 桌面版，PaperBench 官方指令 + 官方附注

三方对照的第三臂是「编码 agent 裸跑」：Codex 桌面版 + 同底座模型。这里放的是它的固定件和跑法；口径的
来龙去脉在 `docs/INPUT_STANDARD.md`（bam 那批），本文 §2 只写 2026-09-19 这批与之不同的地方。

## 1. 文件

| 文件 | 作用 |
| --- | --- |
| `render_prompt.sh <paper> <workspace> [--hours N] [--full]` | 建一篇论文的工作目录并生成 `PROMPT.txt`：官方 `code_only_instructions.txt`（字节级同源，替换 `/home/paper`、`/home/submission` 两处路径，以及 md-only 时 "in both PDF and markdown format"→"in markdown format"）+ `additional_notes.txt`；目录里默认只放 `paper.md`、`addendum.md`、`blacklist.txt`（与 DeepCode 臂同一份字节；`--full` 才放 PDF + assets），空 `agent.env`，`git init` 过的空 `submission/`，**不放 rubric.json / config.yaml**；LFS 指针直接拒绝；末尾自检打印 `PROMPT_OK` 和材料 sha256 |
| `additional_notes.txt` | PaperBench 自己的 `ADDITIONAL NOTES` 段（`paperbench/solvers/basicagent/prompts/templates.py` 的 `additional_notes_template`），按基准的填法填：Compute = `no_gpu_template`（这台 Mac 没有 GPU），Total Runtime = `no_time_limit_template`（不加 `--hours`）或 `time_limit_template`（`--hours N`），API keys 指向工作目录里的 `agent.env`（空文件，句子字面成立）。逐字来自基准，我们一个字不加 |
| `continue_message.txt` | Codex 停下来问 / 停下来没提交时人回的那一句 = PaperBench `DEFAULT_CONTINUE_MESSAGE`，原文。每回一次在工作目录 `interactions.log` 记一行 |
| `paratera_proxy.py` | 直通代理 `127.0.0.1:8787 → llmapi.paratera.com`，逐请求记 model / thinking 字段 / usage（含 `reasoning_tokens`）。`PROXY_THINKING=disabled` 时给每个请求体加 `thinking:{type:disabled}`（Paratera 只认这种写法）并记 `injected`——这是"思考关"口径的旋钮，别的不动。回包侧证据：OpenAI 线（Codex）看 `usage.reasoning_tokens`；Anthropic 线（Claude Code，`/v1/messages`）usage 里没有这个数，思考以 `type: thinking` 的内容块出现，代理逐回包数出 `thinking_blocks`（文档：content 里的 thinking / redacted_thinking 块；流：`content_block_start` 事件）。09-19 实测 V4-Flash：不注入 → `thinking_blocks: 1`（Paratera 在 Anthropic 线上默认思考开），注入 → 0，文档与流都对。两个数都必须全程为 0。key 不落日志 |
| `bare_prompt_suffix.txt` | bam 那批（09-15）用的两句自写后缀；**本批不用**（被基准自己的附注取代），留作历史 |

## 2. 2026-09-19 批：Flash 思考关，5 篇，Codex 臂 vs DeepCode 基线运行

| 项 | 本批 | 与 bam 批（INPUT_STANDARD）的差别 |
| --- | --- | --- |
| 底座 / 思考 | `DeepSeek-V4-Flash`，**思考关**（代理注入；日志每行 `reasoning_tokens` 必须为 0） | bam 是 V4-Pro 思考开 |
| 提示词 | 官方 Code-Dev 指令原文 + **基准自己的 `ADDITIONAL NOTES`**（无 GPU、无时限数字、agent.env、root、"用满时间/不要只写计划"） | bam 用的是我们自写的两句后缀，没有附注段——owner 09-19 指出"时间和运行要求被去掉了"，改回基准原文 |
| 时限句 | 默认 `no_time_limit_template`："work until you have reproduced all the core contributions"——另两臂也没被告知任何小时数 | 要对齐 PaperBench 官方跑法的小时数就 `--hours 12`，三臂口径记录里写明 |
| 续跑 | Codex 停下来（问问题、或说完了但没提交）→ 回 `CONTINUE.txt` 原文，**最多 5 次**，每次记 `interactions.log`；它说完了且已 `git commit` 就停 | bam 是 "Continue; no further input will be provided." |
| 论文 | `sapg`（77 叶）、`pinn`（126）、`adaptive-pruning`（86）、`self-expansion`（70）、`test-time-model-adaptation`（86）——Code-Dev 叶数 70–130 的中等篇，`paper.md` 都经 `check_paper_md.py` 核过完整；**robust-clip 剔除**（官方 md 缺 §2–§3，见 PITFALLS §E）；sapg / pinn 的基线分已有但那是 Paratera serving，本批重跑 | bam 单篇 |
| 输入 | **三臂同一份字节**（owner 09-19）：CLI 工作目录里只有 `paper.md`、`addendum.md`、`blacklist.txt`，**不放 PDF 和 assets**——DeepCode / 本线本来就只读 md + addendum；官方题面里 "in both PDF and markdown format" 那一句改成 "in markdown format"（除两处路径外唯一的改字，`render_prompt.sh` 的自检把它算进去）。`--full` 可恢复官方目录 | bam 给了 PDF 和真图（偏帮裸跑，当时记为保守方向） |
| 裁判 | `DeepSeek-V4-Flash` + `DeepSeek-V4-Pro` 结构化解析器（`PB_JUDGE_MODEL=DeepSeek-V4-Flash bash run_grade.sh`），两臂同一个 | bam 是 V4-Pro |
| 样本 | 每篇每臂 1 份先看方向；差值 < 0.1 的论文再各补 1 份。单篇噪声 0.025（sapg 同份重跑），历史组内摆动 0.09–0.19，n < 5 不说"优于" | 同 |

## 3. 裸跑臂怎么跑（Codex CLI / Claude Code CLI，一篇一条命令）

两个 CLI 都装在 `~/.local/node-v24.21.0/bin`（npm -g 的前缀，不在默认 PATH；脚本自己加）。版本：codex-cli 0.155.1、Claude Code 2.1.278。
两个 CLI 由 owner 的 cc-switch 指向 **api.deepseek.com / `deepseek-flash`**（Codex：`~/.codex/config.toml`；Claude Code：`~/.claude/settings.json` 的 env 块），key 是他们自己的，脚本不碰、不打印。

```bash
V=~/Documents/env/paperbench-judge/validation
bash $V/deepcode_test/bare/run_bare.sh codex  robust-clip          # 工作目录 ~/Documents/env/bare-0919/robust-clip-codex
bash $V/deepcode_test/bare/run_bare.sh claude robust-clip          # 同一篇的 Claude Code 臂，端口不同，可以同时跑
#   [--hours 12] 时限句用 time_limit_template；[--no-pool] 不拷进判分池；[--root DIR] [--pool DIR] [--max-continues N]
```

脚本做的事（`run_bare.sh`）：`render_prompt.sh` 建目录出 `PROMPT.txt` → 卫生检查（`~/.codex/AGENTS.md` 为空、`~/.claude/CLAUDE.md` 不存在、工作目录里没有 AGENTS.md / CLAUDE.md）
→ 起本地代理（`PROXY_UPSTREAM=https://api.deepseek.com PROXY_THINKING=disabled PROXY_USER_AGENT=paperbench-bare/1`，鉴权头原样透传）
→ 非交互跑到底（`codex exec` / `claude -p`，stdin 关闭）→ `submission/` 里没有 commit 就用官方 `DEFAULT_CONTINUE_MESSAGE` 续跑（`codex exec resume --last` / `claude --resume`），最多 5 次，记 `interactions.log`
→ 审计写 `AUDIT.txt`（`CALIBER_OK` / `CALIBER_BROKEN: …`）+ 黑名单 grep + 文件数 → 拷进 `~/pb_submissions/<paper>/<arm>N/`。

### 3.1 为什么必须有这层代理：直连 DeepSeek 时两个 CLI 都关不掉思考（09-19 实测）

DeepSeek 文档（api-docs.deepseek.com/zh-cn/guides/thinking_mode）：OpenAI / Anthropic 格式的开关是 `thinking:{type:enabled|disabled}`，Responses API 的开关是 `reasoning:{effort:none}`（none = 关）；**默认开，effort 默认 high**。

| CLI | 试过的原生开关 | 请求里实际发的 | DeepSeek 回包 |
| --- | --- | --- | --- |
| Claude Code | `CLAUDE_CODE_DISABLE_THINKING=1`、`MAX_THINKING_TOKENS=0`、`CLAUDE_CODE_DISABLE_ADAPTIVE_THINKING=1`、`CLAUDE_CODE_EFFORT_LEVEL=low` | 前三个 = **不发 thinking 字段**；默认和 effort=low 发 `thinking:{type:adaptive}` | 全部仍有 1 个 thinking 块——缺省即开，没有能发 `disabled` 的开关 |
| Codex | `-c model_reasoning_effort=none` | `reasoning:{effort:none}`（正确） | 仍有 reasoning（11–124 token）。**同一个请求体用 curl 重放 → 0**；逐个加回 Codex 的请求头：`User-Agent: codex_exec/0.155.1` 或 `x-codex-turn-metadata` 任一存在 → 14；`originator` / `session-id` / `thread-id` / 其它 `x-codex-*` → 0。即 DeepSeek 对 Codex 客户端有专门档位，无视 effort=none |

代理因此做三件事：Anthropic / chat 线注入 `thinking:{type:disabled}`，responses 线注入 `reasoning.effort=none`；把 User-Agent 换成 `paperbench-bare/1`；丢掉 `x-codex-*` 头。验证：Codex 连跑 3 次 `reasoning_tokens 0 / reasoning_items 0`（不做 UA/头替换时同样注入仍为 14–96）；Claude Code 2 次 `thinking_blocks 0`。真实运行 robust-clip 前 49 个请求全部 0。

| 臂 | 怎么接 | 备注 |
| --- | --- | --- |
| Codex | `codex exec -C $WS --approve-for-me -c sandbox_workspace_write.network_access=true -c model=deepseek-flash -c model_provider=custom -c model_providers.custom.base_url=http://127.0.0.1:8787/v1 --json`；**不改 `~/.codex/config.toml`**，`-c` 只对本进程生效；`--approve-for-me` = workspace-write 沙箱 + 自动审批（论文的 auto approval） | 0.155 已删掉 `wire_api = "chat"`，只剩 responses；base_url 必须带 `/v1`（Codex 只追加 `/responses`）；stdin 必须关（否则 "Reading additional input from stdin" 挂住）；Codex 自己发 `reasoning.effort=high`，记在 `thinking_fields` |
| Claude Code | `claude -p … --setting-sources project --session-id <uuid> --dangerously-skip-permissions --output-format stream-json --disable-slash-commands --strict-mcp-config --no-chrome`，env：`ANTHROPIC_BASE_URL=http://127.0.0.1:8788/anthropic`、`ANTHROPIC_MODEL` + 三个 `ANTHROPIC_DEFAULT_*_MODEL` + `CLAUDE_CODE_SUBAGENT_MODEL` 全 deepseek-flash、`ANTHROPIC_AUTH_TOKEN` = settings.json 里那把（脚本进程内读，不打印）、去掉 `CLAUDECODE` | **`--setting-sources project` 不能少**：`~/.claude/settings.json` 的 env 块会盖过进程 env（第一次冒烟因此直连了官方、思考未关）；Claude 自己发 `thinking:{type:adaptive}` 被替换 |

审计规则（`AUDIT.txt`）：每行 `model=deepseek-flash`、每行有 `injected`、Codex 每行有 `user_agent`、每行 `reasoning_tokens`=0 且 Codex `reasoning_items`=0 / Claude `thinking_blocks`=0，否则 `CALIBER_BROKEN`；黑名单仓库在提交里只允许出现在引用文字里。

**未定**：DeepCode 基线现在走 Paratera 的 `DeepSeek-V4-Flash`，两个裸跑臂走 api.deepseek.com 的 `deepseek-flash`——同一模型、不同 serving。要严格同口径，基线的 ENV_FILE 也指到 api.deepseek.com。

### 3.2 一条命令跑完整批（基线 + 两个裸跑臂 × 五篇）

```bash
cd ~/Documents/env/paperbench-judge/validation
nohup bash deepcode_test/scripts/run_batch.sh > runs/batch_0919.log 2>&1 &      # PAPERS="…" ARMS="baseline codex claude" 可缩
tail -f runs/batch_*.txt                                                          # 账本：每臂每篇 start / exit / CALIBER_OK
```
基线串行（一次一篇，3–4 h/篇，`ENV_FILE=~/Documents/env/deepseek.env` 里放 `DEEPSEEK_API_KEY`），两个裸跑臂与之并行、按篇推进；池子里已有的臂自动跳过。

### 3.3 能不能靠 cc-switch 关思考

不能：cc-switch 的本地代理（`enableLocalProxy`）只做转发 / 记账 / 故障切换，没有改请求体的能力；关思考需要请求体里的字段。能做的是在 cc-switch 里各建一个"DeepSeek (thinking off)"档，base_url 指到本地代理（Codex `http://127.0.0.1:8787/v1`，Claude `http://127.0.0.1:8788/anthropic`），key 不变——切到这个档后 `codex exec` / `claude -p` 裸命令就是关思考的，代理需常驻（`PROXY_UPSTREAM=https://api.deepseek.com PROXY_THINKING=disabled PROXY_USER_AGENT=paperbench-bare/1`）。`run_bare.sh` 不依赖这个：它自己起代理、用 `-c` / env 只对本进程覆盖 base_url，cc-switch 保持原样即可。

## 4. DeepCode 臂（本仓库基线运行）与判分

基线运行：`PAPER=<id> TRIAL=trial1 ENV_FILE=~/Documents/env/deepseek.env DEEPCODE_EXPECT_MODEL=deepseek-flash nohup bash deepcode_test/scripts/run_trial.sh`（09-19 起 `deepcode_config` 在 `deepseek` 档：api.deepseek.com、deepseek-flash、每次请求 `thinking:{type:disabled}`；换回 Paratera：`DEEPCODE_REGEN_CONFIG=1 DEEPCODE_CONNECTION=paratera bash setup.sh`）。五篇都已登记（`run_trial.sh` 的表 + 裁判 `paper_split`）。上游 DeepCode 仓库本身没有 PaperBench 入口（论文没放评测脚手架），`run_trial.sh` / `run_batch.sh` 就是入口。分数回填 `docs/RESULTS-HISTORY.md` 新节。
