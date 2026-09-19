# 裸跑臂（bare）：Codex 桌面版，PaperBench 官方指令 + 官方附注

三方对照的第三臂是「编码 agent 裸跑」：Codex 桌面版 + 同底座模型。这里放的是它的固定件和跑法；口径的
来龙去脉在 `docs/INPUT_STANDARD.md`（bam 那批），本文 §2 只写 2026-09-19 这批与之不同的地方。

## 1. 文件

| 文件 | 作用 |
| --- | --- |
| `render_prompt.sh <paper> <workspace> [--hours N]` | 建一篇论文的工作目录并生成 `PROMPT.txt`：官方 `code_only_instructions.txt`（字节级同源，只替换 `/home/paper`、`/home/submission` 两处路径）+ `additional_notes.txt`；目录里放基准给 agent 的那几样（`paper/` 五件、空 `agent.env`、`git init` 过的空 `submission/`），**不放 rubric.json / config.yaml**；末尾自检打印 `PROMPT_OK` 和四件材料的 sha256 |
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
| 论文 | `sapg`（77 叶）、`pinn`（126）、`robust-clip`（70）、`self-expansion`（70）、`test-time-model-adaptation`（86）——Code-Dev 叶数 70–130 的中等篇；sapg / pinn 的基线运行分数已有（0.716 / 0.691、0.670） | bam 单篇 |
| 图 | Codex 能读目录里的 assets，但 PaperBench 数据集里的 jpg 是 LFS 指针（`render_prompt.sh` 会提示）。**不补图**：DeepCode 臂本批关图，两臂都不看图 | bam 的图是真字节 |
| 裁判 | `DeepSeek-V4-Flash` + `DeepSeek-V4-Pro` 结构化解析器（`PB_JUDGE_MODEL=DeepSeek-V4-Flash bash run_grade.sh`），两臂同一个 | bam 是 V4-Pro |
| 样本 | 每篇每臂 1 份先看方向；差值 < 0.1 的论文再各补 1 份。单篇噪声 0.025（sapg 同份重跑），历史组内摆动 0.09–0.19，n < 5 不说"优于" | 同 |

## 3. 裸跑臂怎么跑（Codex CLI / Claude Code CLI，一篇一条命令）

两个 CLI 都装在 `~/.local/node-v24.21.0/bin`（npm -g 的前缀，不在默认 PATH；脚本自己加）。版本：codex-cli 0.155.1、Claude Code 2.1.278。

```bash
V=~/Documents/env/paperbench-judge/validation
bash $V/deepcode_test/bare/run_bare.sh codex  robust-clip          # 工作目录 ~/Documents/env/bare-0919/robust-clip-codex
bash $V/deepcode_test/bare/run_bare.sh claude robust-clip          # 同一篇的 Claude Code 臂，端口不同，可以同时跑
#   [--hours 12] 时限句用 time_limit_template；[--no-pool] 不拷进判分池；[--root DIR] [--pool DIR] [--max-continues N]
```

脚本做的事（`run_bare.sh`）：`render_prompt.sh` 建目录出 `PROMPT.txt` → 卫生检查（`~/.codex/AGENTS.md` 为空、`~/.claude/CLAUDE.md` 不存在、工作目录里没有 AGENTS.md / CLAUDE.md）
→ 起代理（`PROXY_THINKING=disabled`，`PROXY_UPSTREAM_KEY_ENV=PARATERA_API_KEY`：**key 只在代理的子 shell 里 source，CLI 自己的鉴权被代理替换**，所以 cc-switch 当前切在哪个档都不影响）
→ 非交互跑到底（`codex exec` / `claude -p`，stdin 关闭）→ `submission/` 里没有 commit 就用官方 `DEFAULT_CONTINUE_MESSAGE` 续跑（`codex exec resume --last` / `claude --resume`），最多 5 次，记 `interactions.log`
→ 审计写 `AUDIT.txt`（`CALIBER_OK` / `CALIBER_BROKEN: …`）+ 黑名单 grep + 文件数 → 拷进 `~/pb_submissions/<paper>/<arm>N/`。

| 臂 | 怎么接 | 09-19 实测 |
| --- | --- | --- |
| Codex | `codex exec -C $WS --approve-for-me -c sandbox_workspace_write.network_access=true -c model=DeepSeek-V4-Flash -c model_provider=custom -c model_providers.custom.base_url=http://127.0.0.1:8787/v1 --json`；**不改 `~/.codex/config.toml`**，`-c` 只对本进程生效；`--approve-for-me` = workspace-write 沙箱 + 自动审批（论文的 auto approval） | 0.155 已**删掉 `wire_api = "chat"`**，只剩 responses 线；代理在 `/v1/responses` 上注入 `thinking:{type:disabled}` 验过：注入 → `reasoning_tokens 0, reasoning_items 0`；不注入 → 14 / 14（Codex 自己发 `reasoning.effort=high`，记在 `thinking_fields`）。base_url 必须带 `/v1`（Codex 只追加 `/responses`） |
| Claude Code | `claude -p … --setting-sources project --dangerously-skip-permissions --output-format stream-json --disable-slash-commands --strict-mcp-config --no-chrome`，env：`ANTHROPIC_BASE_URL=http://127.0.0.1:8788`、`ANTHROPIC_MODEL` + 三个 `ANTHROPIC_DEFAULT_*_MODEL` + `CLAUDE_CODE_SUBAGENT_MODEL` 全 V4-Flash、`ANTHROPIC_AUTH_TOKEN` 占位、去掉 `CLAUDECODE`（本机是从 Claude Code 会话里起的） | **`--setting-sources project` 不能少**：这台机的 `~/.claude/settings.json`（cc-switch 写的）有 env 块指向 `api.deepseek.com/anthropic` + deepseek-v4-pro，会盖过进程 env——第一次冒烟就这样直连了 DeepSeek 官方（花了那边约 0.1 USD，未经代理、思考未关）。加了之后走代理：`thinking_blocks 0`，Claude 自己发的 `thinking:{type:adaptive}` 被替换 |

审计规则（`AUDIT.txt`）：每行 `model=DeepSeek-V4-Flash`、每行有 `injected` 和 `auth=proxy:PARATERA_API_KEY`、每行 `reasoning_tokens`=0 且 Codex `reasoning_items`=0 / Claude `thinking_blocks`=0，否则 `CALIBER_BROKEN`；黑名单仓库在提交里只允许出现在引用文字里。

## 4. DeepCode 臂（本仓库基线运行）与判分

基线运行照旧：`PAPER=<id> TRIAL=trial1 ENV_FILE=… nohup bash deepcode_test/scripts/run_trial.sh`（Flash 思考关，`run_trial.sh` 里已是这个口径；
未登记的论文会自动推导 `TITLE_KEY` / `BLOCK_REPO`）。三篇新论文判分前要在裁判里登记（`nano/eval.py` 的 `paper_split` Literal +
`experiments/splits/<id>.txt`，补丁里 pinn 的写法），这一步由维护本仓库的会话做。分数回填 `docs/RESULTS-HISTORY.md` 新节。
