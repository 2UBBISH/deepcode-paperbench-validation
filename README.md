# 0919 test：Codex 桌面版 vs Claude 桌面版 vs DeepCode（同一底座，PaperBench Code-Dev，五篇）

分支 `0919-test`。目标：在**完全相同的条件**下让三套系统各复现五篇论文，产物交回 owner 统一判分。
本分支不含裁判和评分标准（rubric 故意不带，别去找）。

## 0. 三臂各拿什么（公平性一览）

| | Codex 桌面版 | Claude 桌面版（Code 标签） | DeepCode（本仓库基线） |
| --- | --- | --- | --- |
| 论文材料 | `work/<paper>-codex-desktop/paper/`：`paper.md` `addendum.md` `blacklist.txt` | 同左（`…-claude-desktop/paper/`） | `paper.md` + `addendum.md` 并成一个 markdown（DeepCode 只吃一个文件） |
| **不给** | PDF、图、`rubric.json`、`config.yaml` | 同 | 同（`run_trial.sh` 不拷） |
| 题面 | `PROMPT.txt` = PaperBench 官方 `code_only_instructions.txt` 逐字 + PaperBench 自己的 `ADDITIONAL NOTES`，只替换两处路径、"in both PDF and markdown format"→"in markdown format" | 同一份 | **无题面**——DeepCode 自带的规划/写码提示词就是被测系统的一部分 |
| 续跑 | 停下来就回官方 `DEFAULT_CONTINUE_MESSAGE`（`CONTINUE.txt`），≤5 次，记 `interactions.log` | 同 | 不需要，自己跑到底 |
| 模型 | `deepseek-flash` @ api.deepseek.com（cc-switch 的 DeepSeek 档；跑完从 app 自己的会话日志核每一轮的 model） | 同（档里 `ANTHROPIC_MODEL` 和三个 `ANTHROPIC_DEFAULT_*_MODEL` 都是 deepseek-flash，否则 app 会拿 Claude 的模型名去请求） | `deepcode_config` 的 `deepseek` 档 |
| 思考 | **开**（DeepSeek 缺省；论文的三个对照 agent 也都是 -thinking）。证据：`~/.codex/sessions/*/rollout-*.jsonl` 里每轮的 `reasoning_output_tokens` | **开**。证据：`~/.claude/projects/<目录>/*.jsonl` 里 assistant 消息的 thinking 块 / `thinking_tokens` | **开**：每请求带 `thinking:{type:enabled}`；跑完核 llm 日志 `reasoning_tokens` 合计 > 0 |
| 黑名单 | 全局 `git insteadOf` 挡克隆 + 跑完 grep 提交 | 同 | 同 + MCP 层 `DEEPCODE_URL_DENYLIST` |
| 额外能力 | 插件（browser / computer-use / chrome）**关**；网页搜索允许（官方题面允许上网） | 插件 / skills / MCP **关**；无 `CLAUDE.md`、新目录无记忆 | 自带 7 个 MCP 工具（fetch / github 下载等，被测系统的一部分） |
| 审计 | `AUDIT.txt` 必须 `CALIBER_OK`（`audit_desktop.py`：按工作目录匹配会话日志，每轮 model = deepseek-flash、思考 token > 0），会话日志副本随结果交回 | 同 | `RUN_LOG.txt` 末尾"口径核验（思考=enabled）… reasoning_tokens 合计 N"，N > 0 |
| 时长 | **硬上限 3 小时**（`desktop_prep.sh` 打印截止时刻）：到点就 `desktop_finish.sh`，`submission/` 里有什么交什么并记录 | 同 | 自己跑到底（14 h 硬顶） |
| 判分 | PaperBench Code-Dev（只判 Code Development 叶，不执行代码），owner 在主分支用同一裁判判三臂 | 同 | 同 |

和 PaperBench 官方设定的差别只有三处，三臂一致：① 不给 PDF / 图（官方给）；② 人代替 harness 回续跑语（官方的 BasicAgent 也是这句）并设 3 小时上限；③ 底座 deepseek-flash（官方是各家前沿模型）。
"用满时间"那句是官方附注原文，没有时限数字时约束不了时长——agent 自己判断做完就停；真正兜底的是续跑 ≤5 次和 3 小时上限。
依据（官方代码行号、论文原文）：[`docs/CODEDEV-ARMS.md`](docs/CODEDEV-ARMS.md)。思考关的版本（要代理）留在 [`deepcode_test/bare/README.md`](deepcode_test/bare/README.md) §3.1 作参考，本批不用。

## 1. 准备（一次）

| 需要 | 说明 |
| --- | --- |
| macOS，`git` `curl` `uv` `node`(≥18) `npm` `patch` `python3` | 不需要 Docker / GPU |
| DeepSeek 官方 API key | `~/Documents/env/deepseek.env`，一行 `DEEPSEEK_API_KEY=...`（唯一的秘密，脚本只 source 不打印；三臂都用它） |
| Codex 桌面版、Claude 桌面版、cc-switch | cc-switch 里各建一个 DeepSeek 档（真 key 填在档里）：Codex → `base_url = "https://api.deepseek.com/v1"`、`wire_api = "responses"`、`model = "deepseek-flash"`；Claude 桌面 → `ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"`，`ANTHROPIC_MODEL`、`ANTHROPIC_DEFAULT_HAIKU_MODEL`、`ANTHROPIC_DEFAULT_SONNET_MODEL`、`ANTHROPIC_DEFAULT_OPUS_MODEL`、`CLAUDE_CODE_SUBAGENT_MODEL` 全 `deepseek-flash`。切档后重启 app |
| `~/.codex/AGENTS.md` 为空、`~/.claude/CLAUDE.md` 不存在 | 私人指令会喂给 agent，破坏公平 |

```bash
git clone -b 0919-test git@github.com:2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
bash setup.sh                                                             # 数据集校验、DeepCode + venv、口径配置、git 封锁、results/ work/
PREFLIGHT_ONLY=1 PAPER=sapg bash deepcode_test/scripts/run_trial.sh       # DeepCode 臂免费自检
```

## 2. 跑

**DeepCode 臂**（全自动，一次一篇，每篇 2–5 小时；五篇串行约一天）：
```bash
ARMS=baseline nohup bash deepcode_test/scripts/run_batch.sh > runs/batch.log 2>&1 &     # 五篇；单篇：PAPER=sapg bash deepcode_test/scripts/run_trial.sh
```

**桌面臂**（每篇每臂：起代理 → 你在 app 里粘题面 → 结束收尾；每篇 20–60 分钟，可以和 DeepCode 臂并行）：
```bash
bash deepcode_test/bare/desktop_prep.sh codex sapg      # 建 work/sapg-codex-desktop、记开始时间和 3 小时截止、题面进剪贴板、打印 app 里要做的 5 步
#   → Codex app：打开那个文件夹，审批全自动，粘贴题面，停了就回 CONTINUE.txt（记 interactions.log）
bash deepcode_test/bare/desktop_finish.sh codex sapg    # 从 app 会话日志审计、黑名单、收进 results/sapg/codex/（到 3 小时也这样收）
bash deepcode_test/bare/desktop_prep.sh claude sapg     # Claude 桌面版 Code 标签，同样五步
bash deepcode_test/bare/desktop_finish.sh claude sapg
```
app 里的规则：题面是第一条也是唯一一条消息，一个字不加不减；它问问题只回 `CONTINUE.txt`（最多 5 次）；不给任何提示、不改它的代码；说完成且 `submission/` 有 commit 就收尾。
每篇跑完把 `RUN_NOTES.md` 里三行"fill in"（app 版本、留着的插件、审批模式）填上。

## 3. 结果目录（交回的就是这个）

```
results/<paper>/            sapg · pinn · adaptive-pruning · self-expansion · test-time-model-adaptation
  deepcode/  submission/  RUN_LOG.txt
  codex/     submission/  AUDIT.txt  RUN_NOTES.md  session_logs/  interactions.log  PROMPT.txt
  claude/    submission/  AUDIT.txt  RUN_NOTES.md  session_logs/  interactions.log  PROMPT.txt
```
交回：`tar czf results_<你的名字>_$(date +%m%d).tgz results/`。`work/` 是工作目录，不用交。`AUDIT.txt` 不是 `CALIBER_OK` 的结果不算数，连同日志交回说明。

## 4. 目录

```
setup.sh · DeepCode/ · patches/ · config/          DeepCode 臂（HKUDS/DeepCode 21ebc57f + 补丁；deepseek 档、思考开、32768）
frontier-evals/project/paperbench/data/papers/     五篇：paper.md · addendum.md · blacklist.txt（官方数据集子集）
frontier-evals/project/paperbench/paperbench/instructions/code_only_instructions.txt   官方题面
deepcode_test/scripts/run_batch.sh · run_trial.sh  DeepCode 臂
deepcode_test/bare/desktop_prep.sh · desktop_finish.sh · audit_desktop.py · render_prompt.sh · additional_notes.txt · continue_message.txt
deepcode_test/bare/run_bare.sh · paratera_proxy.py · audit.py   同样的两臂改用 CLI 非交互 + 代理关思考（不是本批口径，留作对照）
docs/CODEDEV-ARMS.md                               口径一页
results/ work/ runs/                               产物（交回）· 工作目录 · 日志；不入库
```
主分支 `master` 是完整的验证仓库（裁判、历史结果、全部坑）。
