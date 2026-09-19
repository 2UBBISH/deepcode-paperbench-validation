# 0919 test：同一底座下，DeepCode vs Codex CLI vs Claude Code CLI（PaperBench Code-Dev，五篇）

分支 `0919-test`，给协作者用。目标：在**完全相同的条件**下（模型、思考状态、输入、题面）让三套系统各复现五篇论文，
把产物交回给 owner 统一判分。你不需要跑裁判，也不会看到评分标准（rubric 不在这个分支里，别去找）。

## 0. 三句话

- **比什么**：DeepCode 论文说它比 Codex 好 4.4×，但两边底座不同。这里把底座钉死，看脚手架本身值多少。
- **怎么保证公平**：四个量全部相同——`deepseek-flash` @ api.deepseek.com、思考关（有审计）、输入 = 同一份 `paper.md` + `addendum.md` + `blacklist.txt`（无 PDF、无图、无 rubric）、题面 = PaperBench 官方 Code-Dev 指令 + 官方附注。细节和证据：[`docs/CODEDEV-ARMS.md`](docs/CODEDEV-ARMS.md)。
- **你要交回什么**：`results/` 目录打包（`tar czf results_<你的名字>_<日期>.tgz results/`）。

## 1. 准备（一次）

| 需要 | 说明 |
| --- | --- |
| macOS / Linux，`git` `curl` `uv` `node`(≥18) `npm` `patch` `python3` | 不需要 Docker，不需要 GPU |
| DeepSeek 官方 API key | 写到 `~/Documents/env/deepseek.env`，一行 `DEEPSEEK_API_KEY=...`（**唯一的秘密**，脚本只 source 不打印；三臂都用它） |
| Codex CLI + Claude Code CLI | `npm i -g @openai/codex @anthropic-ai/claude-code`；脚本假定装在 `~/.local/node-v24.21.0/bin`，不是的话 `export PATH` 加上你的 npm bin |
| `~/.codex/AGENTS.md` 为空或不存在；`~/.claude/CLAUDE.md` 不存在 | 起跑前脚本会检查；这两个文件会把你的私人指令喂给 agent，破坏公平 |

```bash
git clone -b 0919-test git@github.com:2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
bash setup.sh          # 校验数据集、DeepCode = 上游 21ebc57f + 补丁、装 venv、生成口径配置、设 git 反抄袭封锁、建 results/ work/
PREFLIGHT_ONLY=1 PAPER=sapg bash deepcode_test/scripts/run_trial.sh      # 免费自检，全过再往下
```

你的 `~/.codex/config.toml`、`~/.claude/settings.json` **不会被读取或修改**：两个 CLI 的模型路由和 key 都在命令行 / 进程环境里给，
指向本机一个 40 行的代理（`deepcode_test/bare/paratera_proxy.py`），代理再带 key 去 api.deepseek.com。

## 2. 跑

```bash
nohup bash deepcode_test/scripts/run_batch.sh > runs/batch.log 2>&1 &     # 五篇 × 三臂，一条命令
tail -f runs/batch_*.txt                                                    # 账本：每臂每篇 start / exit / CALIBER_OK
```

- DeepCode 臂串行（一次一篇，每篇 2–5 小时）；Codex 和 Claude Code 臂与之并行、按篇推进（每篇 20–60 分钟）。整批约一天。
- 只跑一部分：`PAPERS="sapg pinn" ARMS="codex claude" bash deepcode_test/scripts/run_batch.sh`；已有结果的臂自动跳过，重跑先删 `results/<paper>/<arm>/`。
- 单臂单篇：`bash deepcode_test/bare/run_bare.sh codex sapg`、`bash deepcode_test/bare/run_bare.sh claude sapg`、`PAPER=sapg bash deepcode_test/scripts/run_trial.sh`。
- 费用：deepseek-flash 很便宜，整批大约几十元人民币；只有 API 费，没有机器费。
- 中断：`pkill -f run_batch.sh; pkill -f run_bare.sh; pkill -f "codex exec"; pkill -f "claude -p"; pkill -f stage_b_driver`。

## 3. 结果目录（交回的就是这个）

```
results/
  <paper>/                       sapg · pinn · adaptive-pruning · self-expansion · test-time-model-adaptation
    deepcode/
      submission/                DeepCode 生成的代码仓库
      RUN_LOG.txt                全程日志（末尾有"口径核验：N 次调用，reasoning_tokens 合计 0"）
    codex/
      submission/                Codex 生成的代码仓库（git repo）
      AUDIT.txt                  CALIBER_OK / CALIBER_BROKEN: …   ← 必须是 OK
      RUN_NOTES.md               版本、模型、代理、每轮 start/exit、黑名单检查、文件数
      proxy_requests.log         每个模型请求一行：model / injected / user_agent / usage / reasoning_items
      agent_events.jsonl         Codex 的事件流；interactions.log 续跑记录；PROMPT.txt 实际题面
    claude/                      同上（thinking_blocks 代替 reasoning_items）
```

交回：`tar czf results_<你的名字>_$(date +%m%d).tgz results/`，把压缩包和 `runs/batch_*.txt` 一起给 owner。
`work/` 是裸跑臂的工作目录（含论文拷贝和中间状态），不用交。

## 4. 不要做的

- 不要往 `work/<paper>-<arm>/` 或 `results/` 里手动改代码、不要回答 agent 的问题（脚本用官方续跑语自动续，最多 5 次）。
- 不要给 agent 论文的 PDF、图、官方实现或任何额外提示；`blacklist.txt` 里的仓库被 `git insteadOf` 挡住，跑完脚本会 grep 一遍。
- 不要改 `deepcode_test/`、`DeepCode/`、`config/` 里的东西；发现问题记下来交回。
- `AUDIT.txt` 不是 `CALIBER_OK` 的结果不算数，连同日志一起交回说明。

## 5. 目录

```
setup.sh                          一键环境（幂等）
DeepCode/                         HKUDS/DeepCode main 21ebc57f + patches/deepcode_local_changes.patch（补丁逐条说明见主分支 README §5）
patches/                          DeepCode 补丁与核验脚本
config/                           deepcode_config.template.json（口径：deepseek 档、思考关、32768）
frontier-evals/project/paperbench/
  data/papers/<五篇>/             paper.md · addendum.md · blacklist.txt（PaperBench 官方数据集的子集；PDF / assets / rubric 故意不带）
  paperbench/instructions/        code_only_instructions.txt（官方题面，逐字）
deepcode_test/
  scripts/run_batch.sh            整批
  scripts/run_trial.sh            DeepCode 臂（+ stage_b_driver.py）
  bare/run_bare.sh                Codex / Claude Code 臂；render_prompt.sh 出题面；paratera_proxy.py 关思考的代理；README.md 讲为什么必须有代理
docs/CODEDEV-ARMS.md              口径一页：题面 vs 答案卡、四臂怎么接、依据（官方代码行号）
results/  work/  runs/            产物（交回）· 裸跑工作目录 · 日志；都不入库
```

主分支 `master` 是完整的验证仓库（裁判、历史结果、全部坑）；本分支只留跑这一批所需的东西。
