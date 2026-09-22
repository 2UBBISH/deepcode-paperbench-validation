# 交接：用本线把一篇论文跑到第 9 步（compute），另一个会话用

写于 2026-09-18 22:40，给**另一个 Claude 会话**用。目标：拿 PaperBench 的一篇论文，跑本线的第 1–9 步（intake → plan →
references → acquire → index → implement → compute），拿到生成的代码树；**不租机器、不判分、不动 git**。本线的主会话
同时在做 T13 / T4b / pinn 噪声样本，下面写清哪些会冲突、哪些不会。

## 0. 不会冲突的 / 会冲突的

| 事 | 结论 |
| --- | --- |
| 运行目录 | 各用各的：`~/Documents/search/paper2code-runs/<你的名字>`，别复用 `sapg*` / `pinn*` 这些名字 |
| 机器 | 第 9 步只做静态估算 + 查价，**不租机器**；就算跑到第 10 步，`release` 也只动本运行 `lease.json` 里记的那台，不会误删别人的 |
| 模型 key | 共用 `~/Documents/env/paratera.env`，两三条并发没问题（S9 跑过 3 条并发）；**只通过 `--env-file` 传，不要 cat、不要 `bash -x`** |
| 代码 | 同一个 worktree。主会话会在 `apps/v2/agent/paper2code/` 改代码并提交；**已经启动的进程不受影响**（模块启动时加载），新启动的进程用新代码。要完全隔离就用 §1 的"独立 worktree"方式 |
| git | **不要**在这个 worktree 里 `git commit / checkout / stash / pull`；主会话在提交。要看代码状态只 `git log --oneline -3` |
| 判分池 | **不要往 `~/pb_submissions/<paper>/` 放东西**：主会话判分时会把池子里同一篇论文的全部提交一起判掉并归档 |
| 基线脚本 | 验证仓库的 `paperbench/scripts/run_trial.sh` 用 `/tmp/stage_b_*_<paper>.txt` 做状态文件，**同一篇论文不能两处同时跑基线**；本线的运行不受此影响 |
| 论文资产 | `~/Documents/search/paperbench/<paper>/assets/*.jpg` 可能是 LFS 指针（百来字节的文本 `version https://git-lfs…`）。`--figures off` 不读图，无所谓；`--figures on` 时指针记为 `skipped: lfs_pointer`，一张都没描述成功 intake 直接失败（09-19 起；之前是静默跳过、开图形同关图）——开图前先补成真字节。sapg、pinn 已补，其他论文见 §3 |

## 1. 准备

**方式 A（最省事）**：直接用主会话的 worktree，venv 现成：

```bash
cd ~/Documents/search/DeepEvol-Paper_repro_0916/DeepEvol1.0
PY=.venv/bin/python
$PY -c "import docker, apps.v2.agent.paper2code.driver" && echo ok     # venv 能用
git log --oneline -1                                                     # 记下你用的提交
```

**方式 B（完全隔离，主会话改代码时不受影响）**：另开一个 worktree 钉在一个提交上，装自己的 venv（约 5 分钟）：

```bash
cd ~/Documents/search/DeepEvol-Paper_repro_0916/DeepEvol1.0
git worktree add ~/Documents/search/DeepEvol-stage9 0696ed7ec          # 或更新的提交
cd ~/Documents/search/DeepEvol-stage9
uv sync && uv pip install docker                                         # uv 在 ~/Documents/search/.tools/bootstrap/bin
PY=.venv/bin/python
```

两种方式下面的命令一样。环境文件：`E=~/Documents/env`（`paratera.env` 模型 key；`aliyun.env` 第 9 步查价要用，不租机）。

## 2. 跑到第 9 步

```bash
PY=.venv/bin/python; R=~/Documents/search/paper2code-runs; E=~/Documents/env
NAME=<你的运行名>          # 例 rice-x1
PAPER=<paperbench id>      # ~/Documents/search/paperbench/ 下的目录名

$PY scripts/paper2code_canary.py init --run-dir $R/$NAME --paper-dir ~/Documents/search/paperbench/$PAPER \
  --compute aliyun --figures off --planning-fanout --repair-rounds 3
# 口径（与 S9/T5 一致）：三处模型默认 DeepSeek-V4-Flash-Vision-Exp、思考关、单次 32768、上下文 1M、规划扇出开。
# --figures on = 先把论文的图用视觉模型描述成文字再规划（是另一种输入，和基线比时要说明）。

nohup $PY scripts/paper2code_canary.py run --run-dir $R/$NAME --until compute \
  --env-file $E/paratera.env --env-file $E/aliyun.env > $R/$NAME/console.log 2>&1 &
echo $! > $R/$NAME/console.pid

$PY scripts/paper2code_canary.py status --run-dir $R/$NAME | head -60   # 各阶段状态、四道闸
```

时间与花费（sapg / pinn 实测）：references 1–4 min、index 20–55 min（参考仓库多就慢）、implement 10–20 min，整体 40–80 min；
模型 token 3–9M；机器 0。第 9 步结束后 `status.json` 里 `compute` 是 `completed`，`environment_run` 是 `pending`。

## 3. 常见卡点（都在 PITFALLS.md，这里只列到第 9 步会遇到的）

| 现象 | 处理 |
| --- | --- |
| `references` 失败：`reached the maximum number of tool call iterations (80)` | 09-19 起 40 次用尽会自动用 80 重跑一次（长论文如 pinn 124k 字符就靠这个过）；80 也用尽才失败：`DEEPCODE_REFERENCE_MAX_ITERATIONS=160 $PY scripts/paper2code_canary.py rerun --run-dir $R/$NAME --phase references --env-file … && $PY … run --run-dir $R/$NAME --until compute --env-file …` |
| `--figures on` 时 intake 失败 `--figures on but no figure was described … (lfs_pointer)` | 资产是 LFS 指针。补齐：对 `assets/asset_N.jpg` 逐个 `curl -sL https://huggingface.co/datasets/josancamon/paperbench/resolve/main/<paper>/assets/asset_N.jpg`，用指针里的 `oid sha256:` 核对后覆盖，然后 `rerun --phase intake` |
| implement 中途 `finish_reason=length` / 写文件 JSON 截断 | 已默认 `DEEPCODE_IMPLEMENT_MAX_TOKENS=32768`（VENDOR 12），一般不再出现；出现就 `rerun --phase implement` |
| `implementation_status` 门 `incomplete` / `no_tests_discovered` | 正常：生成的仓库没有测试，门按"文件都写了"放行 |
| 生成树里有 0 字节的 .py（pinn 三份都有 `src/pdes.py`） | 引擎行为，两边公平；主会话在做 T13 检查。交接时提一句 |
| 主会话正在提交代码，你的新进程行为变了 | 用方式 B；或告诉主会话你在跑什么 |

## 4. 产物在哪、怎么交回

```
$R/$NAME/
  status.json                       阶段状态 + 四道闸（preflight / plan_source / implementation_status / ownership）
  input/paper.md                    实际输入（paper.md + addendum；--figures on 时含图描述，原字节在 paper.raw.md）
  workspace/tasks/paper_<id>/generate_code/    生成的代码树 ← 这就是"第 9 步的产物"
  workspace/tasks/paper_<id>/initial_plan.txt  蓝图
  phases/<nn>_<name>.json           每阶段结果；09_compute.json 里有算力估算与默认机型
  llm/<seq>.json                    每次模型调用（无 key）
  events.jsonl  canary.log          事件流、日志
```

交回主会话时给：运行目录路径、`git log --oneline -1` 的提交、`status.json` 里四道闸是否都过、`generate_code` 的文件数。主会话拿到后直接 `run --until environment_run` 就是第 10 步（不用重跑前 9 步；目录被移动过就先 `relocate`）。
摆卷用 `submit --dest-root <你的目录>`（第 9 步完成、四道闸过的运行会被接受，记 `tree: stage9`；见 README「生成到第 9 步并摆卷」）；**不要**把 `--dest-root` 指到 `~/pb_submissions`（主会话的判分池）。

## 5. 不要做的

- 不要租机器（`run --until environment_run`）——除非主会话说可以；租了就必须 `release` 兜底。
- 不要读 `~/Documents/env/*.env` 的内容，不要 `bash -x` 跑会 `source` 它们的脚本。
- 不要改 `apps/v2/agent_engine/`（vendored 引擎）和 `apps/v2/agent/paper2code/`（本线）——发现问题记下来交回。
- 不要碰 `~/pb_submissions/`、`~/pb_submissions_archive/`。
