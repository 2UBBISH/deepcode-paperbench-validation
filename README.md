# 0919 test：Codex 桌面版 vs Claude 桌面版，PaperBench 20 篇

分支 `0919-test`。你要做的事：用 **Codex 桌面版**和 **Claude 桌面版**各把 20 篇论文"复现成代码仓库"，把产物交回统一判分。
DeepCode 那一臂在主分支跑，不在这里。评分标准（rubric）也不在这里，别去找。

## 0. 先弄懂在比什么

PaperBench 是 OpenAI 的基准：给 agent 一篇论文，让它从零写出复现代码，裁判拿一棵作者审过的评分树逐条看"这个点实现了没有"（Code-Dev 口径，不跑代码）。
DeepCode 论文说自己比 Codex 好 4 倍多，但两边用的模型不一样。我们把**模型钉成同一个**（DeepSeek 官方的 `deepseek-flash`，思考开），
给三套系统**同一份论文文字、同一份题面、都只能做快速检查、不能跑实验**，看到底差多少。所以下面每一步都在保证"三边拿到的东西一样"，请别自己加东西。

## 1. 准备（只做一次，10 分钟）

1. **装两个桌面 app**：Codex 桌面版（ChatGPT app 里的 Codex）、Claude 桌面版（Code 标签）。
2. **让两个 app 都走 DeepSeek 官方**。用 cc-switch 各建一个档（真 key 填在档里，DeepSeek 平台申请）：
   - Codex：`base_url = https://api.deepseek.com/v1`，`wire_api = "responses"`，`model = "deepseek-flash"`，再加一行 `model_context_window = 1000000`（三边上下文窗口都按 1M 对齐，不然各自压缩历史的时机不一样）
   - Claude 桌面：`ANTHROPIC_BASE_URL = https://api.deepseek.com/anthropic`，再把 `ANTHROPIC_MODEL`、`ANTHROPIC_DEFAULT_HAIKU_MODEL`、`ANTHROPIC_DEFAULT_SONNET_MODEL`、`ANTHROPIC_DEFAULT_OPUS_MODEL`、`CLAUDE_CODE_SUBAGENT_MODEL` 五个都填 `deepseek-flash`（不填的话 app 会拿 Claude 的模型名去请求，DeepSeek 不认）
   切完档**重启 app**。思考模式不用管：DeepSeek 缺省就是开的，我们就用开的。
3. **清掉私人指令**：`~/.codex/AGENTS.md` 要是空的（或不存在），`~/.claude/CLAUDE.md` 不能存在。这两个文件会把你平时的习惯偷偷喂给 agent。
4. **关掉 app 里的附加能力**：Codex 的 browser / chrome / computer-use 插件关；Claude 桌面的 skills / MCP / Chrome 集成关。网页搜索可以留（官方题面允许上网查资料）。
5. 克隆并初始化：
   ```bash
   git clone -b 0919-test git@github.com:2UBBISH/deepcode-paperbench-validation.git && cd deepcode-paperbench-validation
   bash setup.sh      # 检查 20 篇论文、把每篇的官方实现仓库在 git 里封掉（防止 agent 克隆答案）、建 results/ 和 work/
   ```

## 2. 一篇论文、一个 app 怎么跑（重复 20 × 2 次）

以 Codex + `sapg` 为例，Claude 把 `codex` 换成 `claude` 即可。

**第 1 步：准备工作目录和题面**
```bash
bash desktop/desktop_prep.sh codex sapg
```
它会建 `work/sapg-codex-desktop/`，里面是：
```
paper/paper.md  paper/addendum.md  paper/blacklist.txt     ← 论文文字、作者补充说明、禁止访问的仓库（agent 能看到的全部材料）
submission/                                               ← 空的 git 仓库，agent 往这里写代码
PROMPT.txt                                                ← 题面（已经复制到剪贴板）
CONTINUE.txt                                              ← 它停下来时你回的那句话
```
题面 = PaperBench 官方指令原文 + 官方附注（含官方的"你有 3 小时"那句），路径已经换成你机器上的绝对路径。**不要改一个字。**

**第 2 步：在 app 里跑**
1. 打开 `work/sapg-codex-desktop/` 这个文件夹作为项目。
2. 审批模式：Codex 选 **"Agent"**（不要选"完全访问"，否则规则文件弹不出审批）；Claude 用**默认模式**（不要选 bypass / 跳过权限，否则 Bash 不会问你）。
3. 新建对话，把剪贴板里的题面粘进去，发送。前面不加"你好"，后面不加"开始吧"。
4. 然后盯着审批。它每次要跑 `python` / `pytest` / `pip` / `uv` / `bash` / `curl` 都会弹出来问你（读文件、改文件不会问）。**你只做一件事：批快速检查，拒实验。**
   - **批**：语法 / 导入检查（`python -m py_compile`、`python -c "import …"`）、单元测试和冒烟测试（`pytest`、小数据几秒钟跑完的脚本）、装依赖（`pip install` / `uv sync`）、`--help`。
   - **拒**：训练（`train`）、评估（`eval` / `benchmark`）、下载数据集或模型权重（`wget` / `curl` / `huggingface`）、任何一看就要跑几分钟以上的。拿不准就拒。
   - 拒了之后它自己会继续写；它要是问"能不能运行"，按第 5 步回续跑语，不要解释。收尾脚本会把实际跑过的命令和耗时列出来，单条超过 5 分钟或合计超过 30 分钟这篇作废。
   - 题面里告诉它有 **3 小时**，这是官方题面的说法，**不是硬上限**：它自己觉得核心贡献复现完了就会停，到了 3 小时还在写就让它继续写，**不要手动停**；收尾脚本会记实际用时。
5. **它停下来了怎么办**：
   - 它说"做完了"，并且 `submission/` 里已经 `git commit` 了 → 去第 3 步。
   - 它问你问题 / 要你确认 / 说完了但没 commit → 把 `CONTINUE.txt` 里那句话原样发给它（最多 5 次），然后在 `work/sapg-codex-desktop/interactions.log` 里记一行（几点、它问了什么）。**不要回答它的问题，不要给任何提示。**
   - 5 次之后还没完 → 也去第 3 步，有什么交什么。

**第 3 步：收尾**
```bash
bash desktop/desktop_finish.sh codex sapg
```
它会：从 app 自己的会话日志里核对这一轮每次请求用的模型是不是 `deepseek-flash`、思考是不是开着、实际跑过哪些命令各用了多久（写进 `AUDIT.txt`，最后一行是 `CALIBER_OK`，或者跑过命令时的 `CALIBER_REVIEW`——owner 会看那份清单；`CALIBER_BROKEN` 是模型不对 / 思考没开 / 执行超时）；
检查代码里有没有抄黑名单仓库；把 `submission/` 和所有记录拷到 `results/sapg/codex/`。
然后打开 `results/sapg/codex/RUN_NOTES.md`，把三行 "fill in" 填上（app 版本、留着的插件、审批模式）。

**换下一篇**：`bash desktop/desktop_prep.sh codex pinn` …… 20 篇的 id 就是 `data/papers/` 下的目录名。两个 app 可以同时各跑一篇。

## 3. 交回

```bash
tar czf results_<你的名字>_$(date +%m%d).tgz results/
```
把压缩包给 owner。`work/` 不用交。目录长这样：
```
results/<paper>/codex/   submission/  AUDIT.txt  RUN_NOTES.md  session_logs/  interactions.log  PROMPT.txt
results/<paper>/claude/  同上
```
`AUDIT.txt` 不是 `CALIBER_OK` 的（比如模型不对、思考没开、找不到会话日志）也照交，但在 RUN_NOTES 里写一句原因。

## 4. 不要做的（做了这篇就作废）

- 不给 agent 论文 PDF、图、官方代码、任何提示；题面之外不说话（续跑语除外）。
- 不批准实验（训练 / 评估 / 下载数据）。`desktop_finish.sh` 会从 app 的会话日志列出实际跑过的命令和耗时，单条 > 5 分钟或合计 > 30 分钟这篇作废（`AUDIT.txt` 会写明）。
- 不改 `submission/` 里的代码，不帮它装环境，不帮它 commit（收尾脚本会替它 commit 未提交的改动并记录）。
- 不动 `desktop/`、`instructions/`、`data/` 里的东西。
- 同一篇同一个 app 只跑一次；要重跑先删 `results/<paper>/<app>/` 和 `work/<paper>-<app>-desktop/`，并在 RUN_NOTES 里说明。

## 5. 已知情况

- `robust-clip` 的官方 `paper.md` 缺第 2、3 章（方法），这是 PaperBench 数据本身的问题；三边拿的都是这份，照跑，分数会普遍低。
- **只准快速检查，不准跑实验**（09-20 晚定；白天曾定为完全不准执行）：Code-Dev 判分本来就不执行；DeepCode 的写码 agent 只有 `write_file` / `search_code_references` / `read_paper` 三个工具、没有解释器，桌面 agent 若能跑几小时实验来验证就不对等（09-19 的 fre / rice Codex 运行各跑了 177 分钟实验，作废重跑）；但语法 / 导入 / 单测这类秒级检查算基本运行，允许。落实靠人逐条审批：Codex 用 `~/.codex/rules/paperbench-exec.rules`（`desktop_prep.sh` 自动装，把 python / pip / uv / conda / node / bash / sh / make / docker / wget / curl 升级成审批弹窗），Claude 用工作目录 `.claude/settings.json` 把 Bash 设成 ask；题面附注的 Execution 一条把同样的规则告诉 agent；`audit_desktop.py` 列出实际跑过的命令和耗时。
- "3 小时"是用 PaperBench 官方的 `time_limit_template` 写进题面的（官方跑法给 12 小时；我们 20 篇 × 2 个 app 给 3 小时）。官方的意思是"预期你用满这么多时间，除非你已经把核心贡献都复现完了"——所以什么时候停由 agent 自己判断，人不按表停它；`desktop_finish.sh` 只记录实际用时。不能跑实验之后 agent 以写为主，通常远不到 3 小时就自己说做完了。
- Claude 桌面版的上下文窗口：Claude Code 靠模型名后缀 `[1m]` 开 1M 窗口，但 DeepSeek 会不会认 `deepseek-flash[1m]` 这个名字还没验；owner 验完会在这里写明填法，验之前 Claude 臂先按 `deepseek-flash` 跑，RUN_NOTES 里记一句。
- 09-20 第一对分数（fre）：本线 0.876 vs 09-19 那次"跑了实验"的 Codex 0.785（那次作废，只作参考）；正式的 Codex / Claude 分数等本批产物。
- 为什么用桌面版、为什么同模型、依据在哪：[`docs/CODEDEV-ARMS.md`](docs/CODEDEV-ARMS.md)（主分支）。想用 CLI 非交互跑同样两臂（要一层代理来关思考）：`cli-reference/`，不是本批口径。

## 6. 目录

```
setup.sh                      一次性初始化
data/papers/<20 篇>/          paper.md · addendum.md · blacklist.txt（PaperBench 官方数据集的子集）
instructions/                 code_only_instructions.txt（官方题面，逐字）
desktop/                      desktop_prep.sh · desktop_finish.sh · audit_desktop.py · render_prompt.sh · additional_notes.txt · continue_message.txt · paperbench-exec.rules
cli-reference/                run_bare.sh · paratera_proxy.py · audit.py · README.md（CLI + 代理版，参考）
results/  work/               交回的产物 · 工作目录（都不入库）
```
