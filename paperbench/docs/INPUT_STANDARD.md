# bam 三方对照 · 输入标准（按 PaperBench 数据集口径）

适用：`01_deepevol`（我们的线）/ `02_deepcode`（DeepCode 脚手架）/ `03_bare`（裸跑）。
目的：三方拿到**同一份论文材料、同一档运行条件**，差异只剩"系统本身"。凡不符合本标准的产物不判分。

## 0 依据（不是我们定的，是照抄基准）

1. PaperBench 给 agent 的目录，`frontier-evals/project/paperbench/paperbench/nano/task.py:108-128`：
   `/home/paper/paper.pdf`、`paper.md`、`addendum.md`、`blacklist.txt`、`assets/*`，外加 `/home/instructions.txt`。
   **不给 `rubric.json`，不给 `config.yaml`**（两者留在数据集目录里只供裁判用）。
2. Code-Dev 指令 = `paperbench/instructions/code_only_instructions.txt`（本目录 `00_input/code_only_instructions_official.txt` 与之逐字相同）。
   指令本身不含时限数字、不含黑名单 URL；判分规则（部分得分、重要性加权、看趋势、"will not be executed"）它**是**公开讲了的，但只陈述规则，不给"所以你该优先什么"的指令。
3. DeepCode 论文 §4.1（arXiv 2512.07921）：DeepCode "accepts the target paper in both PDF and Markdown formats, along with any supplementary addenda, as primary inputs"；"a source code blacklist is enforced during execution … during web browsing"；沙箱有文件系统、shell、互联网。
   **对 Cursor / Claude Code / Codex 用了什么提示词、给没给 addendum、给没给 rubric，论文一字未提。** 所以裸跑没有"论文口径"可抄，只能用基准自己的指令。

## 1 三层标准

### 1.1 论文材料层 —— 三方字节级相同

| 文件 | 来源 | 我们的线 | DeepCode | 裸跑 |
|---|---|---|---|---|
| paper.pdf | `data/papers/bam/` sha256 `86018f2f…` | 论文包成员 | 不吃（只吃 md） | 目录里 |
| paper.md | sha256 `96cc2c13…` | 论文包成员（Stage 1 正文） | `inputs/paper.md` 前半 | 目录里 |
| addendum.md | sha256 `1def9d46…` | 论文包 → 每阶段可见的 addendum 语句 | 并进 `inputs/paper.md` 末尾 `# Addendum` 节 | 目录里 |
| blacklist.txt | sha256 `1307c275…` | 论文包 → Stage 4/6 拒绝导入 | `DEEPCODE_URL_DENYLIST` + git insteadOf | 目录里（见 1.3） |
| assets/（15 张 jpg） | 数据集自带 | 论文包成员，模型不看图 | 不给 | 目录里，agent 可看 |
| rubric.json / config.yaml | 数据集自带 | **不给** | **不给** | **不给** |

规则：
- 材料只能从 `00_input/paper/` 拷贝；任何一方的材料目录里出现 rubric.json / config.yaml 即作废。
- addendum 必须到达模型：我们的线走论文包，DeepCode 走并稿，裸跑走目录 + 指令点名。三种形态都记在案，不再改。
- assets 是已知不对称（只有裸跑能看图），方向是**偏帮裸跑**，对"我们的线更好"的结论是保守的，接受并记录。

### 1.2 任务指令层 —— 裸跑用官方原文，另两方用各自内建提示

- **裸跑**：`03_bare/workspace/PROMPT.txt` = `code_only_instructions_official.txt` 原文 **+ 固定后缀 `00_input/bare_prompt_suffix.txt`**（两句：不要停下来问/等，做到做不动为止再提交；产物只放 submission/），再把 `/home/paper`、`/home/submission` 两个路径替换成本机路径。后缀是 owner 2026-09-14 决定加的：另两条线的 harness 自己会一直跑到底，Codex 桌面版没有 harness 续跑，这两句补的是 harness 条件（PaperBench 自己的 IterativeAgent 也这么做），不含论文/判分信息。后缀文字冻结，不再改。校验命令见 §3。
  整段一次喂入，不加任何前后文，不加系统提示，不加 CLAUDE.md / AGENTS.md 之类的项目文件（起会话的目录里只能有 `paper/`、`submission/`、`PROMPT.txt`、`paratera_proxy.py`）。
- **DeepCode**：脚手架自带提示词，不改；`run_trial.sh` 里的环境变量只动 max_tokens / 重试 / 超时，不动生成逻辑（脚本注释已逐条说明）。
- **我们的线**：自带阶段提示，不改；不加针对 bam 的任何提示或参数。
- 下列内容**禁止**出现在任何一方的额外输入里（这就是 SNSE 那份倾斜提示词多给的东西，`snse-threeway/03_bare/tilted_not_graded/PROMPT_codedev_tilted.txt`）：
  1. 指定阅读顺序 / 点名"先读 addendum"；
  2. 把黑名单里的 URL 抄进提示；
  3. 除固定后缀那两句以外的流程指令（倾斜版还有"必须全自主"、"决定记进 README"等）；
  4. 工程建议（"不要钉版本"等）；
  5. 在官方陈述之外追加的**行动指令**：官方只说"代码不会被执行、按正确实现判分"（这是公开规则，三方都知道），倾斜版接了一句 "so prioritise … implementation over runnability"，替 agent 做了取舍；
  6. 要求结束时汇报文件数 / 行数 / 提交数。

### 1.3 运行条件层

| 条件 | 标准 | 三方怎么落实 |
|---|---|---|
| 底座 | DeepSeek-V4-Pro @ Paratera | 三方同一 key、同一 base URL |
| 思考 | **开**（Paratera V4-Pro 默认；运行层关不掉，且 DeepCode 论文三方全用 -thinking） | 我们的线 `init --thinking`（不带就会发 disabled，事件会报 `thinking_not_disabled`，口径就乱了）；DeepCode **不设** `DEEPCODE_THINKING`（上游原样）；裸跑走 `paratera_proxy.py` 纯直通，只记日志不改请求体。三方回包 `usage.reasoning_tokens` 须 >0 |
| 上网 | 允许（官方指令允许） | 三方都可搜；参考仓库自己找 |
| 黑名单 | 论文自己的仓库 `modichirag/GSM-VI` 不得访问（**保留**：paper.md:36、:278 两处写明 "A Python implementation of BaM is available at github.com/modichirag/GSM-VI/"，仓库 README 含 ICML 2024 BaM 实现，`examples/example_bam.py`） | 全局 `git insteadOf` 三方共用（同一台机）；DeepCode 另有 URL 拦截；我们的线 Stage 4/6 拒绝；裸跑靠指令 + 事后审计（§3） |
| 人工输入 | **零信息量** | 我们的线：`run` 不带 `--ask`（Stage 7 四段全部默认确认、Stage 8/10 默认）；DeepCode 无交互；裸跑：agent 停下来问就回固定续跑语（§2 ①），逐次记 `interactions.log` |
| 时限 | 指令里不写数字（官方原文也没有） | DeepCode 14 h 硬顶；我们的线 `--run-hours 36`；裸跑单会话到它自己停 |
| 执行环境 | Code-Dev 不执行提交，环境差异不进分 | 我们的线阿里云 Linux；DeepCode / 裸跑 本机 macOS。记录，不对齐 |
| 提交形态 | git 仓库 + README.md，`git clean -fd` 后仍完整 | 三方一致，摆卷到 `~/pb_submissions/bam/<臂名>/` |
| 裁判 | PaperBench Code-Dev 官方判分器，裁判模型 DeepSeek-V4-Pro 恒定 | `run_grade.sh`；论文用 o3-mini，我们换了裁判但三方同一个 |
| 轮次 | 每方 1 轮（成本） | DeepCode 论文未说轮数；单轮噪声 σ≈0.1（rice 实测），解释结果时带上 |

## 2 已拍板（2026-09-14）

① **裸跑 agent 停下来问问题**：只回一条固定、无信息量的续跑语，原文：

```
Continue; no further input will be provided.
```
次数不限，每次在 `03_bare/workspace/interactions.log` 记一行（时间、第几次、agent 问了什么的摘要）。不回答任何实质问题。后缀里已写"不要停"，这条是它仍然停下时的兜底。
② **裸跑 = Codex**（OpenAI 兼容格式，走 `paratera_proxy.py` 直通记录），不是 Claude Code。
③ **思考开**，三方一致；任何一方的 `reasoning_tokens=0` 视为口径破坏。
④ **之前所有三方对比作废**（SNSE 对标前/后、bam 之前的任何设想），从本标准起重新计数。
⑤ 黑名单保留（官方仓库存在，见 1.3）。

### Codex 接法（本机实况：装的是 Codex 桌面版，没有 `codex` CLI）

`~/.codex/config.toml` 已有一个指向 Paratera 的 provider `custom`（`wire_api = "responses"`，能用），**只改两处，其余不动**：

```toml
model = "DeepSeek-V4-Pro"                      # 原来是 DeepSeek-V4-Pro-0813 —— Paratera 上这是两个不同 id，必须和另两方一样用 DeepSeek-V4-Pro

[model_providers.custom]
base_url = "http://127.0.0.1:8787"             # 原来是 https://llmapi.paratera.com；代理原路转发，wire_api/token 都不用改
```

- `model_reasoning_effort = "medium"` 是 Codex 自己的默认，会以 `reasoning.effort` 发出去；另两方不发这个字段。保留（它是"Codex 这个系统"的一部分，论文里 Codex 也是带 effort 跑的），代理会把它记进 `thinking_fields` 留证。
- `~/.codex/AGENTS.md` 现在是 0 字节，起跑前再确认一次；`03_bare/workspace/` 里不得有 AGENTS.md。
- 桌面版开着 browser / chrome / computer-use / pdf 等插件，比论文用的 codex-cli 0.47 多出一截工具。**建议本轮关掉 browser / chrome / computer-use**（上网搜资料靠 shell 里的 curl/git 一样能做，官方指令允许），至少把当时开着的插件清单抄进 `03_bare/workspace/RUN_NOTES.md`。
- 在桌面版里把工作目录选成 `/Users/apple/Documents/env/bam-threeway/03_bare/workspace`，审批模式选全自动（论文 §4.1 的 "auto approval mode"）。

**给 Codex 的输入 = `03_bare/workspace/PROMPT.txt` 全文（官方原文 + 固定后缀，英文），一个字不加、不减、不翻译。** 复制：`cat /Users/apple/Documents/env/bam-threeway/03_bare/workspace/PROMPT.txt | pbcopy`。
不要在前面加"你好/请开始/这是任务"，不要在后面加"开始吧"，不要把路径改成相对路径，不要先让它 `ls` 一下。

## 3 起跑前核验（三条命令，全过才起）

```bash
# a) 裸跑提示词 = 官方原文 + 固定后缀 + 两处路径替换，diff 必须为空
cd /Users/apple/Documents/env/bam-threeway && cat 00_input/code_only_instructions_official.txt 00_input/bare_prompt_suffix.txt | sed -e 's#/home/paper#/Users/apple/Documents/env/bam-threeway/03_bare/workspace/paper#g' -e 's#/home/submission#/Users/apple/Documents/env/bam-threeway/03_bare/workspace/submission#g' | diff - 03_bare/workspace/PROMPT.txt && echo PROMPT_OK
```
```bash
# b) 材料四件套三处 sha256 一致，且裸跑目录里没有 rubric/config
cd /Users/apple/Documents/env/bam-threeway && for f in paper.pdf paper.md addendum.md blacklist.txt; do shasum -a 256 00_input/paper/$f 03_bare/workspace/paper/$f /Users/apple/Documents/env/paperbench-judge/validation/frontier-evals/project/paperbench/data/papers/bam/$f | cut -c1-12 | uniq -c; done; ls 03_bare/workspace/paper | grep -E 'rubric|config' && echo LEAK || echo CLEAN
```
```bash
# c) 黑名单 git 封锁在位
git config --global --get-regexp insteadof | grep -i GSM-VI
```

跑完裸跑再审计两项：`proxy_requests.log` 每行 `model=DeepSeek-V4-Pro`，且至少一条回包记录 `reasoning_tokens>0`（思考确实开着）；`grep -ri "GSM-VI\|modichirag" 03_bare/workspace/submission` 只允许出现在引用/README 文字里，不允许有克隆或复制的痕迹。
