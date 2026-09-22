# 历史结果总表（2026-08-25 → 2026-09-15）

所有数字一处查全；原始 `grade.json`、提交产物、日志与旧分析文档已从仓库移到本地
`~/Documents/env/paperbench-judge/archive/`（不再入库，见文末 §9）。**每张表都标了口径与是否作废**，
引用前先看 §0。分数一律是 PaperBench Code-Dev（`code_only=True`，不执行代码），裁判模型名 + serving 随分数一起给。

## 0. 作废规则与两次全体作废

作废 = 不再作为对比结论引用，只作可追溯记录。

| 作废原因 | 波及 | 说明 |
| --- | --- | --- |
| **输入有偏**（2026-09-14 定） | 09-14 之前全部对比 | 裸跑提示词在官方指令外多给了六类偏帮；DeepCode 臂没拿到 `addendum.md`；"思考关"从未成立（Paratera 忽略 `enable_thinking:false`，只认 `thinking:{type:disabled}`）；裸跑壳是 Claude Code 而非论文里差距最大的 Codex |
| **裁判选文件根目录 bug**（2026-09-15 修） | 09-15 修前全部分数 | 裁判让模型列"最相关文件"，约五分之一回答省掉树根 `submission/`，上游原样拼路径一个读不到，整份摆卷逐叶被判"没实现"。修后 bam 三份分别 0.7644→0.9073、0.6659→0.8367、0.6530→0.7343 |
| 单轮作废 | 见 §6 | 撞墙钟 / stall 熔断 / 假计划 / 余额耗尽 / 提示词含评分元知识 |

仍然成立、与输入无关的：§7 JudgeEval 校准、§5 裁判 serving 依赖、§8 四处静默降级。

## 1. bam 三方对照（2026-09-15，当前唯一有效的对比数字）

> owner 09-21：这一节的 DeepCode > Codex（0.837 vs 0.734）是**单次样本，不是稳定领先**；§1.3 fre 的 Codex 那份（0920_codex-exec）跑时**已带蓝图**，也不是裸跑。两个点都不能当"DeepCode 领先"的先验；线 vs Codex vs Claude 的有效对比从 fre / rice 三臂在同一裁判口径下齐了才开始。

口径：`docs/INPUT_STANDARD.md`（材料五样字节级相同、裸跑用官方指令原文 + 冻结后缀、**思考开**、裁判恒定
DeepSeek-V4-Pro @ Paratera、`PB_JUDGE_CONCURRENCY=20`、裁判已修选文件根目录 bug）。

| 臂 | 系统 | 状态 | 总分 |
| --- | --- | --- | --- |
| 03_bare.gpt5-codex-high | Codex 桌面版 + gpt-5.5 high（底座不同，不进主表） | 完成，12.5 min | **0.9073** |
| 02_deepcode | DeepCode（e0767d0 + 旧补丁，`PAPER=bam TRIAL=trial1`）+ V4-Pro | 完成，3 h 13 min | **0.8367** |
| 03_bare | Codex 桌面版 + V4-Pro | 完成，26 min | **0.7343** |
| 01_deepevol | DeepEvol 复现线（旧架构）+ V4-Pro | 停跑，无分：Stage 9 五个 attempt 未过，owner 09-15 叫停 | — |

同底座（V4-Pro）：DeepCode 0.8367 > 裸跑 Codex 0.7343，差 0.10，来自 §5.2/§5.3 两个实验节；核心算法节 DeepCode 反而低（0.721 vs 0.821）。
三臂全丢的 38 片叶子几乎都是"ADVI 学习率网格搜索"。修裁判前的三份分（0.7644 / 0.6659 / 0.6530）与并发 16 对照（0.6317）作废。
论文 Table 1 bam 列（Sonnet 4.5-think，o3-mini 裁判）：Codex 0.1937 / Claude Code 0.3829 / Cursor 0.3779 / DeepCode 0.8530——只看相对关系：Codex 0.19 的差距在同底座同裁判下不复现。

叶子级明细、各臂 RUN_NOTES 与原始 grade.json：`archive/deepcode_test/bam/` 与 owner 本机 `~/Documents/env/bam-threeway/`。

**2026-09-17 起的新一批**：DeepEvol 的 Paper2Code 线（原装 DeepCode 引擎嵌入）vs 本仓库的基线运行，口径改为 **DeepSeek-V4-Flash、思考关**（`reasoning_tokens == 0`），第一篇 `sapg`。两边并排的过程数字在 DeepEvol 仓库 `apps/v2/agent/paper2code/HANDOFF.md`「Second batch」，判分后再回填到这里。

| 2026-09-17 sapg | 基线运行 `trial1`（本仓库，DeepCode 21ebc57f + 补丁） | DeepEvol 线 C9 运行 |
| --- | --- | --- |
| 状态 | 完成，未判分；摆卷 `~/pb_submissions/sapg/trial1/`（30 文件） | 完成，未判分 |
| 计划 / 参考 / 克隆 / 索引 | 9,733 字符 `generated` 分段 · 5 URL · 5 仓库 · 5 索引（2,103 s） | 9,672 字符 `generated` 分段 · 4 URL · 4 仓库 · 4 索引（1,711 s） |
| 写码 | 29/29 文件，518 s，24 py / 7,319 行；上游未发现测试命令、未执行 | 26/26 文件，599 s，22 py / 6,349 行；远端 compileall 通过、入口冒烟失败（嵌套包） |
| 调用 | 413 次，`reasoning_tokens` 0，无 `length` 截断，每次 `max_tokens` **8192**（目录钳制，下一轮 32768） | 346 次，`reasoning_tokens` 0，每次 32768 |
| 备注 | 后置闸门因运行中改脚本而手动补跑（同一段代码），全过；未判分 | 阿里云 ecs.c7.xlarge，两次 environment_run；未判分 |


### 1.1 sapg 第一对分数（2026-09-17，裁判 DeepSeek-V4-Flash，解析器 V4-Pro，code_only，77 个 Code-Dev 叶）

口径：两边 DeepSeek-V4-Flash、思考关（全程 `reasoning_tokens` 0）、每次调用 32768、同一份 `paper.md` + addendum、同黑名单；本线跑在阿里云 ecs.c7.xlarge，基线在本机。

| 提交 | 总分 | SAPG 实现 (w1) | 实验设置 (w1) | Fig.2 / 5 / 7 / 8 | 产物 | 过程 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 `trial2`（DeepCode 21ebc57f + 补丁） | **0.3374** | 0.413 | 0.611 | 0 / 0 / 0 / 1 | 19 py / 5,143 行 | 96 min，702 次调用，IsaacGymEnvs 预筛回退全量 |
| DeepEvol 线 `deepevol_2`（run 09170541607c） | **0.3180** | 0.382 | 0.526 | 0 / 0 / 0 / 1 | 21 py / 6,279 行 | 76 min，544 次调用，入口冒烟通过 |

叶级：77 个 Code-Dev 叶里两边同过 22，只有本线过 9，只有基线过 11。差 0.019，远小于历史组内摆动（0.09–0.19），**不能说谁优**；两边一起丢的是三张结果图的整条子树（Figure 2/5/7 全 0）、"五个种子"、"六个策略"、"0.001 熵系数"，本线另丢了"熵系数二选一"和 hard 任务的一半。
判分：两份共 4 分钟；Flash 每份约 3.1–3.2M 入 / 0.28–0.30M 出 token，Pro 解析器每份 0.1M 入。原始 `grade.json` 在本地 `archive/deepcode_test/sapg/grades/`。
同批还有一轮基线 `trial1`（`max_tokens` 被钳在 8192，未判，产物在 `~/pb_submissions_archive/sapg/trial1_maxtok8192/`）和本线 C9 运行（未判）。

### 1.2 sapg 成对重跑 S9（2026-09-18，裁判 DeepSeek-V4-Flash，解析器 V4-Pro，code_only，77 个 Code-Dev 叶）

口径：两边 **DeepSeek-V4-Flash-Vision-Exp**、思考关（全程 `reasoning_tokens` 0）、每次调用 32768、同一份 `paper.md` + addendum（sha `04790c3f…`）、同黑名单、
**同两处规划补丁都开**（`DEEPCODE_PLANNING_FANOUT=1`、`DEEPCODE_PLANNER_CONTEXT_WINDOW=1000000`，VENDOR 11）、写码单次输出上限 32768（VENDOR 12）；
本线第 10 步在阿里云 T4（`ecs.gn6i-c8g1.2xlarge`）上跑 搭建环境 → 试跑 → 修复 ≤3 轮，基线在本机（上游自带的本地验证）。

| 提交 | 总分 | SAPG 实现 (w1) | 实验设置 (w1) | Fig.2 / 5 / 7 / 8 | 产物 | 过程 |
| --- | --- | --- | --- | --- | --- | --- |
| 基线 `vexp2`（DeepCode 21ebc57f + 16 文件补丁，开关开） | **0.6910** | 0.979 | 0.917 | 0 / 0.25 / 1 / 1 | 27 py / 7,490 行 | 40 min，308 次调用 |
| 本线 `s9off_pre`（run 091717329303，图描述关，修复前快照） | **0.6677** | 0.910 | 0.846 | 0 / 0.25 / 1 / 1 | 23 py / 9,372 行 | 全程 714 次调用 / 8.9M token（含第 10 步） |
| 本线 `s9off_post`（同一 run，3 轮修复后） | **0.6365** | 0.969 | 0.850 | 0 / 0 / 1 / 1 | 23 py | 第 10 步：环境 1 轮 + 修复 3 轮（每轮 40 调用 / 15 探针），G2 未过、accept |
| 本线 `s9on_post`（run 091717345069，**12 张图由 Vision-Exp 描述后插回原位**，3 轮修复后） | **0.6709** | 0.896 | 0.630 | 0 / 0.5 / 1 / 1 | 21 py / 8,853 行 | 全程 811 次调用 / 9.1M token；G2 未过、accept |
| 基线 `vexp1`（同 `vexp2`，但写码单次上限为上游默认 8192；日志无截断）— **T1 补判 09-18 10:28** | **0.7156** | 0.816 | 0.728 | 0 / 0.75 / 1 / 1 | 25 py / 8,098 行 | 40 min；与 `vexp2` 是"什么都没变"的两次运行 |
| 本线 `s9on_pre`（run 091717345069，图描述开，修复前快照）— **T1 补判 09-18 10:28** | **0.6414** | 0.951 | 0.647 | 0 / 0.25 / 1 / 1 | 21 py / 8,807 行 | 与 `s9off_pre` 唯一变量是 12 张图的描述 |

叶级（77 叶）：基线 vs 本线关-修复前 同过 45、只基线 12、只本线 8；关：修复前 vs 修复后 同过 48、只修复前 5、只修复后 9；开：修复前 vs 修复后 同过 46、只修复前 6、只修复后 5；
关-修复前 vs 开-修复前 同过 38、只关 15、只开 14；关-修复后 vs 开-修复后 同过 44、只关 13、只开 7；基线 `vexp1` vs `vexp2` 同过 41、只 vexp1 6、只 vexp2 16。
读法（T1 补判后改写）：① 六份都在 0.64–0.72，比 09-17 那对（0.34 / 0.32）高一倍——变量是规划补丁（扇出 + 全文 + 附录）与 Vision-Exp，Figure 7 / 8 子树从 0 变 1；
② **运行间噪声标尺 ≈ 0.025**（`vexp1` 0.7156 vs `vexp2` 0.6910，同一代码同一输入跑两次；叶级各自独有 6 / 16，`vexp1` 反而少过 10 个叶，分差全来自 Fig.5 子树 0.75 vs 0.25 的权重）；
③ 修复 3 轮：关 −0.031（0.668 → 0.637），开 **+0.030**（0.641 → 0.671）——方向相反、幅度相同，就是噪声；09-18 早晨"修复轮扣分、rubric 把重写计为退步"的读法不成立；
④ 图描述开 vs 关：修复前 −0.026（0.668 vs 0.641，叶级 38 同 / 15 vs 14 各自独有），修复后 +0.034——也是噪声，sapg 上看不出图描述的收益（四张图子树本就是 0 / 0.25 / 1 / 1，可动的只有 Fig.5）；
⑤ 结论：基线 vs 本线、修复前后、图开关，所有差都在 ±0.03 内；**sapg 单篇单次分不出任何东西**，判据类的改进要用机械指标（G2 是否通过、diff 规模）验收，分数留给多篇多次积样本。
判分：四份 7 分钟、T1 两份 5 分钟；Flash 每份 3.2–4.4M 入 / 0.28–0.34M 出，Pro 解析器每份 0.1M 入。`grade.json` 在 `runs/sapg/grades/`（不入库）；六份提交都已归档到 `~/pb_submissions_archive/sapg/`，`~/pb_submissions/sapg/` 为空。

### 1.3 pinn（"Challenges in Training PINNs: A Loss Landscape Perspective"）成对首跑 T5（2026-09-18 晚，裁判 DeepSeek-V4-Flash，解析器 V4-Pro，code_only，126 个 Code-Dev 叶 / 1963 叶）

口径同 §1.2（两边 Vision-Exp、思考关、32768、规划补丁开、同一份 paper.md + addendum、黑名单 `pratikrathore8/opt_for_pinns`）；论文 124k 字符、15 张图（LFS 指针从 HF 补齐）。
本线两份是**修复前快照**（round-0，第 10 步没跑完就按 owner 的话先判了）；基线在本机。

| 提交 | 总分 | 实验设置 | Fig.3 | NNCG | Fig.4/5 | Table 3 | 产物 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 基线 `vexp1` | **0.6700** | 0.58 | 0.78 | 1.00 | 0.16 | 1.00 | 28 文件；290 次调用 |
| 本线 `pinn_off_pre`（图描述关） | **0.7083** | 0.75 | 0.78 | 1.00 | 0.16 | 1.00 | 32 文件；references 第一次 40 次迭代用尽，80 次重跑 |
| 本线 `pinn_on_pre`（图描述开，15 张图） | **0.6696** | 0.72 | 0.89 | 0.75 | 0.16 | 1.00 | 25 文件 |

叶级：基线 vs 图关 同过 72、只基线 9、只本线 17；图关 vs 图开 同过 74、各自独有 15 / 13；基线 vs 图开 同过 68、13 / 19。
读法：① 三份都在 0.67–0.71，与 sapg 同一量级；② 本线图关比基线 +0.038，略大于 sapg 的噪声标尺 0.025，但 pinn 自己的运行间噪声未量（各一次），只能说"不输"；③ 图开 vs 图关 −0.039，和 sapg 一样看不出图描述的收益；④ Fig.4/5 子树两边都 0.16——画图/实验脚本这一类缺失是两篇论文的共性；
⑤ **三份树里 `opt_for_pinns/src/pdes.py` 都是 0 字节**（基线也是；PDE 定义写到了顶层 `src/pdes.py`）——DeepCode 写码循环的系统性行为，两边公平，但值得本线加一道"零字节计划文件"检查（DeepEvol PLAN-3 T13）。
第 10 步（本线，CPU 档，两段式，无 torch 预装镜像）：`off` 环境轮 7.8 min → 试跑 `ImportError get_pde`（就是那个空文件）→ 修复 ① 正确诊断、改 2 文件 +18−7 → 同签名 → 停；`on` 环境轮 12 min → 试跑 `ImportError PDE` → 修复 ① 后传代码到机器时 `GitDaemonError: connection lost`（本机网络）→ 停。两台机器均已释放，修复后的树未判。
另：T4 预装 torch 的镜像让 RSA 伪证器把 pinn 的 G0 判为"裸镜像即过"而拒绝判据两次，已换回无 torch 镜像（DeepEvol PITFALLS §D）。
判分：三份 9.5 分钟；`grade.json` 在 `runs/pinn/grades/`，提交归档在 `~/pb_submissions_archive/pinn/`。

### 1.3 fre 第一对分数（2026-09-20，裁判 DeepSeek-V4-Flash，解析器 V4-Pro，code_only，306 叶）

口径：两边 `deepseek-flash` @ api.deepseek.com **思考开**、同一份 `paper.md` + addendum + 黑名单（md-only，无 PDF/图）。本线 = DeepEvol Paper2Code 线
`fre-t14`（T14 论文保真：蓝图照抄公式 + `Source: §x.y`、写码期 `read_paper` 回读 181 次；`PAPER2CODE_IMPLEMENT_VERIFY` 关，**不执行任何代码**）到第 9 步的树。
Codex 那份是 09-19 的 Codex 桌面版运行，**跑了 177 min CPU 实验**（`.pytest_cache` 在树里）——按 09-20 定的执行规则（只准秒级检查、不准跑实验）它不算正式样本，只作 Codex 的上界参考；正式 Codex 样本等 owner 用 `paperbench-no-exec.rules` 重跑。

| 提交 | 总分 | 数据集/环境 (w3) | 方法实现 (w3) | 训练/评估 (w3) | 产物 | 无效叶 |
| --- | --- | --- | --- | --- | --- | --- |
| 本线 `line1`（fre-t14，不执行） | **0.8756** | 0.833 | 0.991 | 0.803 | 35 文件 / 29 py | 1（裁判输出撞 65536 上限，记 0） |
| Codex 桌面版 `codex-exec`（执行了实验） | **0.7847** | 0.750 | 0.935 | 0.669 | 42 文件 / 28 py | 0 |

叶级：306 叶里 61 叶不同，本线胜 38、Codex 胜 23。本线赢在 Kitchen 环境（1.00 vs 0.50）、OPAL 架构（0.94 vs 0.61：Codex 把 OPAL 编码器写成 MLP，
置换不变 transformer / 无因果掩码 / 无位置编码 / App.A 超参四叶全丢）、评估节（0.76 vs 0.52：ExORL cheetah 自定义奖励在线评估缺失）。
Codex 赢的叶：walker(RND) 数据集（本线那叶是裁判截断的无效叶）、OPAL 编码器输入 (s,a) 对（本线只喂了状态）、OPAL 自编码目标（本线写成未来状态 MSE + 单位高斯 KL）、
潜变量条件 BC 微调、AntMaze XY 32-bin 离散化没接进 OPAL 训练，以及 **`fre/prior.py:503` 语法错误**（`torch.rand(..., device="cpu", device=device)` 重复关键字）+ `trainer.py:420` 调了不存在的 `prior.evaluate_params`——不执行代码时这类错误没有任何一道闸能拦，裁判读到就扣叶。
读法：① 本线在有执行的 Codex 上界之上 +0.09，是自 sapg/pinn（0.32 / 0.68 量级）以来第一篇上 0.85 的；② fre 08 月作废批里 DeepCode 最好的 trial_fx2 也才 0.49（Pro 裁判、输入有偏），不可直接比；
③ 两边一起丢的集中在训练/评估节；④ 本线值得补一道 `py_compile` 静态检查（不算执行）。
判分：两份 18 分钟；`grade.json` 在 `runs/fre/grades/`（bae9fba6 = line1，d1988ed0 = codex-exec），提交归档在 `~/pb_submissions_archive/fre/0920_{line1,codex-exec}/`。

### 1.5 裁判口径改定：硅基流动 · 整棵树 · 裁判思考关 · Pro 解析器（2026-09-21，`run_grade.sh` 默认）

09-21 一天里裁判换了三次 serving：官方 DeepSeek（json_schema 被拒 → json_object；两次半途作废花 ¥125；判分 key 在 fre 整树那份中途透支到 −¥8.67，165 叶无效，作废）→ 硅基流动（新 key）。最终口径：

| 项 | 值 | 为什么 |
| --- | --- | --- |
| 裁判 | `deepseek-ai/DeepSeek-V4-Flash` @ api.siliconflow.cn，**整棵代码树**进每叶提示词（`PB_JUDGE_WHOLE_CODEBASE=1`，不再每叶选 10 文件；上游 docstring 里的"整库"分支从未存在） | 前缀缓存命中 ~98%，一份 306 叶 ¥32（02:00–08:00 半价 ¥16）；每叶选 10 在硅基约 ¥120；owner："挑十个反而容易错" |
| 裁判思考 | **关**（`PB_JUDGE_THINKING=off` → `enable_thinking:false`，只作用于裁判调用） | JudgeEval 准确率不变（§7），输出 token 少 2/3 |
| 解析器 | `deepseek-ai/DeepSeek-V4-Pro`，`PB_STRUCTURED_JSON_MODE=json_object`（schema 放 system，"回实例不回 schema"引导语，`model_validate_json` 校验） | Flash 解析器随机把 `Score: 0` 判无效（9/178；解析器也关思考时 23/178）；Pro 178/178 |

**fre 第三份 = `fre/line3`（fre-t17，ADR 0004：蓝图 `Source:` 指针 → manifest → 整节读回 → 写前检查 → 审计只记录）**：**0.9210**，306 叶 0 无效，0.833 / 0.991 / 0.939（数据环境 / 方法 / 实验三个子树），2026-09-21 12:24，硅基整树、**思考开**（当时思考开关还没定；与最终默认口径差这一项，待重判）。`grade.json` 在 `runs/fre/grades/fre_9871437d_sf_tree_thinkon.grade.json`，存档 `~/Documents/0919-test/`。**§1.3 / §1.4 的分数是 Paratera 每叶选 10 口径的**，与 line3 不可直比；line1 / line2 不重判（owner）。

从这一份起，Code-Dev 三臂（deepcode / codex / claude）的提交与分数统一收口在 `~/Documents/0919-test/`（README 在那里）。

### 1.6 09-22 夜：18 篇 deepcode（t19）+ Codex 树的第一夜判分（硅基半价窗，4 路并行）

7 篇成对（同口径：硅基 V4-Flash 整树思考关 + V4-Pro 解析器重试 3 次）：

| 论文 | deepcode（t19 / t17） | Codex 桌面 |
| --- | --- | --- |
| fre | 0.961 | 0.914 |
| rice | 0.978 | 0.976 |
| pinn | 1.000 | 1.000（两份，都 1.0） |
| lbcs | 0.987 | 0.993 |
| lca-on-the-line | 0.925 | 0.898 |
| robust-clip | **0.642** | 0.906 |
| what-will-my-model-forget | 0.990 | 0.988 |

只有 Codex 的：bam 1.000、ftrl 0.602。robust-clip 是官方 `paper.md` 缺 §2–§3 方法章的那篇：线靠读回原文，原文没有方法章就吃亏。其余 6 篇差值都在噪声内（±0.03），pinn / what-will / lbcs 两边都接近满分——rubric 对强系统没有区分度。存档与逐叶 grade.json：`~/Documents/0919-test/`（README 表由 `update_readme.py` 从 `grades/` 重建）。

工程教训（都已修）：`run_grade.sh` [4/4] 取全局最新运行组 → 并行时拷错/拷空（本节修为取本论文最新组）；夜间脚本 v1 的 `runs/<paper>/` 目录不存在、v2 的 deepcode 源被先挪走、v3 依赖 [4/4]——最终 v4 直接从 nanoeval 运行目录按时间标记收分。100 路在途（5 棵 × 20 叶）整夜 0 个 429，但单叶延迟拉长，总吞吐约 10–17 叶/分钟。

### 1.4 fre 第二份：ADR 0003 结构化 Source 义务 + 读后写 + 语法检查（2026-09-20 晚，同裁判口径，306 叶）

本线 HEAD `623bbcbdd`→`3d3d45c8e`（`fre-t15`）：planner 输出 27 条义务 / 43 文件（31 paper），写码 43/43、78 次回读收据（13 万字符）、fidelity 审计通过、`compile()` 0 错；449 次调用 12.5M token；83 min 到第 9 步。

| 提交 | 总分 | 数据集/环境 | 方法实现 | 训练/评估 | 备注 |
| --- | --- | --- | --- | --- | --- |
| 本线 `line1`（fre-t14，09-19 自由文本 Source） | 0.8756 | 0.833 | 0.991 | 0.803 | §1.3 |
| 本线 `line2`（fre-t15，ADR 0003） | **0.8389** | 0.833 | 0.991 | 0.693 | 无效叶 0 |
| Codex 桌面版 `codex-exec`（执行了实验，作废） | 0.7847 | 0.750 | 0.935 | 0.669 | §1.3 |

叶级 73 叶不同（line1 胜 40 / line2 胜 33）；前两个大节完全相同，差在**训练节 0.85 → 0.66**：FB / SF 基线（controllable_agent）line2 写成"out of scope"没实现——这正是 manifest 里**唯一没被任何文件认领的义务 `baseline.fb_sf`**；另有 IQL target critic 软更新、OPAL 子轨迹采样几叶。读法：−0.037 ≈ 单篇噪声量级，且成因定位到一条未绑定义务 → 本线加了"有未认领义务就重规划一次"（`79ff0f057`）；ADR 0003 的强约束本身没有把分数拉高，第一次真跑的收益是**机制跑通**（提前失败、审计、语法检查）而不是分。运行中修的四处 manifest 过严和 74 次被拒的 `read_paper` 见本线 HANDOFF / PITFALLS。
判分 15 min；`grade.json` `runs/fre/grades/fre_be597371…`，提交归档 `~/pb_submissions_archive/fre/0920_line2/`。

## 2. sequential-neural-score-estimation（2026-09-14，对标前的数，裁判修 bug 前）

裁判 DeepSeek-V4-Pro @ Paratera，67 叶，无效叶 0；**三份都是思考开的分数**（当时 `enable_thinking:false` 无效，49 次调用 reasoning 78.7 万 / completion 113 万 token）；
三方输入不对标（DeepCode 只拿 paper.md、DeepEvol 只拿 pdf + 手工上传两仓库、裸跑未起）。

| 提交 | 配置 | 得分 | 规模 |
| --- | --- | --- | --- |
| deepevol_s10 | DeepEvol 复现线（旧架构）Stage 10 导出（写码 + 真环境 + 冒烟修复轮后） | 0.7729 | 25 py / 3,854 行 |
| deepcode_trial2 | DeepCode + V4-Pro（索引 max_tokens 16000） | 0.7280 | 35 py / 11,431 行 |
| deepevol_s9 | DeepEvol Stage 9 导出（只写码 + judge，未经修复轮） | 0.6854 | 25 py / 3,817 行 |

| 一级维度（权重） | deepevol_s10 | deepevol_s9 | deepcode_trial2 |
| --- | --- | --- | --- |
| 任务集（App. E.1）(10) | 0.889 | 0.778 | 0.778 |
| VESDE / VPSDE (10) | 0.944 | 0.833 | 0.833 |
| 基线 NPE / SNPE / TSNPE (10) | 0.667 | 0.500 | 0.000 |
| C2ST (1) | 0.000 | 0.000 | 0.000 |
| NPSE (20) | 1.000 | 0.750 | 1.000 |
| TSNPSE (20) | 0.660 | 0.706 | 0.752 |
| 第 5 节结果复现 (20) | 0.607 | 0.607 | 0.755 |

观察（机制层面仍成立）：DeepCode 索引了 `sbi`（NPE/SNPE 就在里面）却一个基线没写；DeepCode 的实验脚本铺得更全。
过程事实：DeepCode trial1（思考开、索引 8000/4000 截断丢 20/51 文件）与第一次 trial2（索引 16000 但思考开）作废；有效 trial2 4 h 26 min，写码期 Paratera 空响应重试 8 次。
DeepEvol 侧 68 次调用约 1,015 万 token ≈ ¥34，跑在租的阿里云 `ecs.c7.2xlarge`。判分每份约 ¥38。

## 3. fre（2026-08-26 → 08-29；**作废：输入有偏**）

306 叶；裁判 DeepSeek-V4-Pro；SF = SiliconFlow serving，PT = Paratera serving（09-03 重判）。DeepCode = e0767d0 + 旧补丁（15 处未门控改动，见 §8）。

| 提交 | 臂 | 规模 | 耗时 | SF | PT | 失分要点（SF） |
| --- | --- | --- | --- | --- | --- | --- |
| anchor | Claude Code + Sonnet 4.5 裸跑 | 15 py / 2,657 行 | ~1 h | 0.4839 | 0.5044 | 数据集/环境 0.333；GC-IQL/GC-BC 满分、OPAL 0 |
| bare_v4 | Claude Code + V4-Pro 裸跑 | 15 py / 3,070 行 | ~5 h | 0.4817 | 0.4807 | 主方法 0.815、三基线 0.67/0.80/0.94；数据集/环境 0.417 |
| trial1 | DeepCode + V4-Pro | 21 py / 6,751 行 | ~4 h | 0.5184 | 0.4682 | 数据集/环境 0.833 最高；**三基线全 0（文件不存在）** |
| trial5 | DeepCode + V4-Pro | 28 py / 8,438 行 | ~5 h | 0.4246 | 0.3101 | 写得最多分最低；三基线全 0；主方法 0.685 |
| **DeepCode / 裸跑** | | | | **0.98×** | **0.81×** | 论文声称 1.34× |

三个一级维度（权重均 3，SF）：数据集与环境搭建 bare 0.417 / trial1 0.833 / trial5 0.750 / anchor 0.333；方法实现 0.809 / 0.412 / 0.343 / 0.792；训练与评估 0.219 / 0.310 / 0.181 / 0.327。
方法实现细目：FRE 主模型 (3) 0.815 / 0.824 / 0.685 / 0.917；GC-IQL (1) 0.667 / 0 / 0 / 1.0；GC-BC (1) 0.800 / 0 / 0 / 1.0；OPAL (1) 0.944 / 0 / 0 / 0。
机制：规划器一次定死文件树，漏掉的对比方法后面几百轮写码补不回来（trial1 计划里"baselines"只在散文出现 2 次，trial5 0 次）。
反事实"补上基线 → trial1 0.6524 = 裸跑的 1.354×"**被修复轮推翻**（§6：基线有分、主方法下滑）。
CodeRAG 预筛失效不是低分主因：规划先于索引（21:17:12 两条日志）；失效率 40%/20%/40% 与分数 0.5184/0.4378/0.4246 不相关；全量索引是筛选的超集。其代价是时间（google-research 8,885 py 全量需 ~140 h）。
判分侧"无文件"叶：bare_v4 13/306、trial5 13、trial1 9、anchor 3；全按上界修正后倍数 0.95×，主结论不变。

## 4. rice（2026-08-29 → 09-02；**作废：输入有偏**）

178 叶；同上两裁判。

| 提交 | 臂 | 规模 | 耗时 | SF | PT | 失分要点（SF） |
| --- | --- | --- | --- | --- | --- | --- |
| bare_v4 | Claude Code + V4-Pro 裸跑 | 11 py / 4,173 行 | 单次 | 0.4680 | **0.1452** | 环境搭建 0.000（9 个环境一个没写）；实验 II (w4) 0.750 全场最高 |
| trial2 | DeepCode + V4-Pro | 22 py / 12,583 行 | 5 h 21 m，¥31 | 0.5447 | 0.4033 | 环境 0.389、解释方法 0.821、实验 III 0.833 |
| trial3 | DeepCode + V4-Pro | 39 py / 25,630 行 | 6 h 46 m，¥45 | 0.4374 | 0.3446 | 环境 0.575、策略网络 0.875；实验 II 0.396（其中 7 叶是判分侧空 `<files>`） |
| **DeepCode / 裸跑（V4-Pro）** | | | | **1.05×** | **2.58×** | 结论随裁判翻转；论文声称 1.95× |
| bare_kimi | Claude Code + Kimi-K2.7-Code 裸跑 | 18 py / 2,463 行 | 22 min | 0.4633 | 0.1865 | 与裸跑 V4-Pro 几乎重合 |
| trial_k1 | DeepCode + Kimi | 36 py / 12,056 行 | 3 h 02 m | 0.2815 | 0.2403 | 实验逻辑还原不足 |
| trial_k2 | DeepCode + Kimi | 32 py / 8,266 行 | 4 h 15 m | 0.4760 | 0.3754 | 1 个语法错误文件 |
| **DeepCode / 裸跑（Kimi）** | | | | **0.82×** | **1.65×** | 同样随裁判翻转 |

维度（SF，bare_v4 / trial2 / trial3）：环境搭建 (1) 0.000 / 0.389 / 0.575；策略网络 (1) 0.375 / 0.625 / 0.875；解释方法 (2) 0.626 / 0.821 / 0.511；实验 I (3) 0.375 / 0.312 / 0.375；精化方法 (2) 0.463 / 0.500 / 0.350；实验 II (4) 0.750 / 0.583 / 0.396；实验 III (2) 0.306 / 0.833 / 0.444；实验 IV (2) 0.333 / 0.333 / 0.333。
实验 II 按精化方法：Ours 1.00 / 0.88 / 0.75；StateMask-R 0.88 / 0.75 / 0.25；JSRL 0.50 / 0.62 / 0.25；PPO fine-tuning 0.75 / 0.25 / 0.38。
这次不是"没写"：trial3 基线相关文件 7/7/10 个，是裸跑的两倍以上，得分只有 41%；7 个失分叶的裁判原话是 "no files or code"——**裁判在大目录树上挑不到文件**（判分侧静默失败，见 §8）。
"2.58×"来自裸跑塌方（−0.32）而非 DeepCode 上涨（−0.14 / −0.09）：严格裁判专门惩罚"抽象类 + 可配置参数"的写法。
合规审计：两份 DeepCode 产物零上游指纹；trial2 日志里唯一一次黑名单命中是我们补的 URL 拦截挡下了对 `chengzelei/rice` 的 fetch。

## 5. 裁判 serving 依赖（2026-09-03；与输入无关，仍成立）

同一提交、同 rubric、同 PaperBench 版本，只换裁判服务商（模型名都是 DeepSeek-V4-Pro）：

| | SiliconFlow 裁判 | Paratera 裁判 |
| --- | --- | --- |
| fre DeepCode/裸跑 | 0.98× | 0.81× |
| rice DeepCode/裸跑（V4-Pro） | 1.05× | **2.58×** |
| rice DeepCode/裸跑（Kimi） | 0.82× | **1.65×** |

rice bare_v4 判分侧诊断（SF vs PT）：每叶输入 60,757 vs 62,852 token；输出 2,937 vs 1,267；零分叶 87 vs 131；疑似"没看到文件"零分叶 39 vs 35；45 个叶 1→0、1 个 0→1。
两裁判看到相同输入，对"通用可配置实现是否算已实现"判断相反；JudgeEval（§7）裁不了。**任何分数必须连同裁判 serving 一起报告；LLM 裁判分不做目标函数。**

## 6. 作废轮与修复轮（不入统计）

约 30 轮复现里 7 轮因流水线自身问题作废：

| 轮 | 配置 | 原因 | 沉没 |
| --- | --- | --- | --- |
| fre trial2 | DeepCode + V4-Pro | 撞 4 h 写码墙钟 | ¥2.90 |
| fre trial3 | DeepCode + V4-Pro | 写码被 900 s stall 熔断，残缺（判了 0.4378，只作参考） | 已判 |
| fre trial4 | DeepCode + V4-Pro | 语料仅 1 仓库（下载侧；报告本身完整 5 条） | — |
| fre trial6 | DeepCode + V4-Pro | 白天 API 限流，写到 9/24 三次重试打完，`status=incomplete` | ¥19.08 |
| rice trial1 ×5 | DeepCode + V4-Pro | ① 克隆 TLS 断流 ② 换网络节点 ③ 预筛静默回退全量索引（140 h）④ 假计划 ⑤ stall 1800 s 熔断 | ≈¥115 |
| rice trial2（首次） | DeepCode + V4-Pro | 随 trial1 一起因 stall 阈值作废 | ¥15.29 |
| fre trial_fx1 首跑 / 二跑 | + 修复①②③ | 挖掘报告截断只见 1 仓库；调大上限重启 | ≈¥3 |
| **fre trial_fx1** | DeepCode + V4-Pro（PT）+ 修复①②③④ | 完整跑完、PT 0.3618；**提示词含评分元知识，整体作废** | ¥22 |
| **fre trial_fx2** | 同上 | 完整跑完、PT 0.4873；同因作废 | ¥21 |
| 判分批 ×2 | rice 三份 / fre 六份 | 余额耗尽 161~168/178 叶无效；裸模型名未登记上下文表 6 份 64 秒全失败 | ≈¥76 |
| snse trial1、trial2 首次 | DeepCode + V4-Pro | 思考开 + 索引截断；思考仍开 | — |
| bam 修裁判前三份 + 并发 16 对照 | 三臂 | 裁判选文件根目录 bug | 已重判 |

修复轮的维度证据（Paratera 裁判，虽作废但机制观察成立）：

| 细目（权重） | bare_v4 | fx1 | fx2 | trial1 | trial5 |
| --- | --- | --- | --- | --- | --- |
| FRE 主模型 (3) | 0.981 | **0.259** | 0.704 | 0.824 | 0.769 |
| GC-IQL (1) | 1.000 | 0.667 | 1.000 | 0.000 | 0.333 |
| GC-BC (1) | 1.000 | 0.600 | 0.200 | 0.000 | 0.000 |
| OPAL (1) | 0.944 | 0.222 | 0.556 | 0.000 | 0.000 |

基线补上了，主方法塌了；覆盖审计两轮均 `ran + adopted`，计划 16,470→19,534 / 16,257→19,231 字符。这就是 README 里"基线运行不开 fix-①②③"的实证：
①②的提示词按评分维度去补文件，得到的是对 rubric 的拟合，不是引擎能力。

## 7. 裁判校准（JudgeEval，rice/0 作者官方仓库，178 叶，code_only；与输入无关）

| 裁判 | 准确率 | macro F1 | 通过率（人工 0.539） | 偏向 | token（入 / 出） | 花费 | 用时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DeepSeek-V4-Pro @ SiliconFlow（08-26） | 0.685 | 0.685 | 0.449 | 严 9.0 pp | — | ¥27.7 | — |
| DeepSeek-V4-Pro @ Paratera（09-03） | 0.719 | 0.719 | 0.449 | 严 9.0 pp（FP 17 / FN 33） | 7.98M / 0.44M | ¥28 | — |
| **DeepSeek-V4-Flash @ Paratera（09-17）**，解析器仍 V4-Pro | **0.719** | **0.716** | **0.562** | **宽 2.2 pp**（FP 27 / FN 23） | 7.68M / 1.19M（+ Pro 解析 0.22M / 0.02M） | 未查账单；按 Flash ≈ Pro 单价 1/4 折算约 ¥13–25 | 9.5 min |

| 硅基 V4-Flash 整树、裁判思考关、Flash 解析器（09-21 `0921c`） | 0.722（169 叶有效，**9 无效**） | F1 0.702 | — | prec 0.744 / rec 0.703 | 150.5M / 0.23M（整树：每叶 ~85 万入，缓存命中） | ≈ ¥40 | 72 min |
| **硅基 V4-Flash 整树、裁判思考关、Pro 解析器**（同一批 0921c 裁判文本离线重解析，引导语修后 3 叶补解析） | **0.702–0.720**（**178 叶 0 无效**） | F1 0.686 | — | prec 0.719 / rec 0.688 | 解析器 0.24M / 0.09M | 解析器 ≈ ¥3 | — |

整树 + 思考关不掉准确率（三种口径都在 0.70–0.72，同 gpt-4o 档）；差别全在解析器的有效率：Flash 解析器随机把明明白白的 `**Score: 0**` 判成"无效"（重跑一遍 8/9 有效——是噪声不是规律，加提示词也救不了），Pro 在 json_object 下会把 schema 原样吐回（2/178，引导语改成"回实例并给示例形状"后消失）。09-21 13:41 那一遍（`0921b`）178 叶全无效是 completer 把 `NOT_GIVEN` 传给了 `extra_body`，代码 bug，作废。原始输出 `runs/judge_eval/0921c_*/`（含 `reparse_pro*.json`）。

Flash vs Pro（Paratera，同一批叶子）：叶级一致 138/178（77.5%）；都对 108、只 Flash 对 20、只 Pro 对 20、都错 30。
准确率相同、偏向相反：Pro 偏严（把人判过的判挂），Flash 偏宽且更接近人工通过率。两者都在 gpt-4o 档（0.681），谁也不比谁准。
Flash 的输出 token 是 Pro 的 2.7 倍（思考开着；PaperBench 的判分请求没有关思考的开关，三次都是思考开）。
**坑**：Flash 不能做二级结构化解析器——Paratera 上带 `response_format` 的请求返回前缀重复的坏 JSON（90 秒 7 个叶子失败），
`PB_STRUCTURED_PARSER_MODEL` 必须留 V4-Pro（见 README §6）。原始输出在本地 `archive/paperbench_changes/judge_eval_results_rice_flash/`。

同一提交上两裁判一致 150/178（84.3%）；28 处分歧里都对 111、只 SF 对 11、只 PT 对 17、都错 39。
官方对照（5 卷宏平均 Code-Dev）：o1-high 0.740 / o3-mini 0.720 / gpt-4o 0.681 / gpt-4o-mini 0.588——两个 serving 都在 gpt-4o 档，无法仲裁 rice 的翻转。
原始 JudgeEval 输出：`archive/paperbench_changes/judge_eval_results_rice{,_paratera}/`。

## 8. 工程发现（独立于分数，可单独引用）

DeepCode（e0767d0）流水线里同一模式的四处静默降级——LLM 输出超限或为空后，下游当正常继续、只留 INFO 日志：

| 位置 | 现象 | 后果 |
| --- | --- | --- |
| CodeRAG 预筛（`tools/code_indexer.py`，`max_tokens=2000`） | 大仓库 JSON 截断 | 静默回退全量索引（17 文件仓库成功，151/239 文件 100% 失败） |
| 参考挖掘报告（`maxTokens=4096`） | 报告截断，续写只留尾段 | 下载侧只看见 1/5 仓库，整轮语料贫瘠 |
| 预筛返回合法空列表 | 与"调用失败"共用分支 | 同样回退全量 |
| 判分侧文件选择返回空 | `<files>` 为空 | 叶子静默得 0，`valid_score` 仍为 True |

另：写码阶段从不验证自己的产物——跨 31 个归档 2,443 次工具调用，`command_executor` 被调用 50 次全是 `mkdir`/`touch`/`find`/`ls`/`cat`，没有一次运行生成的代码；索引模式下写码 agent 只有 `write_file` 与 `search_code_references` 两个工具。
旧补丁里 15 处未门控改动（挖掘 maxTokens 4096→8192、`max_iterations` 8→80、墙钟 7200→14400、stall 300→1800、下载 agent 提示词重写 + `max_iterations=40`、fetch 限流、空 code_base fail-fast、分段提示加固等）使"官方默认配置"的表述不成立；本仓库 2026-09-17 起的补丁已全部 env 门控、默认等于上游（README §对上游的改动）。
编译检查：fre/rice 全部有效提交 `py_compile` 通过（trial_k2 31/32）；可运行性不构成任何一方的优势。

试点期（不入统计）：阶段 A 白卷 0.000；阶段 B（规划 DeepSeek + 写码 Kimi，无 CodeRAG）43.3；阶段 B′（+CodeRAG，8 轮拼装、人工裁剪语料）60.5，正式协议下不再现；E1 全 Kimi 干净端到端 14 分钟塌方。

时间线与成本：装环境 08-25~26（13 个坑）→ 试点 08-26 → 协议定案 08-26 晚 → fre 08-26~29 → rice 08-29~31 → Kimi 09-01~02 → 修复验证 09-02~03 → 双裁判重判 09-03 → 公开 09-03 → snse 09-14 → 输入标准 + bam 三方 09-14~15。
总花费约 ¥1,600（约 30 轮复现 + 25 份判分 + 2 次 JudgeEval，含全部废轮）；单价：一轮复现 ≈ ¥20 / 3~6 h，一份判分 ≈ ¥38 / 40~100 min（V4-Pro 裁判）。

## 9. 原始文件在哪（本地，不入库）

`~/Documents/env/paperbench-judge/archive/`：

- `deepcode_test/{fre,rice,sequential-neural-score-estimation,bam}/`：`RESULTS.md`、`grades/*.grade.json`（Paratera 版带 `paratera_` 前缀；作废的文件名标明原因）、`submissions/`、`logs/`、`workspaces/`、`task_archives/`
- `deepcode_test/docs/`：CONCLUSIONS / FINDING_judge_serving_dependence / FINDING_prefilter_silent_failure / FINDING_generic_pipeline_failures / REVIEW_local_changes_2026-09-03 / RESULTS_MASTER / PROJECT_CHRONICLE / DECISIONS / DEEPCODE_INTERNALS / OPTIMIZER_NOTICE / PAPERBENCH_RUNBOOK / ARCHITECTURE_* / HANDOFF_* / CC_FRE_PROMPT.txt（作废的裸跑任务书）
- `paperbench_changes/{judge_eval_results_rice,judge_eval_results_rice_paratera,rubrics}/`
- `DeepCode-e0767d0-patched/`：跑出上面全部 DeepCode 数字的那份源码（上游 e0767d0 + 旧补丁）
