# Paper2Code 线第二批：基线运行与判分准备

本文是 2026-09-17 访谈（PLAN.md 之后的第二轮）的施工清单。词汇按根 `CONTEXT.md`：**对比方法**是论文里被比较的算法，**基线运行**是原装 DeepCode 在同口径下的一次运行；不再用裸的"基线"。上一批的结果与偏差在 `HANDOFF.md` 和 PLAN.md §10。

状态栏由我维护：`todo` / `doing` / `done`，附提交号或运行 id。

## 0. 决策摘要

| 项 | 决定 |
| --- | --- |
| 判分 | 暂不判分；本线只做自测（四道闸门、compileall、发现测试时的 pytest、入口冒烟） |
| 基线运行用的 DeepCode | main `21ebc57f` + 与本线 VENDOR.md 相同的"带走"补丁子集；fix-①②③ 带上但默认关（见 §1） |
| 验证仓库 | `2UBBISH/deepcode-paperbench-validation`：只保留验证脚手架与判分脚本；DeepCode 继续 vendor 完整副本 + `patches/` + `setup.sh`；产物与日志移出，旧结果统一进 `docs/RESULTS-HISTORY.md`；README 大更新，clone 即跑 |
| 口径 | DeepSeek-V4-Flash，thinking 关闭（按 `reasoning_tokens == 0` 判），同一端点；两边一致 |
| 迭代预算 | 本线与验证仓库默认 `DEEPCODE_REFERENCE_MAX_ITERATIONS=40`、`DEEPCODE_DOWNLOAD_MAX_ITERATIONS=12` |
| 索引 | 不加文件数与时长上限 |
| requirements 镜像 | best-effort 不变；`environment_run` 结果显式记 `requirements_installed` |
| 入口冒烟 | 加，只记录不判失败 |
| 判分脚本 | vendor 进 DeepEvol `vendor/paperbench-judge/`；来源是验证仓库，VENDOR.md 记提交 |
| submit | 本线加 `submit` 子命令，四闸全过才允许摆卷到 `~/pb_submissions/<paper>/<trial>/` |
| 踩坑记录 | `apps/v2/agent/paper2code/PITFALLS.md` 统一维护；HANDOFF 只留运行记录 |
| 推送 | `Paper_repro_0916` 已在 origin；回填只在本会话做 |
| key | 你把有效 key 写回 `~/Documents/env/paratera.env`（有效的那把现在在 `paratera_backup.env`） |

## 1. 为什么基线运行不开 fix-①②③

三条都在 `patches/deepcode_local_changes.patch` 里，env 开关，默认关：

- fix-① `DEEPCODE_PLAN_COVERAGE_CHECK`：蓝图出完后追加一次"审计"调用，提示词按 (1) 每个对比方法 (2) 每个正文实验 (3) 每个数据集/环境 三条去查漏并补文件。
- fix-② `DEEPCODE_ALLOW_PLAN_EXTENSION`：实现循环里告诉模型"蓝图不是上限，缺对比方法或实验的文件就新建"。
- fix-③ `DEEPCODE_POSTWRITE_COMPILE`：每次 `write_file` 后本机 `py_compile`，失败回灌让模型重写。

①② 的提示词就是 PaperBench rubric 的三个评分维度。带着它们跑出来的分数衡量的是"我们对 rubric 的了解"，不是引擎；这就是过拟合评测。基线运行的意义是给本线一个原装的参照，所以基线**必须**关着它们；带进仓库只是为了让人能 A/B 出这三条各值多少分。本线不带 ①②：对比方法的覆盖交给计划审阅（`--ask`）和将来的 A3 自建循环；③ 的事本线已由远端 compileall 作业机械完成，"发现错误后让模型修"属于 PLAN.md §8 的闭环修复，到时在同一个挂点做，不另起一套。

## 2. 分步实施

每步一个提交（验证仓库的提交在它自己的仓库）。

### D0 · 本线小改 — `done`（提交见 git log：paper2code: D0）

做什么：

1. `config.py` `ENV_DEFAULTS`：`DEEPCODE_REFERENCE_MAX_ITERATIONS` 8→40，`DEEPCODE_DOWNLOAD_MAX_ITERATIONS` 8→12；PLAN.md §6 表同步。
2. `phases.py` `environment_run`：结果加 `requirements_installed`（读 `jobs/image.json`）；加入口冒烟：入口 = 蓝图 `implementation_strategy` / `file_structure` 里标注的入口文件，否则根目录或唯一子项目目录下的 `main.py`、`run*.py`、`train.py`、`experiment*.py`，找不到记 `entry: none`；先 `python <入口> --help`，非零再裸跑，各 60 秒，超时算未跑通；只记录（`entry_smoke` 字段）。
3. `job_executor.py`：无改动（后台容器已落地）。
4. 测试：`test_driver_offline.py` 加入口冒烟断言（假端口收到 `environment_run:entry` 作业）；`test_gates.py` 不变。

验证：`pytest tests/v2_paper2code -q` 全绿；`ruff` 干净。

### D1 · 验证仓库更新 — `done`（origin master `6ce8cfc` → `e668087` → `ad5fd7e`，2026-09-17）

仓库：`~/Documents/env/paperbench-judge/validation`（origin `2UBBISH/deepcode-paperbench-validation`）。目标布局：

```
README.md                       大更新：是什么、怎么用（clone 即跑）、口径、坑、作废规则
setup.sh                        建 venv、装 DeepCode、打 patches/、装 PaperBench 补丁
DeepCode/                       HKUDS main 21ebc57f 原样副本（vendor）
patches/
  UPSTREAM_BASE.txt             21ebc57f
  deepcode_local_changes.patch  基于 21ebc57f 重生成：带走子集 + fix-①②③（默认关）+ thinking compat 已在上游
  paperbench_local_changes.patch
config/                         deepcode_config.json 模板（V4-Flash、compat.thinking=disabled、无阶段覆盖）
deepcode_test/scripts/          run_trial.sh run_grade.sh gates/ stage_b_driver.py ...
paperbench_changes/             PaperBench 补丁副本
docs/RESULTS-HISTORY.md         bam/fre/rice/snse 全部历史数字与结论（含作废标记）
docs/INPUT_STANDARD.md          三方输入标准（从 bam-threeway 搬来）
```

做什么：

1. `DeepCode/` 换成 `21ebc57f` 副本；用本线 VENDOR.md 的第 2–7 条 + fix-①②③ 在 `21ebc57f` 上重生成 `deepcode_local_changes.patch`（`code_implementation_workflow.py` 手工合）；`setup.sh` 打补丁并校验 sha。
2. `run_trial.sh`：读 `$DEEPCODE_HOME`（缺省 `~/.deepcode`）；口径检查改为 `DeepSeek-V4-Flash` + `compat.thinking == disabled`；注入 `REFERENCE 40 / DOWNLOAD 12`；黑名单与 addendum 拼接不变；摆卷段不变。
3. 瘦身：`deepcode_test/{bam,fre,rice,sequential-neural-score-estimation}` 的产物、日志、提交移出仓库（本地归档到 `~/Documents/env/paperbench-judge/archive/`），`grade.json`、`RESULTS.md` 类小文件的数字并入 `docs/RESULTS-HISTORY.md` 后一并移出。
4. README 重写，章节：用途 → 口径 → 快速开始 → 目录 → 对上游的改动（含 §1 的过拟合说明）→ 坑 → 作废规则 → 历史结果索引 → 诚实声明 → 许可证。
5. 推送到 origin main。

验证：干净机器上 `git clone && bash setup.sh && bash deepcode_test/scripts/run_trial.sh --help` 走通；`git ls-files | wc -l` 不含产物；README 里每条命令可复制执行。

施工记录与偏差（2026-09-17）：

- `DeepCode/` 提交的是**打过补丁的完整副本**（1,011 文件），不是"原样副本 + setup 时打补丁"：后者会让 `git status` 永远脏、`git pull` 冲突。等价保证改由 `patches/verify_deepcode.sh` 给出（15 个文件 sha256 清单 + 补丁反向干跑），setup.sh 每次都跑。
- 补丁 15 文件 +628/−49，分四组（README §5.1）：A = VENDOR.md 2–7；**B = 基线运行必需的平权项**，超出了"只带 2–7"——`mcp.py` 名字消毒 / `DEEPCODE_URL_DENYLIST` / 重复 fetch 账本、重试 env、`compat.thinking`（来自 DeepCode 维护者 main 工作树未提交的改动，21ebc57f 上游**没有**它，PLAN-2 原写"thinking compat 已在上游"不成立）、以及本线引擎已带的四处静默失败修复（未缩进围栏取计划、`write_multiple_files` 计数、写类工具按参数摘要键、验证根目录下钻）；C = fix-①②③ 默认关；D = 去掉索引复用与 `loop_detector` 豁免。
- `DEEPCODE_HOME` 缺省改为 `<仓库>/.deepcode-home`（不是 `~/.deepcode`），按"本地 venv + 本地 DEEPCODE_HOME"的隔离规则；key 走 `ENV_FILE`（`apiKeyEnv: PARATERA_API_KEY`），不碰任何 credentials.json。
- 上游 21ebc57f 用 `requirements.txt`（`uv.lock` 为空）；Intel macOS 上 `cryptography` 50 无 wheel，`--only-binary cryptography`（装到 48.0.1）。`DEEPCODE_WORKSPACE` 不能设（pydantic-settings 把它当 workspace 配置对象）。filesystem / fetch MCP 装成固定路径（npx 首次解析 20 s+ 撞连接超时；uvx 重编 cryptography），且 `deepcode_lab/` 要先建。
- 瘦身：2,830 → 1,052 个跟踪文件；产物、旧文档、退役脚本、JudgeEval 输出、rubric 副本在 `~/Documents/env/paperbench-judge/archive/`。
- 验证：本机 `PREFLIGHT_ONLY=1` 过闸；并稿输入 sha256 `04790c3f…` 与本线 C9 `run.json` 一致；DeepCode venv 里 7 个 MCP 全连（37 工具）、`apply_chat_compat` 出 `thinking:{type:disabled}`；scratch 里 `git clone → setup.sh → --help → preflight` 走通。

### D2 · 基线运行：sapg — `done`（2026-09-17 12:36–13:24，task `paper_83d09fcd`，摆卷 `~/pb_submissions/sapg/trial1/`；两边并排的表在 HANDOFF；本轮 `max_tokens` 被目录钳在 8192、后置闸门因我改了运行中的脚本而手动补跑，均已记录）

做什么：在这台 Mac 上用 D1 的仓库跑 `PAPER=sapg TRIAL=trial1 bash run_trial.sh`（`DEEPCODE_HOME=search/.deepcode-home`，DeepCode 自己的凭据，本地沙箱）。跑完把两边并排写进 HANDOFF：计划字符数与模式、参考仓库数、索引数、生成文件数、各阶段耗时、token、`reasoning_tokens`、验证状态。**不判分**。

验证：run_trial.sh 的三道闸（口径、假计划、状态）通过；产物目录 ≥ 5 个文件；`llm` 日志 `reasoning_tokens` 全 0。

### D3 · DeepEvol 侧：judge vendor 与 submit — `done`（2026-09-17）

做什么：

1. `vendor/paperbench-judge/`：从 D1 定稿的验证仓库复制 `run_grade.sh`、`stage_b_driver.py`、`scripts/gates/`、`paperbench_changes/`（补丁形态）+ `VENDOR.md`（来源提交、复制命令、不改动声明）；ruff 排除；footprint 登记。
2. `scripts/paper2code_canary.py submit --run-dir R --paper <name> --trial <id>`：四闸全过且 `environment_run` completed 才允许；复制 `generate_code/` 到 `~/pb_submissions/<paper>/<trial>/`（排除 `__pycache__`、`.git`）；在 run 目录写 `submission.json`（目标路径、文件数、sha256 清单、口径）；`status.json` 记 `submitted_at`。
3. 测试：`test_driver_offline.py` 加 submit 用例（闸门未过拒绝、通过后清单正确）。

验证：本线离线测试全绿；`bash vendor/paperbench-judge/run_grade.sh --help` 可运行（不真判）。

施工记录：`run_grade.sh` 本身不认 `--help`，且它按验证仓库的相对布局找 PaperBench（上两级的 `frontier-evals/`），所以 vendor 目录**原样保留相对布局**（`deepcode_test/scripts/run_grade.sh`），另加我们自己的 `judge.sh`（`--help` / `setup` / `grade`）作入口；没有 vendor `stage_b_driver.py`（那是 DeepCode 的驱动，不是判分脚本）。`submit` 拒绝已存在的目标，`--force` 才覆盖。

### D4 · PITFALLS.md — `done`（2026-09-17，六组 60 条，每条指向旋钮或测试；HANDOFF 的 Pitfalls 节改为一行指向）

来源：HANDOFF "Pitfalls seen"、验证仓库 README 与 `deepcode_test/README.md` 的 pit 条目、`~/Documents/env/bam-threeway/HOWTO.md` 与 `HANDOFF.md`、`search/SETUP.md` 的 DeepSeek 思考截断条目。每条按"现象 → 原因 → 本线怎么处理 / 哪个旋钮"。写完把 HANDOFF 的 Pitfalls 节改成一行指向。

验证：每条能对应到代码里的一个旋钮或一个测试；无重复。

### D5 · 复跑一次远端 — `done`（2026-09-17，两次：第一次 requirements 装 CUDA 版 torch 超时；第二次 CPU 索引 2 分钟装好，compileall 过，入口冒烟暴露生成仓库嵌套包不能启动；机器均已释放）

做什么：`rerun --phase environment_run` 跑 C9 的 `sapg`（V4-Flash 口径，`paratera.env` 写回后），验证后台容器、requirements 安装（torch）、入口冒烟在 Aliyun 上的首次实战；结果补进 HANDOFF。

验证：`jobs/` 有 pip 构建日志且 `image.json` 为 `ok` 或有明确失败原因；`compile_check.passed`；`entry_smoke` 有记录；`lease.json` `released_at`；控制台无 `p2c-*`。

## 3. 不做

判分本身、Gateway、Product 表、闭环修复、A3、extract phase（PLAN.md §8 原样）。

## 4. 验收

D0–D5 状态全 `done`；HANDOFF 有 sapg 两边并排的数字；验证仓库 README 能让第三方 clone 即跑。

**2026-09-17 验收**：D0–D5 全 `done`；HANDOFF「Second batch」有并排表；验证仓库 scratch 里 clone → setup → 预飞 → 7 个 MCP 全连已走通（origin master `f9aa253`）。
