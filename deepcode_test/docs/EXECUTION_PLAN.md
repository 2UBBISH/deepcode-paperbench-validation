# DeepCode × PaperBench 执行计划书 v2

> 2026-08-26 · 供用户审批 + 接棒模型(Opus 5)执行
> 配套文档:`PAPERBENCH_RUNBOOK.md`(环境与六坑)· `MORNING_REPORT.md`(阶段 A 战报)

---

## 0. 执行规则(接棒模型必读)

1. **每个阶段只在用户明确口令后启动**(如"开跑B"/"跑C"),阶段内部的子步骤无需逐个再批;
2. **闸门是硬约束**:超线立即停、保留产物、报告实际花费,等新口令;
3. 用户偏好:中文交流;小白友好(比喻+人话,已建立"考试院/考生/裁判"术语体系);证据导向(结论都要能指到日志/judge 判词);**先批后跑,宁停勿闯**;
4. 杀进程用 `pkill -f "模式带[括]号"` 防自杀;`pgrep -f` 会匹配自身命令行,验证进程存活必须用括号技巧;
5. 花费无法从硅基流动 API 实时读取,靠 token_usage(judge 侧在 grade.json)+ DeepCode 日志调用计数估算,每阶段结束记入第 7 节台账。

## 1. 当前状态快照(2026-08-26 02:10)

| 项 | 状态 |
| --- | --- |
| 阶段 A(dummy+真裁判+code_only) | ✅ 跑通:178/178 有效判分,白卷 0 分,判分链路验证为真 |
| 六个上游坑 | ✅ 全部修毕(清单见 RUNBOOK「六处落差」;含 2 处最小代码补丁,均带 `[local compat]` 注释) |
| paperbench `.env` | ✅ 硅基流动 key + base_url + `PB_STRUCTURED_PARSER_MODEL`(启动自动 load_dotenv,无需 source) |
| DeepCode | ✅ init 完成;provider=siliconflow(key 在其私有凭证库);默认模型 DeepSeek-V4-Pro;真推理冒烟通过 |
| B 点火器 | ✅ `run_stage_b.sh` + `stage_b_driver.py`(导入链已验证) |
| C 前置 | ✅ pb-reproducer 镜像 v2(python/pip/jupyter/PEP668 全解);NVIDIA 容器运行时可用 |
| 正在运行的东西 | **无**(B 曾误启 1 分钟已杀净,只跑到文档分析,花费 <¥0.5) |
| 裁判成本实测单价 | **≈¥17/篇**(178 条 code_only;每条携带全文论文 ~2.5 万 token) |

## 2. 闸门(已按用户要求上调)

| 闸门 | 数值 | 超线动作 |
| --- | --- | --- |
| 阶段 B 总额(复现+判分) | **≤ ¥100** | 停、保留产物、报实际数 |
| 阶段 C 总额(复跑+全维判分) | **≤ ¥60** + 复跑墙钟 2h | 同上;复跑超时自动强杀 |
| 阶段 C′ 侦察 | ¥0(只读);小跑另行报价审批 | — |
| **会话累计**(含已花 ≈¥32) | **≤ ¥200** | 全线停 |
| 单步卡死 | 无输出 >40 分钟视为卡死 | 杀 → 按排障手册分诊 → 报告 |

## 3. 阶段 B 详案:DeepCode 真考 rice(等口令"开跑B")

**目标**:用 DeepCode 完整流水线复现 rice 论文 → PaperBench 真裁判按 178 条 Code Development 细则判分 → 与 dummy 的 0 分基线对照,得到"DeepCode 复现能力"的第一个实测数据点。

**启动命令**(一条,内含三步):
```bash
bash /home/deepevol/deepevol/run_stage_b.sh 2>&1 | tee -a ~/deepevol/stageB_console.log
```
(降级选项:`bash run_stage_b.sh --fast` 跳过参考挖掘+索引,费用约减半,但测的不再是完整看家本领——仅在完整模式失败时用)

### B1 · DeepCode 复现(预计 40min~2.5h,¥15~45)
- 完整模式:Phase 0-10 全走(含参考文献挖掘、GitHub 仓库下载、代码索引、循环写码 ≤2h 墙钟);
- 输入:`frontier-evals/.../data/papers/rice/paper.md`(直接喂 md,绕过 PDF 转换环节);
- 日志:`~/deepevol/stageB_deepcode_<时间戳>.log`(驱动器 tee 输出);
- **成功判据**:流水线状态 `completed`/`completed_with_warnings`,且 `generate_code/` ≥5 个代码文件;
- **失败分支**:
  - a. 参考挖掘/GitHub 下载环节网络失败 → 用 `--fast` 重跑(¥5~15);
  - b. 规划(Phase 5)三次重试仍不合格 → 停,报告日志段落,等用户;
  - c. 实现循环提前终止(墙钟/复读)→ **部分产物照样进 B2 判分**(判分对"写了多少"同样有信息量);
  - d. 无输出 >40 分钟 → 按卡死处理。

### B2 · 摆卷(脚本自动,秒级)
- `generate_code/` 全量拷入 `~/pb_submissions/rice/submission/`(路径由 `/tmp/stage_b_code_dir.txt` 传递)。

### B3 · 判分(预计 ~10min,¥17~25)
- PBDirectSubmissionSolver + code_only + DeepSeek-V4-Pro 裁判(命令在 run_stage_b.sh 第 3 步,含全部六坑修正参数);
- **成功判据**:`n_gradings_failed=0`,178 条有效判分;
- 产出:`frontier-evals/project/paperbench/runs/<最新组>/rice_*/grade.json`。

### B4 · 解读(¥0,自动执行不需再批)
- 报告:总分、按 rubric 子树的得分分布、抽样 5-8 条"得分/失分"判词原文;
- 对照口径:dummy 基线 0 分;**不可**直接对标官方 73.5%(模型/时限/维度都不同,报告里必须写明这一点);
- 花费实算(token_usage)记入台账。

## 4. 阶段 C 详案:GPU 全量复跑(等口令"跑C";⚠️ 风扇高转最长 2h,建议白天)

**目标**:把 B 的提交在 pb-reproducer 容器里真执行,补上 Execution + Result Match 两类维度,共 361 条全维判分。

**前置**(全绿):镜像 v2 ✅ / NVIDIA 运行时 ✅ / B 的提交存在。

**启动命令**:
```bash
cd /home/deepevol/deepevol/frontier-evals/project/paperbench && export PATH="$HOME/.local/bin:$PATH"
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$HOME/pb_submissions/ \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.computer_runtime.env.is_nvidia_gpu_env=true \
    paperbench.reproduction.timeout=7200 \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model='deepseek-ai/DeepSeek-V4-Pro' \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder
```
- 预计:复跑 0.5~2h(取决于 DeepCode 写的 reproduce.sh 实际跑什么)+ 判分 ¥25~35;
- **成功判据**:`submission_executed_metadata.json` 产生 + 361 条判分完成;
- 失败分支:复跑爆新坑 → 连修 2 个仍不过则停(老规矩);reproduce.sh 秒退 → 照常判分(Execution 项自然低分,本身就是结论)。

## 5. 阶段 C′ 详案:SWE-bench 无 GPU 线(等口令"侦察C′")

1. **侦察(¥0)**:读 `DeepCode/eval/swebench/` 挂架(README/run.py/dataset.py),确认:用哪个数据集(SWE-bench Lite/Verified?)、评测容器需求、单实例预计 token 消耗;
2. 产出**报价单**:跑 N 个实例的费用/时长方案(如 5 实例试跑);
3. 用户批准后小跑;执行级验证(单元测试过=对),全程无 GPU、安静。

## 6. 排障速查(接棒模型用)

| 症状 | 大概率原因 | 处置 |
| --- | --- | --- |
| chz `Extraneous argument ...env` | 忘了先显式设 `computer_runtime=` 实现类(坑1) | 按本文命令原样跑 |
| `GRADER_OPENAI_API_KEY is not set` | 坑2 | `.env` 已配,确认 cwd 在 paperbench 目录 |
| 复跑拉 Docker Hub 镜像失败 | 坑3(reproduction runtime 没配) | 命令里三行 reproduction.* 别删 |
| 容器报无 python/pip/jupyter | 坑4/PEP668(理论已绝迹) | `docker run --rm pb-reproducer:latest sh -c 'python --version && pip --version && jupyter --version'` 验证;失败则重跑 bootstrap 第5步 |
| `Model ... not found in context window lengths` | 坑5(换了新模型名) | 在 `common/preparedness_turn_completer/utils.py` 表里加行 |
| 判分 178 条全 invalid / 400 Model does not exist | 坑6(解析模型) | 确认 `.env` 里 `PB_STRUCTURED_PARSER_MODEL` |
| 硅基流动 401 | DeepCode 侧 provider 凭证丢失 | `deepcode provider test siliconflow --model deepseek-ai/DeepSeek-V4-Pro` 诊断;key 源头在 `DeepEvol/DeepEvol1.0/config/models/llm_models.yaml` |
| Responses API 404 | 硅基流动没有该接口(实测) | 不要用 BasicAgent 考生,别浪费时间 |
| DeepCode 流水线卡某 Phase | 看 `stageB_deepcode_*.log` 尾部 + `deepcode_lab/tasks/<id>/logs/` | 按 B1 失败分支表分诊 |

## 7. 花费台账(人民币,硅基流动)

| 时间 | 项目 | 金额 |
| --- | --- | --- |
| 08-26 01:4x | A2 判分(调用成功解析失败,费用照产生) | ≈15(估) |
| 08-26 01:5x | A3 判分(成功,token_usage 实测) | 17.2 |
| 08-26 各次 | 连通/结构化/provider 冒烟 ×4 | <0.1 |
| 08-26 02:05 | B 误启 1 分钟(输入获取+文档分析起步) | <0.5 |
| 08-26 02:09 | B 第 1 轮:分段+规划(产出 23KB 方案,后续复用) | ≈3 |
| 08-26 02:15 | B 第 2 轮:分段+参考+实现(20 次调用,中止于进度停滞) | ≈4 |
| 08-26 02:33 | B 第 3 轮:fast 模式重跑(仍卡 3 文件,25 次调用) | ≈2 |
| 08-26 02:57 | B 第 4 轮:Kimi-K2.7-Code 实现成功(119 次调用,23 文件) | ≈9 |
| 08-26 03:29 | **B 判分成功:43.3 分**(in 10.88M / out 0.74M) | **37.1** |
| 08-26 10:11 | **JudgeEval(rice·code_only)**:裁判 F1=0.685(in 7.94M/out 0.65M) | 27.7 |
| 08-26 10:31-15:17 | B′ 八轮攻坚(参考挖掘/下载/索引/写码,拦截判分×4省¥148) | ≈22 |
| 08-26 15:2x | **B′ 判分:60.5 分**(in 10.10M/out 0.61M) | 34.0 |
| **累计** | | **≈170** / 上限 200 |

**成本结构实测结论**:DeepCode 跑一轮很便宜(**¥3~5**,几十次调用);
**真正贵的是 PaperBench 判分(¥17/篇)**,因为 178 条判分每条都要携带全文论文。
所以"多试几次 DeepCode、只判分一次"是最经济的策略。

## 8. 待办队列(按口令触发)

- [x] **"开跑B"** → ✅ 完成:**43.3 分**(107/178),4 轮才成功,详见 `MORNING_REPORT.md`
- [ ] **"跑C"** → 第 4 节(建议白天,有噪音)
- [ ] **"B′ 补 CodeRAG 重跑"** → 用已修好的 filesystem/fetch 再跑,看 43.3 能提多少(≈¥52,安静)
- [ ] **"侦察C′"** → 第 5 节第 1 步(免费)
- [ ] **"读判分"** → 逐条翻 178 条判词(免费)

## 9. 阶段 B 结论(2026-08-26 03:30)

**得分 43.3%**(107/178 条 Code Development 通过),对照白卷基线 0%。
强项:解释方法 0.83、环境搭建 0.82;弱项:实验 II 复现 0.19(权重最高)、精炼方法 0.31。
产物 23 文件 / 7269 行。**保留意见**:CodeRAG 全程未生效(缺搜索后端),
所以这是"减配成绩";产物分裂在 `rice/` 与 `RICE/` 两个根目录;只测了 Code Development 一个维度。

**新增两坑(DeepCode 侧,已写进 RUNBOOK)**:
- 坑7:`deepcode init` 不写自己流水线的 MCP 服务器 → 全程无工具;
- 坑8:推理模型 + 默认 8192 token = 写码必死(DeepSeek 225s/12K token vs Kimi 29.6s/2.2K,快 7.6 倍)。
  修法已落配置:`agents.implementation` 指向 `moonshotai/Kimi-K2.7-Code`,`maxTokens=32768`。
