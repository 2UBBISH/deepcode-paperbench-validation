# 任务:用 fre 验证 DeepCode 论文效果是否真实

## 终极诉求

验证 DeepCode 论文(arXiv 2512.07921,第四节)声称的效果是否真的这么厉害。以 PaperBench 的 **fre** 任务为样本,三方设定:

1. **被测**:DeepCode 框架 + **DeepSeek-V4-Pro** 底座(SiliconFlow),跑 fre 复现,2~3 次独立 trial 取均值(论文协议是 3 次);
2. **锚点**:用户的 **Claude Code 订阅**(Sonnet 4.5,thinking 开)手动按 PaperBench Code-Dev 协议跑一次 fre——对照论文 Table 1 里 Claude Code 在 fre 的 **0.6286**,用来标定自建裁判的偏移量。注意:订阅不能也不允许接成 API 喂给 DeepCode,只能用 Claude Code 本体跑;超 5 小时用量窗口就分段续跑,中途不人工给提示;
3. **裁判**:PaperBench SimpleJudge(code_only)+ **DeepSeek-V4-Pro** 做 completer——`run_stage_b.sh` 第 [3/3] 步已配好,全程恒定不换。已知此裁判对官方 o3-mini 的 F1≈0.685,**绝对分不能直接对论文数字**,只能:(a) 用锚点校正后再对,(b) 比较同裁判下的相对差距。

## 判读标准

- 论文 fre 参考值(Table 1):Codex 0.4095 / **Claude Code 0.6286** / Cursor 0.6344 / **DeepCode(Sonnet 4.5)0.8435**;图 5 换底座阶梯(fre):Claude 4.5 ≈0.823、GPT-5 ≈0.77、中档模型 0.44~0.57、DeepSeek-R1 ≈0.29。
- V4-Pro 在论文里**没有参考值**,结论按档位读:校正后 ≥0.7 → 前沿档;0.45~0.6 → 中档;≈0.3 → 与 R1 无实质差别。
- 论文声称的架构增益检验:同裁判下 DeepCode(V4-Pro) vs Claude Code 锚点的比值,及锚点自身与 0.6286 的偏差(锚点校正后明显高于 0.63 → 论文可能把基线跑弱了)。

## 环境事实(2026-08-26 已逐一核实,勿重新怀疑)

- 工作目录 `~/deepevol`;DeepCode 仓库在 `~/deepevol/DeepCode`(有**未提交**的本地补丁,勿 checkout/stash 丢掉)。
- **历史教训(已结案)**:V4-Pro 之前"写码无输出"的根因 = implementation 阶段 maxTokens 只有 8192,thinking 吃光输出预算 → `Output truncated`/`Empty response` 反复重试 → 单轮 2 分半 → 被 300s stall 护栏误杀。**不是模型不行**。现已修复:
  1. `~/.deepcode/deepcode_config.json`:`agents.implementation.maxTokens=32768`(已改好);
  2. `DeepCode/utils/loop_detector.py:99`:write_file 连续调用豁免(已打);
  3. `DeepCode/workflows/code_implementation_workflow.py:432`:stall 阈值 300→900(已打)。
  修复后 V4-Pro **尚未重试过**(之后的成功 run 都是 Kimi-K2.7-Code,见 stageB_deepcode_0826_1439.log,24/24 文件跑通)。
- **必改**:`~/.deepcode/deepcode_config.json` 里 `agents.implementation.model` 当前是 `moonshotai/Kimi-K2.7-Code`,要切回 `deepseek-ai/DeepSeek-V4-Pro`。`agents.defaults.model` 同理检查。
- **建议预防**:`code_implementation_workflow.py:432` 的 LoopDetector 加 `timeout_seconds=1800`(单文件默认 600s,V4-Pro 推理慢容易误杀);治本方案是把 `LoopDetector.note_llm_wait()` 接进 workflow 的 LLM 调用处(钩子已存在但零调用点)。
- **thinking 保持开启**:论文所有模型都带推理跑,关 thinking 只作为频繁截断时的最后退路,用了必须在结论注明。
- fre 资产齐全:`~/deepevol/frontier-evals/project/paperbench/data/papers/fre/`(paper.pdf、paper.md、addendum.md、blacklist.txt、rubric.json)。执行时遵守 blacklist.txt(禁访原作者仓库)。
- **把管线从 rice 切到 fre,共三处**:
  1. `~/deepevol/stage_b_driver.py:25`:`data/papers/rice/paper.md` → `data/papers/fre/paper.md`;
  2. `~/deepevol/run_stage_b.sh`:`~/pb_submissions/rice` 等路径改为 fre;
  3. 判分 split:`frontier-evals/project/paperbench/experiments/splits/debug.txt` 当前只有 `rice`,新建 fre 专用 split(或临时改 debug.txt),并同步 `paperbench.paper_split` 参数。
- 踩坑手册:`~/deepevol/PAPERBENCH_RUNBOOK.md`(13 个坑及修法,先读再跑)。
- 成本参考(rice 实测):复现一次 ¥10~40(--fast 约 ¥5~15)+ 判分 ¥17~25;fre 论文更长,预算略上浮。

## ⚠️ 2026-08-27 追加发现(会改变结论写法,务必读)

### 1. 判分噪声极大:rice 两次判分有 47.8% 的叶节点翻转

对 rice 已有的两次判分(B 无 CodeRAG 0.4332 / B′ 有 CodeRAG 0.6047)逐叶节点 diff:

| 指标 | B | B′ |
|---|---|---|
| 原始通过条数 | 107/178(60.1%) | 114/178(64.0%) |
| 加权最终分 | 0.4332 | 0.6047 |

- **净增仅 7 条(+3.9pp),加权分却涨 17.1 分** —— 增益全压在少数高权重节点上,单点翻转被放大 4 倍多;
- **85/178 = 47.8% 的判断在两次之间改变**,其中 46 条新通过、**39 条新失败**。近乎对称的 churn 是裁判噪声特征(F1 仅 0.685),不是能力差异;
- 新通过与新失败的**条目主题高度重叠**(都是 "autonomous driving / StateMask / Random explanation / measuring cumulative reward" 那几族),同族标准两次判成相反结果。

**推论**:单轮单判的分数不可靠。三轮取均值是必须的,且**报告必须给出置信区间,不能报单点**。

### 2. "CodeRAG 有无"不是单变量,是四变量捆绑

`stage_b_driver.py` 的 `--fast` 走 `enable_indexing`,该开关一次性 gate 掉:

1. Phase 6 参考情报挖掘、2. Phase 7 仓库下载、3. Phase 8 索引构建,
4. **且 Phase 9 运行模式跟着换**——`code_implementation_workflow.py:416`:`False`→standard 模式(完整工具面);`True`→indexed 模式(仅 `write_file`+`search_code_references` 两工具 + 专用系统提示词)。日志里表现为 `code-implementation[standard]` vs `[indexed]`。

**推论**:此前记录的"CodeRAG 证实(+40% 相对)"应下调为**方向性支持、未证实**。真要摘出 CodeRAG,需两组都跑 indexed 模式、只让 `search_code_references` 返空做对照。

### 3. 产物结构差异是另一个混淆项

`archive_b4_no_coderag/generate_code`(B,无 CodeRAG)的包结构是**断的**:核心模块(`rnd.py`/`utils.py`/`mask_trainer.py`/`refiner.py`/`explanation.py`)在顶层 `rice/`,而包骨架 `RICE/rice/__init__.py` 与四个 env 文件在另一棵树;代码写的是 `from rice.X import ...`,解析不了。且少了 `experiment2_refining.py`(23 文件 vs B′ 的 24)。B′ 结构完全自洽。所以 B→B′ 的差里混着"布局修复+文件补全"。

### 4. DeepCode 实现是 Phase 0~10 共 11 阶段固定流水线

论文的"三阶段"是概念分组;实现见 `workflows/agent_orchestration_engine.py`。注意:**论文 Phase 3(静态分析+沙箱执行+LSP 修复)在这条主流水线里没有独立 phase**,即当前跑法根本没执行论文的验证环节。骨架固定但每 phase 内调 LLM、Phase 9 是真 agent 循环,故结构确定、内容随机。

### 5. fre 三轮的现场状态(截至 08-27 17:40)

- trial1 已跑;**trial2 失败退出(退出码 1,约 ¥2.90,见 `fre_trial2_console.log`)—— 死因未查,必须查**。若又是护栏误杀(单文件 600s 那条**至今未改**),幸存轮次会系统性偏乐观(失败的往往是最难的文件);
- trial3 进行中(09:17 起,Phase 9 写码),底座确认为 **DeepSeek-V4-Pro**,模式"完整(含参考挖掘+索引)"。**V4-Pro 已能跑通写码阶段,证明 maxTokens+护栏修复生效**;
- 此刻尚无任何 fre 的 grade.json。

### 6. 必做的加测:判分噪声基线

拿**同一份产物判两次**,量同一裁判下的分数漂移与叶节点翻转率。没有这个数,三轮均值的置信区间无从谈起,和论文 0.8435 的差距也无法区分"真差距"还是"仪器抖动"。成本约 ¥17~25。

## 产出要求

1. DeepCode+V4-Pro 的 fre 复现分(≥2 trial)+ Claude Code 锚点分,同裁判;
2. 用锚点偏移校正后,给出:V4-Pro 落在论文图 5 阶梯的档位、DeepCode/Claude Code 相对差距 vs 论文的 0.8435/0.6286≈1.34×;
3. **所有分数必须带噪声区间**(用第 6 条的基线),单点数字一律不作结论;
4. 明确回答:论文声称的效果在 fre 上是否成立、成立到什么程度、哪些环节存疑(裁判噪声、单篇样本、协议偏差、自我偏好偏差都要列)。

## 一个未解决的设计漏洞

V4-Pro **既当被测底座又当裁判**,存在自我偏好偏差:它会抬高 DeepCode 侧(V4-Pro 生成)而不抬高锚点侧(Sonnet 生成),使"DeepCode/锚点"比值虚高——**偏差方向正好指向"论文是对的"**,是最坏的一种。论文本身无此问题(o3-mini 判 Sonnet 产物)。缓解:两份产物各再用第二个裁判(如 Kimi-K2.7-Code)判一次,看比值是否稳定。
