# fre 验证 · 交接文档

> 2026-08-28 15:40 · 接手者读这一份即可上手
> 详细过程见 `FRE_VALIDATION_PLAN.md`;对外解释稿见 `EXPLANATION_PLAIN.md`(白话)与 `EXPLANATION_DRAFT.md`(精确)

---

## 0. 一句话现状

**五份全部判完,主结论已成立:同底座、同任务、同裁判下,DeepCode 脚手架未带来增益
——实测 0.98×,论文声称 1.34×。** 2×2 表已补齐。
剩下的只是加一个方差点(重跑 trial6)和回填对外稿。**当前无进程在跑。**

## 1. 已完成

### 1a. 全部有效判分(可直接引用)

| 提交 | 配置 | 得分 | 规模 |
| --- | --- | --- | --- |
| trial1 | DeepCode + V4-Pro | **0.5184** | 21 py / 6,751 行 |
| **bare_v4** | **裸 Claude Code / V4-Pro** | **0.4817** | 15 py / 3,070 行 |
| anchor | 裸 Claude Code / Sonnet 4.5 | **0.4839** | 15 py / 2,657 行 |
| trial3 | DeepCode + V4-Pro(900s 熔断) | **0.4378** | 15 py / 5,934 行 |
| **trial5** | **DeepCode + V4-Pro** | **0.4246** | 28 py / 8,438 行 |

**质量核验**:五份均 306 叶、`judge_type=simple`、`code_only=True`、无效叶 ≤1。
**论文对照(fre)**:Claude Code 0.6286 / Cursor 0.6344 / Codex 0.4095 / DeepCode(Sonnet4.5) 0.8435。

### 1b. 主结论:同模型对照(唯一变量=有没有脚手架)

|  | 裸跑 | + DeepCode |
| --- | --- | --- |
| Sonnet 4.5 | 0.4839 | ✗ 无 API |
| **V4-Pro** | **0.4817** | **0.5184 / 0.4246**(完整轮)· 0.4378(熔断轮) |

- 完整轮均值 **0.4715** ÷ 裸跑 0.4817 = **0.98×**(论文声称 1.34×)
- 最好的一轮 0.5184 ÷ 0.4817 = 1.076×,仍在噪声内
- **DeepCode+V4-Pro 轮间跨度 0.0938**(0.5184~0.4246),小于论文自报的 0.15,量级一致

**代码量与得分反相关**:trial5 写了 8,438 行(全场最多)却只有 0.4246,
低于 bare_v4 的 3,070 行 / 0.4817。**多写代码不等于多得分。**

### 1c. 两个底座裸跑的可比性(降级表述,勿说过头)

bare_v4 0.4817 与 anchor 0.4839 只差 0.0022,但**推不出"两个底座水平相同"**,因为:
任务书不同(官方原文 vs 压缩版 `CC_FRE_PROMPT.txt`)、推进方式不同(≤4 轮续跑 vs 单次)、
耗时差 5 倍(1~3h vs 22 分钟)、bare_v4 仅 n=1、且 306 个 Code-Dev 叶子可能存在量表压缩。

**可以说**:同一评分口径下两条裸跑基线都落在 0.48 附近,未观察到底座选择把分数推离该位置。
**不能说**:两个模型能力相同。主结论不依赖这条(它走的是同底座路径)。

### 1c. 作废轮次(不要拿去判分)

| 轮次 | 死因 | 花费 |
| --- | --- | --- |
| trial2 | 撞 4h 墙钟 | ¥2.90 |
| trial4 | `ReferenceAnalysisAgent` 只提名了 1 个仓库,语料贫瘠(**已摆卷,判分前必须移走**) | — |
| trial6 | 白天 API 限流,写码到 9/24 时三次重试打完放弃(`status=incomplete`,闸门已拦下未摆卷) | ¥19.08 |

## 2. 本轮新增的四个发现

### 2a. 裸 V4-Pro 审计全清,但该轮经过重启

按 `tool_use` 计数(**不是 grep 关键词**)的结果:

| 检查项 | 结果 |
| --- | --- |
| WebFetch / WebSearch | **0 / 0** —— 全程离线 |
| 黑名单串命中(全 transcript) | **0 处** |
| Read 调用 | 仅 4 次:prompt、paper.md、addendum.md、blacklist.txt |
| Bash 调用 | 仅 4 次:`ls`、`mkdir`、`wc -l`、`git init/commit` —— 无 curl/wget/clone |
| 提交物上游指纹 | 无 |
| paper.md | 472 行**完整读入,未截断** |

**须如实披露**:该任务前后开了 4 个会话(UTC 17:38–18:33),前 3 次废弃,其中一次因人工要求"先写计划"触发了 plan 模式。
最终产物**全部来自最后一次纯净会话**(会话 `30e0af18`,UTC 18:11–18:33,**耗时仅 22 分钟**,无 ExitPlanMode)——
现存 16 个文件 mtime 无一早于该会话起点,可证未混入早期产物。

### 2b. 白天跑显著更容易失败,真凶是模型 API 不是 GitHub

全部日志的网络类错误按小时分布,最大一坨在 **08 点:38 次**(14×429 限流、10×504、7×503、7×502),
全部来自 SiliconFlow 接口,**日志里一次 git clone 失败都没有**。

| 运行 | 时段 | 索引速率 |
| --- | --- | --- |
| trial3 | 白天 09:20~17:12 | ~77 秒/文件 |
| trial5 | 凌晨 02:57~07:14 | ~62 秒/文件 |

trial6 是直接证据:白天 15:14–15:24 连吃三次 180s 请求超时,整轮报废。
**结论:trial 应安排在凌晨跑。** 样本仅 10 次且各轮语料大小不同,只能说"倾向一致",未证实。

### 2c. 轮间方差的真实来源:参考仓库选择

两轮各自提名 5 个仓库并**全部克隆成功**(`github_download.txt` 均为 "All 5 repositories cloned successfully"):

```
trial5 → implicit_q_learning · D4RL · neural-processes · controllable_agent · CQL
trial6 → implicit_q_learning · D4RL · neural-processes · controllable_agent · exorl
```

**4 个重合,只有第 5 个不同**。同一个 D4RL 两轮建卡数几乎一致(15 vs 13),证明索引器行为稳定。
索引总量 trial5 245 卡 / trial6 121 卡,差异全在 CQL(136 卡)↔ exorl(16 卡)那一格。
这是 LLM 提名的温度抖动,**属于要测量的方差本身,不是缺陷**。

### 2d. 更正:trial4 不是网络故障

trial4 日志里 **119 次 API 调用全部 200、零下载错误**。它只挖到 1 个仓库是
`ReferenceAnalysisAgent` 自己只提名了 1 个,属模型决策差异。
先前"TLS 网络故障"的归因是错的,已从所有文档撤下。

## 2e. CodeRAG 预筛器静默失效(2026-08-29 新发现)

`code_indexer.py` 预筛 `max_tokens=2000` 对大仓库必然截断 → JSON 解析失败 →
**静默回退全量索引**,只打一条 INFO。A/B 实测(同一棵文件树,唯一变量是 max_tokens):

| | 2000(官方) | 16000(修复) |
| --- | --- | --- |
| tianshou(239 文件) | ❌ char 8576 截断 → **全量 239** | ✅ **筛出 13 个**,61 秒 |

fre 三轮有 **20~40%** 的仓库在失效状态下建卡。
**但已查证它不是 fre 低分主因**(规划先于索引,基线缺失是规划失败;
且失效率与分数不相关)——详见 `../fre/RESULTS.md` §3d。

修复已落地:默认值不变,`DEEPCODE_PREFILTER_MAX_TOKENS` 覆盖,`run_trial.sh` 注入 16000。
完整证据:`FINDING_prefilter_silent_failure.md`。
**口径影响:fre 三轮在未修复状态下测得,rice 将用修复版,两条线不可直接比绝对分数。**

## 3. 待办(按优先级)

### ⭐ A. 判分 bare_v4 + trial5

这是当前唯一能推进结论的动作。bare_v4 补齐 2×2 表的最后一格:

|  | 裸跑 | + DeepCode |
| --- | --- | --- |
| Sonnet 4.5 | 0.4839 ✅ | ✗ 无 API,跑不了 |
| **V4-Pro** | **← bare_v4,待判** | 0.5184 / 0.4378 ✅ |

**判分前必做的三件事**(顺序不能错):

1. 把 **已判过的 anchor / trial1 / trial3** 和 **作废的 trial4** 一起移出 `~/pb_submissions/fre/`
   —— 否则会重判、每份白烧 ¥38
2. 把 bare_v4 摆进去:`cp -r cc_dsv4_run/submission/. ~/pb_submissions/fre/bare_v4/`
3. 确认目录里**只剩 bare_v4 和 trial5 两份**

```bash
DRY=1 bash run_fre_grade.sh   # 先看报价
bash run_fre_grade.sh          # 真判分,约 ¥76(2 份 × ¥38)
```

脚本会自动数提交份数设 `n_tries`(**不设则只判一份,其余静默丢弃**)。

### B. 重跑 trial6(建议凌晨)

抗限流配置已就位(见 §6),**排在凌晨跑**:

```bash
TRIAL=trial6 nohup bash run_fre_trial.sh > fre_trial6_console.log 2>&1 &
```

约 5~6h、¥32。跑完检查 `code_base/` 是否有 5 个仓库、状态是否 `completed`。

目的:目前 DeepCode+V4-Pro 只有 trial1(0.5184)和 trial3(0.4378,熔断残缺)两个可用点,
trial5 判完是第三个。要谈"轮间方差"至少需要三个**完整**轮次。

### C. 回填对外稿

`EXPLANATION_PLAIN.md` 与 `EXPLANATION_DRAFT.md` 的结果段仍是旧数据,判分后回填。

## 4. 作弊审计(每轮产出后必做)

```bash
bash audit_anchor_blacklist.sh <transcript_dir>
```

⚠️ **必须以 `type=="tool_use"` 计数为准,不能用 grep 关键词** ——
系统提示里的工具清单会造成大量假阳性(锚点那次 grep 报 "WebFetch 6 次",实为 0 次)。
黑名单命中若来自"读 blacklist.txt 本身"或"论文摘要自带的仓库地址",属无害。
注:该脚本的 `SUB` 路径硬编码为 `anchor_fre/submission`,审计别的产物要手工改或直接跑等价检查。

## 5. 关键结论(写报告时直接引用)

1. **未复现论文的 1.34× 架构增益**:实测 0.99×,差距远小于噪声线 0.15(论文自身三次跑差 0.15)。
2. **锚点验证了裁判偏移真实存在**:我方锚点 0.4839 vs 论文同位置 0.6286,低 0.145;
   与 judge_eval 独立测得的"我方裁判偏严约 9pp"方向一致、量级吻合。
3. **裁判可信**:我方 DeepSeek 裁判 F1=0.685,o3-mini-high 在同 178 叶上 0.687±0.017,同一区间。
   但只测过一次,**只能说"不矛盾",不能说"水平相同"**;分数**只可同裁判内部相对比较**。
4. **数值校正不可行**:n=1 锚点偏移的 95% CI 达 ±0.499,且偏移非标量(章节间 I²=85.8%)。
5. **V4-Pro 不弱于 Sonnet 4.5**:SWE-bench Verified 80.6 vs 77.2(公开数据),
   **故"底座弱"不能作为低分借口**。
6. **口径天花板(双方共享)**:Code-Dev 只审代码不执行。论文全篇**零执行类实验、零案例研究**
   证明生成代码能跑出原结果,且**未将此列为局限**。PaperBench 本有三类维度
   (Code Development 306 / Code Execution 124 / Result Match 7 叶),**论文只用了第一类**。

## 6. 环境与闸门

- **模型配置**:`~/.deepcode/deepcode_config.json` 的 defaults 与 implementation 均为
  `deepseek-ai/DeepSeek-V4-Pro`(maxTokens 32768)。Kimi 版备份在同目录 `.bak_kimi_*`。
- **闸门**:总预算 **¥600**(已花 ~¥337.76);单轮 trial ¥100;脚本硬顶 14h;
  写码相位墙钟 4h;stall 阈值 1800s(`DEEPCODE_STALL_THRESHOLD` 可覆盖)。
- **抗限流(2026-08-28 新增)**:`run_fre_trial.sh` 注入
  `DEEPCODE_LLM_RETRY_MODE=persistent`、退避 `5,15,30,60,120`、上限 300s、同错容忍 30 次、
  请求超时 600s。**DeepCode 源码里的默认值保持官方原样,只加了 env 覆盖能力**。
- **git 反抄袭封锁**:`kvfrans/fre` 三条 insteadOf 已配,实测生效且不误伤同作者其他仓库。
- **URL 黑名单**:`DEEPCODE_URL_DENYLIST` 由 `run_fre_trial.sh` 从官方 blacklist.txt 自动注入。
- **监控脚本**:`scratchpad/fre_tick.sh`(`TRIAL=trialN bash fre_tick.sh`),
  配合 Monitor 每 15 分钟报点,进程结束时以 exit 9 自动停表。

## 7. 我方对上游的修改(被质疑时的答案)

**paperbench(判分侧)只改 2 个文件,均不碰评分逻辑**:
- `judge/simple.py`:二级解析模型 + 判分并发上限改为可配置(**默认值均保持官方原样**);
- `nano/eval.py`:split 白名单加 `fre`/`lite`(纯参数校验,顺带修好了官方自带却不可用的 lite.txt)。
- `grade.py`、`rubric/tasks.py`、`nano/entrypoint.py` **未修改**(git 可验)。

**DeepCode(被测侧)修的全是工程缺陷,核心方法零改动**:
补 7 个 MCP 配置、工具名连字符消毒(Kimi 无法调用带 `-` 的工具名)、
重写从未接线的下载 prompt、给下载/分段补落盘校验与 fail-fast、
补上论文声称有但代码里没有的 URL 黑名单拦截、
**把 DeepCode 自带但从未启用的 persistent 重试模式改为可用**(`core/providers/base.py`、
`core/compat/request_params.py`,均为 env 覆盖、默认值不变)。
规划/CodeRAG/写码循环的逻辑与提示词**一概未动**。
