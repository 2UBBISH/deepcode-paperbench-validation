# 全部测试结果总表

> 2026-09-03。每一份提交、每一个裁判、每一轮作废,一处查全。判分 JSON 在 `fre/grades/`、`rice/grades/`(Paratera 版文件名带 `paratera_` 前缀);维度级分析在 `fre/RESULTS.md`、`rice/RESULTS.md`;机制在 `docs/CONCLUSIONS.md`。

## 1. 有效提交(11 份)× 两个裁判 serving

裁判均为 DeepSeek-V4-Pro,`code_only=True`,无效叶全部为 0。SF = SiliconFlow `deepseek-ai/DeepSeek-V4-Pro`;PT = Paratera `DeepSeek-V4-Pro`。

### fre(306 叶)

| 提交 | 臂 | 规模 | 耗时 | SF | PT | 失分要点(SF) |
| --- | --- | --- | --- | --- | --- | --- |
| anchor | Claude Code + Sonnet 4.5 裸跑 | 15 py / 2,657 行 | ~1h | 0.4839 | 0.5044 | 数据集/环境 0.333 弱;方法 0.792、基线 GC-IQL/GC-BC 满分、OPAL 0 |
| bare_v4 | Claude Code + V4-Pro 裸跑 | 15 py / 3,070 行 | ~5h | 0.4817 | 0.4807 | 主方法 0.815、三基线 0.67/0.80/0.94;数据集/环境 0.417 |
| trial1 | DeepCode + V4-Pro | 21 py / 6,751 行 | ~4h | 0.5184 | 0.4682 | 数据集/环境 0.833 最高;**GC-IQL/GC-BC/OPAL 全 0**(文件不存在) |
| trial5 | DeepCode + V4-Pro | 28 py / 8,438 行 | ~5h | 0.4246 | 0.3101 | 写得最多分最低;三基线全 0;主方法 0.685 |
| **DeepCode / 裸跑(V4-Pro)** | | | | **0.98×** | **0.81×** | 两裁判一致:无增益 |

### rice(178 叶)

| 提交 | 臂 | 规模 | 耗时 | SF | PT | 失分要点(SF) |
| --- | --- | --- | --- | --- | --- | --- |
| bare_v4 | Claude Code + V4-Pro 裸跑 | 11 py / 4,173 行 | 凌晨单次 | 0.4680 | **0.1452** | 环境搭建 0.000(9 个环境一个没写);实验 II(w4)0.750 全场最高;PT 下 45 叶 1→0(抽象可配置写法被判未实现) |
| trial2 | DeepCode + V4-Pro | 22 py / 12,583 行 | 5h21m,¥31 | 0.5447 | 0.4033 | 环境 0.389、解释方法 0.821、实验 III 0.833;实验 II 0.583 |
| trial3 | DeepCode + V4-Pro | 39 py / 25,630 行 | 6h46m,¥45 | 0.4374 | 0.3446 | 环境 0.575、策略网络 0.875;实验 II 0.396,其中 7 叶为判分侧"空文件"零分;基线均值 0.29 |
| **DeepCode / 裸跑(V4-Pro)** | | | | **1.05×** | **2.58×** | 结论随裁判翻转;2.58× 来自裸跑塌方(−0.32)而非 DeepCode 上涨(−0.14/−0.09) |
| bare_kimi | Claude Code + Kimi-K2.7-Code 裸跑 | 18 py / 2,463 行 | 22 分钟 | 0.4633 | 0.1865 | 与裸跑 V4-Pro 几乎重合 |
| trial_k1 | DeepCode + Kimi | 36 py / 12,056 行 | 3h02m | 0.2815 | 0.2403 | 实验逻辑还原不足,方差大 |
| trial_k2 | DeepCode + Kimi | 32 py / 8,266 行 | 4h15m | 0.4760 | 0.3754 | |
| **DeepCode / 裸跑(Kimi)** | | | | **0.82×** | **1.65×** | 同样随裁判翻转 |

论文声称:fre 1.34×(Claude Code 0.6286 vs DeepCode 0.8435)、rice 1.95×(Claude Code 0.3787 / Cursor 0.4186 / Codex 0.3645 vs DeepCode 0.7380)。

## 2. 作废轮(不入统计)

| 轮 | 配置 | 原因 | 沉没 |
| --- | --- | --- | --- |
| fre trial2 | DeepCode + V4-Pro | 撞 4h 写码墙钟 | ¥2.90 |
| fre trial3 | DeepCode + V4-Pro | 写码被 900s stall 熔断,残缺(判了 0.4378,只作参考) | 已判 |
| fre trial4 | DeepCode + V4-Pro | 语料仅 1 仓库(下载侧问题;报告本身完整 5 条) | — |
| fre trial6 | DeepCode + V4-Pro | 白天 API 限流,写到 9/24 三次重试打完,`status=incomplete` | ¥19.08 |
| rice trial1 ×5 | DeepCode + V4-Pro | ① 仓库克隆 TLS 断流(外部 GitHub 争用)② 换网络节点 ③ 预筛静默回退全量索引(140h)④ 假计划 ⑤ stall 1800s 熔断 / 退出码 2 | ≈¥115 |
| rice trial2(首次) | DeepCode + V4-Pro | 随 trial1 一起因 stall 阈值作废 | ¥15.29 |
| fre trial_fx1 首跑 / 二跑 | DeepCode + V4-Pro(Paratera)+ 修复①②③ | 挖掘报告截断只见 1 仓库;调大上限重启 | ≈¥3 |
| **fre trial_fx1** | DeepCode + V4-Pro(Paratera)+ 修复①②③④ | **完整跑完、PT 0.3618**;三基线文件齐备(210/336/351 行),但 FRE 主方法 0.82→0.26;整体作废:提示词含评分元知识 | ¥22 |
| **fre trial_fx2** | 同上 | **完整跑完、PT 0.4873**;主方法 0.70;同因作废 | ¥21 |
| 一次判分批 | rice 三份 | 余额耗尽,161~168/178 叶无效 | ≈¥76 |
| 一次判分批 | fre 六份(Paratera) | 裸模型名未登记上下文表,6 份 64 秒全失败 | ≈0 |

## 3. 修复轮的维度证据(虽作废,机制观察成立)

| 细目(权重) | bare_v4 | fx1 | fx2 | trial1 | trial5 |
| --- | --- | --- | --- | --- | --- |
| FRE 主模型 (3) | 0.981 | **0.259** | 0.704 | 0.824 | 0.769 |
| GC-IQL (1) | 1.000 | 0.667 | 1.000 | 0.000 | 0.333 |
| GC-BC (1) | 1.000 | 0.600 | 0.200 | 0.000 | 0.000 |
| OPAL (1) | 0.944 | 0.222 | 0.556 | 0.000 | 0.000 |

(全部 Paratera 裁判。)基线补上了,主方法塌了;覆盖审计两轮均 `ran + adopted`,计划 16,470→19,534 / 16,257→19,231 字符。

## 4. 裁判校准(JudgeEval,rice/0 作者官方仓库,178 叶,code_only)

| 裁判 serving | 准确率 | macro F1 | 通过率 | 偏向 | 花费 |
| --- | --- | --- | --- | --- | --- |
| SiliconFlow(08-26) | 0.685 | 0.685 | 0.4494 | 严 9.0 pp | ¥27.7 |
| Paratera(09-03) | 0.719 | 0.719 | 0.4494 | 严 9.0 pp | ¥28 |

同一提交上两裁判一致 150/178(84.3%);分歧 28:都对 111、只 SF 对 11、只 PT 对 17、都错 39。官方对照(5 卷宏平均 Code-Dev):o1-high 0.740 / o3-mini 0.720 / gpt-4o 0.681 / gpt-4o-mini 0.588。

## 5. 判分侧诊断数据(rice bare_v4,SF vs PT)

| 指标 | SF | PT |
| --- | --- | --- |
| 每叶输入 token | 60,757 | 62,852 |
| 每叶输出 token | 2,937 | 1,267 |
| 零分叶 | 87 | 131 |
| 疑似"没看到文件"零分叶 | 39 | 35 |
| 叶子翻转 | — | 45 个 1→0,1 个 0→1 |

## 6. 编译检查(语法层人人及格)

fre trial1/trial5/bare_v4/anchor、rice trial2/trial3/bare_v4/trial_k1 全部 `py_compile` 通过;trial_k2 31/32(1 个语法错误文件)。可运行性不构成任何一方的优势。

## 7. 试点期(不入正式统计)

| 轮 | 配置 | 分 |
| --- | --- | --- |
| 阶段 A | dummy 白卷 + 真裁判 | 0.000(预期) |
| 阶段 B | DeepCode(规划 DeepSeek + 写码 Kimi,无 CodeRAG) | 43.3 |
| 阶段 B′ | 同上 + CodeRAG(8 轮拼装,人工裁剪语料) | 60.5 |
| E1 干净端到端 | 全 Kimi | 14 分钟塌方(D1–D5) |
