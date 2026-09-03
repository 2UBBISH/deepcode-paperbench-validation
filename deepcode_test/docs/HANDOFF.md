# DeepCode × PaperBench 交接文档

> 2026-08-26 · 写给接手者(未来的自己 / 接棒模型 / 想复现的人)
> 配套:`PAPERBENCH_RUNBOOK.md`(13 坑修法)· `EXECUTION_PLAN.md`(台账)· 验证战报(artifact 5d699011)
> **本文档的存在理由**:战报给结论,这里给"结论是怎么拼出来的"——特别是 B′=60.5 的真实身世。

---

## 0. 一句话现状

rice 单篇 Code-Dev 三分对照:白卷 0 / 无 CodeRAG 43.3 / 有 CodeRAG **60.5**(裁判恒定,F1 0.685);
花费 ≈¥170;13 项修补全部留痕;**无任何进程在跑**。

---

## 1. ⚠️ B′ = 60.5 的真实身世(接手前必读)

**60.5 不是一次端到端跑出来的**。它是跨 8 轮攻坚、复用多轮(含失败轮)中间产物、
中途两次切换模型、参考语料两次人工精简之后的**拼装产物**。逐件溯源(时间戳实证):

### 1.1 产物溯源表

| # | 产物 | 生成时间/轮次 | 生成者 | 备注 |
| --- | --- | --- | --- | --- |
| 1 | `paper.md`(输入) | 00:10 · B 误启轮 | 纯代码转换 | 无模型参与 |
| 2 | `initial_plan.txt`(23KB 施工方案) | 02:12 · **B1(失败轮)** | **DeepSeek-V4-Pro** | ⭐ B 与 B′ 的所有轮次共享同一份——对照有效性的正面因素 |
| 3 | `document_segments/` | 02:15 · B2(失败轮) | DeepSeek-V4-Pro | 分段+目录 |
| 4 | `reference.txt`(20KB 参考报告) | 10:46 · B′-2(失败轮) | DeepSeek-V4-Pro | 修完坑9(迭代 8→40)后产出;SB3 排名第一 |
| 5 | `code_base/` 参考仓库 | 11:05~11:0x · B′-3/5(失败轮) | 编排 DeepSeek;克隆是 git | agent 自主下了 5 个仓库;**曾自主选中论文官方仓库(禁抄名单),被 git 层封锁** |
| 5a | ——人工精简 #1 | 11:41 | **人工(我)** | SB3 内部:删 tests/docs/无关算法,239→(SB3 50) |
| 5b | ——人工精简 #2 | 12:21 | **人工(我)** | 全局:删 metadrive(325MB)/tianshou(58MB),softlearning 裁到算法核心;全局 ≈93 文件 |
| 6 | `indexes/` 三份索引卡 | 13:51/13:55/14:18 · **B′-7(失败轮)** | **Kimi-K2.7-Code** | 精简语料上构建;2 个文件关系抽取 JSON 解析失败被跳过;B′-6 建的版本被无幂等重建覆盖,不在最终链里 |
| 7 | `generate_code/` 24 文件 | 14:42~15:16 · **B′-8(成功轮)** | **Kimi-K2.7-Code** | indexed 模式(2 工具);`search_code_references` 仅调用 **4 次** |
| 8 | 判分 grade.json | 15:16+ | DeepSeek-V4-Pro 裁判 | 全程未换,与 A/B/JudgeEval 同一测量仪器 |

### 1.2 模型混用时间线(为什么产物链是"双模型混血")

```
02:00  agents.defaults = DeepSeek-V4-Pro     ← 规划(02:12)/分段(02:15)/参考挖掘(10:46)/下载编排 都是它
02:52  agents.implementation → Kimi-K2.7-Code ← 写码相位专用(坑8修复)
11:41  agents.defaults → Kimi-K2.7-Code       ← 为提速索引;此后默认相位全是 Kimi(索引 13:51+/记忆摘要)
不变   裁判 + 二级解析器 = DeepSeek-V4-Pro    ← .env,从未动过(可比性的锚)
```

即:**"理解与规划"侧(方案/分段/参考)是 DeepSeek 的产物,"构建"侧(索引/写码)是 Kimi 的产物**。
这不是设计,是修坑过程的沉积——接手者若要复现,须知道这一层。

### 1.3 这对结论意味着什么

**站得住的**(战报的核心对照不受损):
- B(43.3)与 B′(60.5)共享同一份方案(#2)、同一写码模型(Kimi)、同一裁判——B→B′ 的差异仍干净地隔离在"CodeRAG 有无"上;
- 裁判恒定链条(A/B/B′/JudgeEval 同脑)完好。

**必须打折的**:
- ❌ "DeepCode 能**自主端到端**跑出 60.5" —— 不成立。成立的是:"**修复 13 处并人工辅助策展后**的 CodeRAG 流水线,其产物得 60.5";
- ❌ "CodeRAG 索引是全自动建立的" —— 语料经两次人工精简(裁剪理由与站得住程度见战报 05 节,metadrive 一刀最弱);
- ⚠️ 检索仅被使用 4 次/24 文件——B′ 的提升是"索引模式整体"(检索+专用提示词+2 工具面)的组合效应;
- ⚠️ **修补全就位后的"一次干净端到端"从未被验证过**(所有成功产物都来自拼装)——这是当前最大的已知空白,见 §4 待办第一条。

---

## 2. 系统当前状态

### 2.1 关键配置(改过的三处)

`~/.deepcode/deepcode_config.json`:
- `agents.defaults`:provider=siliconflow,model=**Kimi-K2.7-Code**,maxTokens=32768(注意:已不是 DeepSeek!若想恢复"规划用 DeepSeek",把 defaults.model 改回 `deepseek-ai/DeepSeek-V4-Pro`,implementation 相位保持 Kimi)
- `agents.implementation`:Kimi-K2.7-Code / 32768 / temp 0.2
- `tools.mcpServers` ×6:code-implementation、code-reference-indexer、document-segmentation(均 `python -m tools.xxx`)、filesystem(npx)、fetch(uvx)、github-downloader(`python -m tools.git_command`)

`frontier-evals/project/paperbench/.env`:硅基流动 key + base_url + `PB_STRUCTURED_PARSER_MODEL`。

**git 全局配置里有反作弊封锁**(url.insteadOf 把 `github.com/chengzelei/RICE` 重写到假地址)——
换论文验证时需按新论文的 `blacklist.txt` 增删;做别的项目若需克隆该仓库,记得
`git config --global --unset-all url.https://blocked.invalid/blacklisted.insteadof` 类似清理。

### 2.2 环境注意

- **DeepCode venv 是 Python 3.11,低于官方 3.12 下限**(坑11 我方误判的根源)。目前带补丁能跑;
  正规化待办:`cd DeepCode && rm -rf .venv && uv venv --python 3.12 && uv pip install -r requirements.txt`(约 5 分钟);
- 源码补丁共 5 处(全带 `[local compat]` 注释,`git -C DeepCode diff` / `git -C frontier-evals diff` 可查):
  ctx 表 +1 行、解析器环境变量化、迭代配额 ×2、LoopDetector 写豁免、stall 900s、索引幂等跳过。

### 2.3 目录地图

```
~/deepevol/
├── DeepCode/deepcode_lab/tasks/paper_e8af8afa/   ← B′ 全部中间产物(溯源表实体)
├── archive_b4_no_coderag/                        ← B(43.3)产物存档(对照组)
├── frontier-evals/project/paperbench/runs/       ← 四组判分树(A/B/B′/JudgeEval)
├── 图解/                                          ← 三张整页 PNG
├── bootstrap.sh · run_stage_b.sh · stage_b_driver.py
└── PAPERBENCH_RUNBOOK.md · EXECUTION_PLAN.md · HANDOFF.md(本文)
```

---

## 3. 如何复现 / 继续

### 3.1 🥇 待办第一条:验证"干净端到端"(≈¥25+37,约 2 小时)

13 项修补就位后,理论上一次跑通,但**从未实证**。做法:

```bash
mv ~/deepevol/DeepCode/deepcode_lab/tasks/paper_e8af8afa ~/deepevol/archive_bprime_frankenstein
cd ~/deepevol && STAGE_B_INPUT=~/deepevol/frontier-evals/project/paperbench/data/papers/rice/paper.md \
  nohup bash run_stage_b.sh > stageB_clean_e2e.log 2>&1 &
```

预期:参考挖掘(~10min)→ 下载(受封锁约束)→ 索引(**无人工裁剪**,agent 下多少建多少——
可能重现 4 小时问题;若接受人工裁剪,复刻 §1.1 的 5a/5b 两刀)→ 写码 → 判分。
产出的分数与 60.5 对比,即可回答"拼装是否引入了偏差"。

### 3.2 其他可选后续(按价值排序)

1. **×3 重复**(≈¥120):B 与 B′ 各再跑 2 次取均值——治"实验 III 0.67→0.11"级别的单次方差;
2. **C · GPU 全维复跑**(≈¥60,风扇 2 小时):补 Execution/Result 维度,命令在 EXECUTION_PLAN §4;
3. **20 篇全量**(判分 ≈¥740 + 考生):才有资格与官方榜并列;
4. CodeMem 消融复现(论文另一主张,未检验)。

---

## 4. 遗留问题清单(诚实版)

| 问题 | 状态 |
| --- | --- |
| 干净端到端从未单次验证 | ❌ 最大空白,见 §3.1 |
| DeepCode venv 3.11 < 官方 3.12 下限 | ⚠️ 带补丁可跑,待重建 |
| 索引器串行慢(55s/文件)+ 不跳测试文件 | ⚠️ 靠人工裁剪绕过,未根治 |
| Kimi 建索引偶发坏 JSON(2 文件关系抽取被跳过) | ⚠️ 已知损耗,未修 |
| 产物目录分裂(rice/ 与 RICE/ 两根)| ⚠️ B 与 B′ 同病,不影响对照 |
| B′ 判分仅 1 次 | ⚠️ 单次方差未消 |
| Execution / Result Match 维度 | ❌ 完全未测(C 阶段) |
