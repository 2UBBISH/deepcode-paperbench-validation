# ☀️ 晨报 · 2026-08-26 · 阶段 B 完成

> **TL;DR:DeepCode 拿到 43.3 分(107/178 条评分点通过)。**
> 对照:白卷基线 0 分;官方 PaperBench Code-Dev 榜首(IterativeAgent o1-high)43.4 分。
> 途中修掉 4 个新坑,其中一个是"换模型立竿见影"级别的关键发现。
> 全程花费 ≈¥86,在 ¥200 闸门内。**当前系统全静,没有东西在跑、在花钱。**

---

## 一、成绩单

| 对象 | 得分 | 说明 |
| --- | --- | --- |
| 空白卷(阶段 A 基线) | **0.0%** | 验证裁判不放水 |
| **DeepCode(阶段 B)** | **43.3%** | 107/178 条通过,178 条全部有效判分、0 失败 |
| 官方 Code-Dev 榜首(参考) | 43.4% | ⚠️ 见下方"不能直接对标"说明 |

**产物**:23 个文件、**7269 行 Python 代码**,流水线状态 `completed`(计划内文件全部写完)。

**⚠️ 为什么不能说"追平 o1"**:官方那 43.4 是 **20 篇论文的平均分**、用 o3-mini 当裁判、跑 3 次取均值;我们是 **1 篇论文(rice)、DeepSeek-V4-Pro 当裁判、单次**。诚实的说法是:**落在同一量级**,不是同一口径的对标。

### 得分结构(按 rubric 一级子树)

| 子树 | 得分 | 权重 |
| --- | --- | --- |
| 环境搭建(MuJoCo 等) | **0.82** | 1 |
| 解释方法实现(StateMask/RND) | **0.83** | 2 |
| 实验 III 复现 | 0.67 | 2 |
| PPO 策略网络 | 0.38 | 1 |
| 实验 I 复现 | 0.38 | 3 |
| 实验 IV 复现 | 0.33 | 2 |
| 精炼方法实现 | 0.31 | 2 |
| **实验 II 复现** | **0.19** | **4(权重最高)** |

**读法**:DeepCode 把**论文的核心算法写得相当好**(解释方法 0.83、环境 0.82),但**实验复现的广度和细节掉分严重**——尤其权重最高的实验 II 只拿 0.19,直接拖低总分。

失分的典型模式(裁判原话摘录):
- **整块缺失**:"submission contains no files or code, providing no evidence of HalfCheetah environment implementation"(论文要求 4 个 MuJoCo 环境,它只做了 2 个)
- **细节遗漏**:"lacks observation normalization for Walker2d-v3 as required by the paper"
- **特定依赖没做**:"fails to correctly implement the MetaDrive Macro-v1 environment"

得分的典型(说明裁判是真读代码):
- ✓ "correctly implements the Hopper environment as 'Hopper-v3'. The code uses 'Hopper-v3' exclusively (in mujoco_envs.py)..."

---

## 二、跑了 4 轮才成功,每轮修掉一个真问题

| 轮 | 结果 | 根因 | 修法 |
| --- | --- | --- | --- |
| 1 | 全程无工具,参考分析吐 475 字节垃圾 | **`deepcode init` 没写自己流水线的 MCP 服务器配置** | 补 3 个服务器(必须 `python -m tools.xxx` 模块方式启动) |
| 2 | 写 3 个文件熔断,路径 `rice/` 与 `RICE/rice/` 打架 | **模式不对称**:完整模式只给 2 个工具,模型看不见自己写过什么 | 改用 fast 模式(11 个工具) |
| 3 | 仍卡在 3 个文件 | **模型太慢+token 上限太低**(见下) | 换模型 + 提上限 |
| **4** | **✅ 23 文件全部写完** | — | — |

### 🔑 第 3 轮的关键发现(最值钱的一条)

同一道题("写 80 行 RND 模块")实测:

| 模型 | 耗时 | 输出 token | 结果 |
| --- | --- | --- | --- |
| deepseek-ai/DeepSeek-V4-Pro | **225 秒** | 12312(绝大部分烧在推理) | 默认 8192 上限下**必然截断** |
| moonshotai/Kimi-K2.7-Code | **29.6 秒** | 2236 | 干净完整 |

**快 7.6 倍**。默认 `maxTokens=8192` 装不下"推理 + 代码",于是连续返回空响应/截断输出,300 秒无进展被熔断——现象链是
`Empty response on turn N` → `Output truncated` → `🐌 Progress stall`。

**修法**(已写进配置):
```json
"agents": {
  "defaults":        {"provider":"siliconflow","model":"deepseek-ai/DeepSeek-V4-Pro","maxTokens":32768},
  "implementation":  {"model":"moonshotai/Kimi-K2.7-Code","maxTokens":32768,"temperature":0.2}
}
```
规划相位留给 DeepSeek(慢但方案质量好,产出的 23KB 方案很扎实),写码相位交给 Kimi。

---

## 三、成本实测(人民币,硅基流动)

| 项目 | 金额 |
| --- | --- |
| 阶段 A 判分(白卷基线) | 17.2 |
| A2 失败跑(解析失败,费用照产生) | ≈15 |
| B 第 1–3 轮(50 次调用,DeepSeek) | ≈8 |
| B 第 4 轮实现(119 次调用,Kimi) | ≈9 |
| **阶段 B 判分**(in 10.88M / out 0.74M) | **37.1** |
| 各类冒烟测试 | <1 |
| **累计** | **≈¥86** / 上限 ¥200 |

**成本结构结论**:**DeepCode 跑一整轮只要 ¥9,判一次分要 ¥37**(178 条判分每条都携带全文论文 + 7269 行代码)。
所以最经济的策略是:**反复调 DeepCode 直到满意,只在最后判一次分**——我全程按这个原则,拦下了对两次 3 文件残品的判分,省了约 ¥50。

---

## 四、必须说明的三个保留意见

1. **CodeRAG 全程没真正生效**。DeepCode 的招牌能力之一是"挖参考实现 → 建索引 → 写码时查阅",但这需要网页搜索后端,我们没有。参考分析阶段的 agent 一直在说"请把论文内容给我"。我已补上 `filesystem` + `fetch` 两个 MCP 服务器(实测连通,14+2 个工具),但**本次成绩是在无 CodeRAG 状态下取得的**——换句话说,**43.3 分是 DeepCode 的"减配成绩"**。
2. **产物结构不干净**:核心模块写在 `rice/`,而 README、配置、实验脚本、测试写在 `RICE/`,分裂成两个根目录。计划文件虽然都写全了,但一个真人接手会觉得别扭。
3. **只判了 Code Development 一个维度**(code_only 模式)。"代码能不能真跑起来""跑出的数字对不对得上"这两类维度(Execution / Result Match)完全没测——那是阶段 C 的事。

---

## 五、醒来后的选项

| 选项 | 干什么 | 预算 | 噪音 |
| --- | --- | --- | --- |
| **C** 全量复跑 | 把这 23 个文件在 GPU 容器里真执行 + 361 条全维判分 | ≈¥40~60 | **风扇高转最长 2h** |
| **B′** 补 CodeRAG 重跑 | 用刚修好的 filesystem/fetch 再跑一次,看 43.3 能提多少 | ≈¥15(实现)+¥37(判分) | 安静 |
| **C′** SWE-bench 线 | 换执行级验证(改真实 GitHub bug → 跑单元测试),无 GPU | 先免费侦察 | 安静 |
| **读判分** | 我带你逐条翻 178 条判词,看裁判怎么想的 | ¥0 | 无 |

**推荐顺序**:先"读判分"(免费,能看清 DeepCode 到底强在哪弱在哪)→ 再决定 B′(公平重测)还是 C(执行验证)。

---

## 六、系统状态与改动清单

- **无进程、无容器在跑,不产生任何费用**;
- 结果落盘:`frontier-evals/project/paperbench/runs/2026-08-25T19-29-35-GMT_run-group_direct_submission_solver/rice_*/grade.json`;
- 代码产物:`DeepCode/deepcode_lab/tasks/paper_e8af8afa/generate_code/`(23 文件)与 `~/pb_submissions/rice/submission/`;
- 本轮新增改动:`~/.deepcode/deepcode_config.json`(MCP 服务器 ×5、agents 相位模型/token)、`stage_b_driver.py`(加断点续跑入口);
- 两个上游仓库的**源码改动仍只有此前那 2 处最小补丁**(都带 `[local compat]` 注释);
- 八个坑的完整修法已写进 `PAPERBENCH_RUNBOOK.md`,花费台账在 `EXECUTION_PLAN.md`。
