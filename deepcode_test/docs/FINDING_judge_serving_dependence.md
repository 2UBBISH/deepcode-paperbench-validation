# 发现:结论依赖裁判的 serving —— 同名模型、两家服务商,rice 结论翻转

> 2026-09-03 · 全部 11 份有效提交用第二家服务商的同名裁判模型重判后的对照
> 原判分:SiliconFlow `deepseek-ai/DeepSeek-V4-Pro`;重判:Paratera `DeepSeek-V4-Pro`
> 其余一切相同:同一 PaperBench 版本、同一 rubric、`code_only=True`、同一批提交 tarball

---

## 一句话

**换一家服务商跑"同一个"裁判模型,fre 的结论不变,rice 的结论从"无增益(1.05×)"翻成"强增益(2.58×)"。**
我们没有依据判定哪个裁判更接近人工评分;在这一点被 JudgeEval 仲裁之前,
本项目此前所有"DeepCode 无增益"的结论都必须加上"在 SiliconFlow 裁判下"的限定。

## 双裁判对照(全部无效叶 = 0,均为有效判分)

### fre(306 叶)

| 提交 | SiliconFlow | Paratera | 差 |
| --- | --- | --- | --- |
| bare_v4(裸跑 V4-Pro) | 0.4817 | 0.4807 | −0.001 |
| anchor(裸跑 Sonnet 4.5) | 0.4839 | 0.5044 | +0.021 |
| trial1(DeepCode) | 0.5184 | 0.4682 | −0.050 |
| trial5(DeepCode) | 0.4246 | 0.3101 | −0.115 |
| **DeepCode / 裸跑** | **0.98×** | **0.81×** | 两裁判一致:无增益 |

### rice(178 叶)

| 提交 | SiliconFlow | Paratera | 差 |
| --- | --- | --- | --- |
| bare_v4(裸跑 V4-Pro) | 0.4680 | **0.1452** | **−0.323** |
| trial2(DeepCode) | 0.5447 | 0.4033 | −0.141 |
| trial3(DeepCode) | 0.4374 | 0.3446 | −0.093 |
| **DeepCode / 裸跑(V4-Pro)** | **1.05×** | **2.58×** | **结论翻转** |
| bare_kimi(裸跑 Kimi-K2.7) | 0.4633 | **0.1865** | **−0.277** |
| trial_k1(DeepCode+Kimi) | 0.2815 | 0.2403 | −0.041 |
| trial_k2(DeepCode+Kimi) | 0.4760 | 0.3754 | −0.101 |
| **DeepCode / 裸跑(Kimi)** | **0.82×** | **1.65×** | **结论翻转** |

论文声称 rice 上 1.95×。Paratera 裁判下我们"复现"了它;SiliconFlow 裁判下没有。

## 诊断:不是没看到文件,是判得更严

以 rice bare_v4 为例(0.4680 → 0.1452):

| 指标 | SiliconFlow | Paratera |
| --- | --- | --- |
| 每叶输入 token | 60,757 | 62,852 |
| 每叶输出 token | 2,937 | **1,267** |
| 零分叶 | 87 | 131 |
| 疑似"没看到文件"的零分叶 | 39 | 35 |
| 叶子翻转 | — | **45 个 1→0,1 个 0→1** |

输入几乎相同 → 两个裁判看到的是同样的代码;输出减半 → 推理量不同;
"没看到文件"类零分反而更少 → 不是判分侧文件选择失败(`rice/RESULTS.md` §4d 那种)。

翻转叶的解释针对**同一份代码**给出相反判断:

> **SiliconFlow(1 分)**:"`DiscreteMlpPolicy` 可配置为 hidden_sizes=(128,128,128,128),满足四层 MLP 要求"
> **Paratera(0 分)**:"只提供通用 MLP 策略类,没有按要求实例化自私挖矿 PPO agent,不满足"

即:**对"通用可配置实现"是否算"已实现",两个 serving 的判断标准不同。**
rice 的裸跑代码大量采用"抽象回调 + 可配置类"的写法(其 README 明言跳过环境特定实现),
在严格裁判下被成片判零;DeepCode 的产物按环境铺了具体文件,受影响小。
fre 两条线的写法差异没这么极端,所以 fre 结论稳定。

## 为什么"同一个模型"会不一样

两家的 API 都返回 `reasoning_content`,都能正确回答简单的判断题。可观察到的差异:

- 同一条 145 token 的消息,两家报的 `prompt_tokens` 分别为 145 与 66 —— **对话模板/分词处理不同**
- 判分时的输出 token 相差一倍 —— **推理深度或采样设置不同**
- 无法从外部确认权重版本、量化方式、系统提示是否一致

"DeepSeek-V4-Pro"这个名字并不能保证判分行为一致。**裁判恒定应以 serving 为单位,不能以模型名为单位。**

## 这对本项目意味着什么

1. **fre 的"无增益"结论稳健**:两个裁判一致(0.98× / 0.81×)。
2. **rice 的结论不稳健**:1.05× 与 2.58× 之间,取决于裁判 serving。此前写在 CONCLUSIONS 里的
   "rice 无增益"必须降级为"在 SiliconFlow 裁判下无增益"。
3. **"四条裸跑基线聚在 0.46~0.48"这条观察在 Paratera 裁判下不成立**(rice 裸跑掉到 0.15~0.19)。
4. **谁对 —— JudgeEval 仲裁(2026-09-03 已跑,rice/0,178 叶,code_only)**:

   | 裁判 serving | 准确率 | macro F1 | 偏向 |
   | --- | --- | --- | --- |
   | SiliconFlow(08-26) | 0.685 | 0.685 | 严 9.0 pp |
   | Paratera(09-03) | 0.719 | 0.719 | 严 9.0 pp |

   两者通过率完全相同(0.4494)、净偏向完全相同,差别只在**哪些叶子判错**:
   同一份提交上两裁判 **84.3% 一致、28 叶分歧**;都对 111、只 SiliconFlow 对 11、只 Paratera 对 17、都错 39。
   Paratera 多对 6 叶,在 n=178 下落在抽样噪声内(标准误约 3.4 pp),**不能据此说它更可信**。
   更关键的限制:JudgeEval 的 rice/0 是**作者官方仓库**(具体实现完整),并不覆盖两裁判真正分歧的
   "通用可配置写法是否算已实现"这一区间。**结论:JudgeEval 无法仲裁 rice 的翻转;两裁判在人工标注上同等水平
   (≈ gpt-4o 级),彼此却有 16% 的叶级分歧 —— 这就是 PaperBench Code-Dev 在这类模型上的裁判噪声底。**
5. **对论文的含义**:论文用 Sonnet 4.5 裁判;我们两家 V4-Pro serving 之间的分歧,说明
   PaperBench Code-Dev 的绝对分数对裁判实现高度敏感,**跨论文、跨团队比较倍数时必须报告裁判 serving**。

## 数据位置

- 重判 JSON:`fre/grades/paratera_<name>_<score>.grade.json`、`rice/grades/paratera_*.grade.json`
- 原判 JSON:同目录下按实例 id 命名的文件(fre 见 `fre/RESULTS.md`,rice 见 `rice/RESULTS.md`)
- 判分原始输出:`frontier-evals/project/paperbench/runs/2026-09-02T23-45-29-GMT_*`(fre)、`2026-09-03T01-29-00-GMT_*`(rice)
