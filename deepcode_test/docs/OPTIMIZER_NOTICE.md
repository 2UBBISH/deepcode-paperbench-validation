# 如果你要在这个仓库上跑自优化循环,先读这份

> 2026-09-03 · 面向想把本仓库当作自进化 / 自优化算法(AutoSOTA 类)迭代目标的人
> 一句话:**机制上闭环可跑,但默认的目标函数是噪声,直接优化会得到一条漂亮且错误的曲线。**

---

## 1. 闭环确实是通的

```
setup.sh                              # 环境(已验证三轮干净克隆)
  → 改 DeepCode/ 里的任何东西
  → PAPER=fre TRIAL=xxx bash deepcode_test/scripts/run_trial.sh    # 3~6h, ≈¥20
  → PAPER=fre bash deepcode_test/scripts/run_grade.sh              # 40~100min, ≈¥38/份
  → grade.json 里的 score
```

可改的东西、反馈回路、结果参照都在。问题只在"结果参照能不能承受被优化"。

---

## 2. 为什么不能拿 PaperBench 裁判分当目标函数

### 2.1 实测的信噪比

| 量 | 数值 | 来源 |
| --- | --- | --- |
| 同组轮间跨度(σ) | 0.094 / 0.107 / 0.158 | fre trial1-trial5、rice trial2-trial3、Kimi 线 |
| 真实组间效应 | 0.010 / 0.023 | 裸跑 vs DeepCode 均值差 |
| 同名裁判换 serving 的叶级分歧 | **15.7%**(28/178) | JudgeEval rice/0 双 serving |
| 两个 serving 在人工标注上的准确率 | 0.685 / 0.719 | 同等水平,**无法仲裁谁对** |

**噪声比被优化的效应大 4.6~9 倍。**

### 2.2 后果:零效应也能刷出 +30%

用实测 σ=0.1 模拟一个**改动完全无效**的优化器,按 best-iterate 口径上报:

| 迭代轮数 | 报告的"提升" | 相对基线 0.48 |
| --- | --- | --- |
| 5 | +0.117 | +24% |
| 8 | +0.143 | **+30%** |
| 12 | +0.163 | +34% |
| 20 | +0.186 | +39% |

作为对照:AutoSOTA 自己在 ICML 上报的成功案例**中位提升是 3.43%**,29% 的案例 ≤1%。
也就是说,**纯噪声在这个 benchmark 上刷出来的"提升",比真实系统报告的中位提升大一个数量级。**

复现这个模拟:

```python
import random, statistics
random.seed(0); SIGMA, BASE = 0.10, 0.48
for n in (5, 8, 12, 20):
    g = [max(BASE + random.gauss(0, SIGMA) for _ in range(n)) - BASE for _ in range(20000)]
    print(n, round(statistics.mean(g), 3))
```

### 2.3 要分辨真实效应需要多少轮

σ≈0.1、目标检出 Δ=0.05(远大于我们实测到的任何真实效应)、双臂 80% power:
**每臂约 63 轮**。按每轮 5~8 小时 + ¥58 计算,一次对照 ≈ ¥7,300、数百小时。

**结论:任何 n<5 的对照只能说方向,不能说优劣;n=1 的"提升"没有信息量。**

---

## 3. 推荐的替代目标函数

`deepcode_test/scripts/gates/exec_level.py` —— 确定性、零成本、秒级,不调用任何模型:

```bash
python3 deepcode_test/scripts/gates/exec_level.py --json <提交目录>
# → {"score": 4, "max": 5, "gates": {"compiles": true, ...}}
```

五项判据:全部 `.py` 语法可解析 / 有 `__main__` 入口 / 有依赖声明 / 有复现脚本 / 有配置文件。

在本仓库 10 份已判提交上的实测(它与裁判分**几乎不相关**,测的是不同的东西):

| 提交 | 执行级 | 裁判分(SiliconFlow) |
| --- | --- | --- |
| rice trial2(DeepCode) | 5/5 | 0.5447 |
| fre trial1 / trial5(DeepCode) | 4/5 | 0.5184 / 0.4246 |
| rice trial3(DeepCode) | 4/5 | 0.4374 |
| rice bare_kimi | 4/5 | 0.4633 |
| fre anchor / rice trial_k1 | 3/5 | 0.4839 / 0.2815 |
| fre bare_v4 / rice bare_v4 | 2/5 | 0.4817 / 0.4680 |
| rice trial_k2(有语法错文件) | 2/5 | 0.4760 |

**⚠️ 这些是结构性就绪度,不是"跑通了"。** 真正的执行级验收(装依赖 → import →
冒烟训练 → 指标 schema → 环境模块核验)需要容器,设计见
`ARCHITECTURE_v0.2_OPTIMAL.md` 的度量体 B。本脚本是那之前的最小可用替代。

---

## 4. 强制纪律:rubric 不得进入优化循环

`paperbench_changes/rubrics/` 里有 fre(437 叶)与 rice(361 叶)的判分树。
它们放在这里是**给人事后核对失分分析用的,不是给优化器读的**。

本项目有直接教训:修复轮 trial_fx1/fx2 的提示词里只是出现了一句评分结构元知识
(*"Graders assign separate credit to each baseline; omitting them forfeits those points"*),
两轮产物就整体作废(见 `REVIEW_local_changes_2026-09-03.md`)。

**跑优化循环前先执行:**

```bash
bash deepcode_test/scripts/ci/check_no_rubric_leak.sh    # 扫描禁词,有命中就别跑
```

或者干脆把 rubric 移出工作树:

```bash
git rm -r --cached paperbench_changes/rubrics && echo "paperbench_changes/rubrics/" >> .git/info/exclude
```

---

## 5. 成本与时间预算

| 项 | 单价 | 说明 |
| --- | --- | --- |
| 一轮复现 | ≈¥20 / 3~6h | DeepSeek-V4-Pro,含参考挖掘 + 索引 + 规划 + 写码 |
| 一份判分 | ≈¥38 / 40~100min | code_only,fre 306 叶 / rice 178 叶 |
| **一次完整迭代** | **≈¥58 / 5~8h** | 上面两项串行 |
| 中位 8 轮 | ≈¥460 / 数天 | AutoSOTA 的迭代中位数 |
| 一次有统计意义的对照(每臂 63 轮) | ≈¥7,300 | 见 §2.3 |

本项目累计花费约 ¥1,600,其中相当一部分在废轮上。作废原因与防范见 `PITFALLS.md`。

---

## 6. 已知会误导优化器的地方

| 陷阱 | 说明 |
| --- | --- |
| 伪造计划 | 规划三连败后上游会造一个通用脚手架计划并标 `success` / `completeness_score=1.0`。消费 `initial_plan.txt` 前必须验 `planning_result_meta.json.source == "generated"` |
| 静默降级 | 检索/挖掘/判分三侧共四种"输出超限或为空 → 当正常继续",只留 INFO 日志。见 `FINDING_prefilter_silent_failure.md` |
| 效应不可加 | 反事实"补上三个基线 → 1.354×"被实验推翻:基线补上了,主方法反而下滑。**不要假设各维度独立可加** |
| 15 处未门控改动 | 本地默认行为已不等于上游,见 `REVIEW_local_changes_2026-09-03.md`。要严格对齐上游需从 `e0767d0` 重新检出 |
| 写码阶段不验证 | 执行工具被调用过 50 次,但全是 `mkdir`/`touch`/`find`;**没有一次运行生成的代码**。索引模式下工具面只有 `write_file` + `search_code_references` |

---

## 7. 如果你只想要一个能用的循环

最小可行配置:

1. 目标函数用 `gates/exec_level.py`(§3),不要用裁判分
2. 跑循环前执行 `ci/check_no_rubric_leak.sh`(§4)
3. 每轮把 `{iteration, commit, exec_score, gates, cost, wall}` 追加进 `scores.jsonl`
4. 裁判分只在里程碑处跑,**双 serving 并列**,当审计而非目标(见 `FINDING_judge_serving_dependence.md`)
5. 任何"提升"在 n≥5 之前只报方向

更完整的设计(度量体与复现体分离、论文锚定测试、哈希锁、独立审计)见
`ARCHITECTURE_v0.2_OPTIMAL.md`。
