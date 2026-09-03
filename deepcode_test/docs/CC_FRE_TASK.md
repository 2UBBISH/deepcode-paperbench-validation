# 任务:用你当前的模型复现 fre 论文(对照实验)

> 给 Claude Code 的任务书。**照做即可,不需要理解下面的实验背景。**
> 如果你想知道这是在干什么:见文末「实验背景」。

---

## 你要做的事

按 PaperBench 官方的 Code-Dev 协议,复现一篇研究论文,把代码写进指定目录。

**论文材料在**:`/home/deepevol/deepevol/cc_dsv4_run/paper/`
- `paper.md` — 论文正文(markdown)
- `paper.pdf` — 论文原件
- `addendum.md` — 补充说明(**必读**,写明了哪些实验不在范围内)
- `blacklist.txt` — **禁止访问的资源清单**

**你的产出放在**:`/home/deepevol/deepevol/cc_dsv4_run/submission/`

---

## 官方任务书原文(逐字执行)

You are tasked with reproducing a research paper.

You have been provided with a research paper located at `/home/deepevol/deepevol/cc_dsv4_run/paper`. The directory contains the paper in both PDF and markdown format.

### TASK

Your task is to replicate as many of the core contributions of this paper as possible, within the allotted time you have available. You may not be able to complete everything within the allotted time; you should make prioritisation decisions in order to do as much as possible within your allotted time.

Do not worry if you fail to completely reproduce particular parts of the paper. You will be assigned partial credit, such that if only a specific part of your reproduction is incorrect, only that part will fail to receive credit, while remaining correctly implemented parts of your work will be marked as successful. Note that the partial credit you will be assigned will be weighted by importance to the paper's main contributions. Use your judgement to determine what parts of the paper are more important than others to reproduce. Note that we will check that your results match the general trends of the original paper and we will allow for a reasonable margin of error, so you should not worry if metrics do not match exactly between your reproduction and the paper.

Any experiments that are only introduced in the Appendix are not considered core contributions of the paper, and so are out of scope. This is the case even if the experiment is referenced in the main text. HOWEVER, if an experiment is described in the main body of the text, but some of the details used (e.g. details about a prompt used, or a long mathematical expression) have been put in the appendix, then that experiment is considered still in scope. The reason here is that the experiment is described in the main body of the paper.

### RESOURCES

You may or may not be provided with a rubric called `rubric.json` in the `/home/deepevol/deepevol/cc_dsv4_run/paper` directory to guide your efforts. The `rubric.json` has a hierarchical tree structure, at each level breaking down the paper into increasingly fine-grained tasks. Note that you may be provided with only a subset of the full task hierarchy, or even just a single task asking you to reproduce the whole paper - in this case, you should use your best judgment to break down the tasks you are given into appropriate sub-tasks.

There is also an addendum file at `/home/deepevol/deepevol/cc_dsv4_run/paper/addendum.md` which provides additional context needed (if any) to reproduce the paper, as well as potential clarifications about what is not in scope for the reproduction.

You are allowed to browse and use resources you can find online, however you must not look at or use the paper's codebase or any other resources listed in `blacklist.txt`, as this is considered cheating.

### SUBMISSION

The output of your work must be a git repository containing all the necessary components to reproduce the results of the paper. Your repository should be located at `/home/deepevol/deepevol/cc_dsv4_run/submission/`, and the committed files should not exceed 1GB in size (i.e. you should only commit source code without any heavy artifacts). We will run `git clean -fd` on the submission directory before grading it, so any untracked files will be lost.

Your submission will be graded based on whether the code necessary for reproducing the paper has been correctly implemented. The code will not be executed during grading.

Finally, please also include a README.md file that describes what you were able to achieve in your reproduction attempt and explains how your codebase relates to various parts of the reproduction.

---

## 三条硬性纪律(违反会让这次实验作废)

1. **不许访问 `blacklist.txt` 里列的仓库**(即论文作者的官方实现)。
   这是作弊。事后会审计你的完整操作记录,一旦发现访问过就作废。
   注:该仓库地址会在论文摘要里出现,**读到不等于访问**,不要主动去看它。

2. **不许向人类提问、不许等待人类补充信息**。
   这一轮要模拟"无人值守的自动复现",任何人工提示都会污染对照。
   遇到判断不清的地方,自己做决定并在 README 里说明理由。

3. **产出必须是 git 仓库,且全部提交**。
   `git init` + `git add -A` + `git commit`。未跟踪的文件在判分前会被 `git clean -fd` 删掉。
   同时必须有 `README.md`,说明你实现了什么、没实现什么、为什么。

---

## 做完请报告

在最后一条回复里给出:

- 实现了论文的哪些核心组件(列清单)
- 哪些没做,为什么(时间不够 / 需要外部代码库 / 判断为超出范围)
- 提交目录的文件数、Python 代码总行数、commit 数
- 你自评这次复现的完整度

---

# 实验背景(可选阅读)

我们在验证一篇叫 DeepCode 的论文。它声称:给 AI 配上一套"论文复现专用脚手架"
(自动规划 + 检索参考代码库 + 结构化写码循环),复现效果能比通用编码工具**高 1.34 倍**。

已经测过的:
- **DeepCode + DeepSeek-V4-Pro**(有脚手架):0.5184 / 0.4378 / (第三轮待判)
- **裸 Claude Code + Sonnet 4.5**(无脚手架):0.4839

问题是这两组用的是**不同的底座模型**,所以分不清"打平"是因为脚手架没用,
还是因为模型差异抵消了脚手架的贡献。

**你这一轮要补的就是缺失的对照格**:

|  | 裸跑 | + DeepCode 脚手架 |
| --- | --- | --- |
| Sonnet 4.5 | 0.4839 ✅ | ✗ 跑不了(无 API) |
| **DeepSeek-V4-Pro** | **← 你这一轮** | 0.5184 ✅ |

如果你(裸跑)也拿到 0.50 左右,说明**脚手架没有带来增益**——同模型、同任务书、同裁判,
唯一变量就是有没有脚手架,这是最干净的证据。

评分用 PaperBench 官方 SimpleJudge,只审查代码是否忠实实现了论文所述算法,
**不会运行你的代码**。所以重点是把论文的方法完整、准确地写出来,不必纠结能不能跑通。
