# 判分树(rubric)—— 仅供事后核对,严禁进入复现流水线

`fre.rubric.json`(437 叶)与 `rice.rubric.json`(361 叶)自上游 PaperBench
`data/papers/<paper>/rubric.json` 原样复制,未做任何修改。放在这里是为了让读者能自行核对
本项目的失分分析(哪个叶子得 0、权重多少、两个裁判在哪 28 个叶子上分歧)。

## 使用纪律

**这些文件是评测答案,不是输入。** 本项目已有一次教训:修复轮 trial_fx1/fx2 的提示词
里出现了一句评分结构元知识("Graders assign separate credit to each baseline"),
两轮产物因此整体作废(见 `../../deepcode_test/docs/REVIEW_local_changes_2026-09-03.md`)。

因此:
- 复现流水线的任何阶段(规划、写码、检索、修复)都不得读取本目录;
- 复现 agent 的工作区不得包含本目录;
- 只在判分完成后,用于人工核对与统计。

## 未上传的部分

`data/papers/rice/judge/`(37MB)是被引论文(JSRL、StateMask)的 PDF 与插图,
属第三方版权内容,不在本仓库分发;需要时由 `setup.sh` 从上游按固定 commit 拉取。
