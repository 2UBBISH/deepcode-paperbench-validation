# 裸跑臂操作说明（owner 看的）

工作区：本目录。`paper/`（论文 md/pdf、addendum、blacklist），`submission/` 空目录。
任务书 `PROMPT.txt` = rice/fre 那份**逐字**，只换了路径和黑名单 URL；没有别的提示，没有评分知识
（rice/fre 的两轮 fx 就是因为提示词多了一句评分结构的话整体作废的）。

口径：DeepSeek-V4-Pro @ Paratera，思考关，单次会话不设时限，同 DeepCode trial2 / 我们的 Stage 9/10 导出。
输入差异如实记：裸跑只有论文 + addendum（和 rice/fre 一样）；DeepCode 自己搜了 5 个仓库；我们的线上传了 addendum 点名的两个仓库快照。

## 跑

```bash
cd /Users/apple/Documents/env/paperbench-judge/validation/deepcode_test/sequential-neural-score-estimation/workspaces/bare_v4
```
起 Claude Code（接 V4-Pro、思考关），把 `PROMPT.txt` 整段喂进去。记开始/结束时间。

## 判分

```bash
rm -rf ~/pb_submissions/sequential-neural-score-estimation/bare_v4 && cp -r /Users/apple/Documents/env/paperbench-judge/validation/deepcode_test/sequential-neural-score-estimation/workspaces/bare_v4/submission ~/pb_submissions/sequential-neural-score-estimation/bare_v4 && cd /Users/apple/Documents/env/paperbench-judge/validation && PATH=/Users/apple/Documents/env/bin:$PATH PAPER=sequential-neural-score-estimation bash deepcode_test/scripts/run_grade.sh
```

约 ¥38。判完：`grades/` 新出的 grade.json 改名 `bare_v4.grade.json`；`submission` 拷到 `../../submissions/bare_v4`
（去 `.git`，另存 `GIT_HISTORY.txt`）；已判提交移到 `~/pb_submissions_archive/sequential-neural-score-estimation/bare_v4`。
然后分数和维度分解进 RESULTS.md。
