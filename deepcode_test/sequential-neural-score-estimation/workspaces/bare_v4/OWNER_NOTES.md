# 裸跑臂操作说明（owner 看的，不给写码会话）

工作区：本目录。`paper/` 是任务输入（论文 md/pdf、addendum、blacklist、两份 addendum 点名的仓库快照），
`submission/` 空目录，会话往里写。写码会话只读 `BRIEF.md` 起步。

## 口径（和 DeepCode trial2、我们的 Stage 9/10 导出一致）

- 底座：DeepSeek-V4-Pro @ Paratera（`~/Documents/env/paratera.env` 那把 key），**思考关**。Claude Code 接法照 rice/fre 那次。
- 输入：论文 + addendum + blacklist + 两份点名仓库快照。DeepCode 拿到的是论文 + 它自己搜的 5 个仓库；我们是论文 +
  文献层 4 个仓库 + 同样两份上传。不给 rubric（三方都没给）。
- 单次会话，不设时限；记开始/结束时间，写进 RESULTS.md。

## 跑

```bash
cd /Users/apple/Documents/env/paperbench-judge/validation/deepcode_test/sequential-neural-score-estimation/workspaces/bare_v4
# 起 Claude Code（接 V4-Pro、思考关），第一句：读 BRIEF.md，然后开始。
```

## 判分

会话结束后（`submission/` 里已 git 提交）：

```bash
rm -rf ~/pb_submissions/sequential-neural-score-estimation/bare_v4 && cp -r submission ~/pb_submissions/sequential-neural-score-estimation/bare_v4 && cd /Users/apple/Documents/env/paperbench-judge/validation && PATH=/Users/apple/Documents/env/bin:$PATH PAPER=sequential-neural-score-estimation bash deepcode_test/scripts/run_grade.sh
```

约 ¥38。判完把 `grades/` 里新出的 `sequential-neural-score-estimation_<uuid>.grade.json` 改名 `bare_v4.grade.json`，
`submission` 拷一份到 `../../submissions/bare_v4`（去掉 `.git`，另存 `GIT_HISTORY.txt`），
已判的提交移到 `~/pb_submissions_archive/sequential-neural-score-estimation/bare_v4`。然后我把分数和维度分解补进 RESULTS.md。
