# 裸跑臂操作说明（owner 看的）

**输入按数据集来**（2026-09-14 晚定）：三方（我们的线、DeepCode、裸跑）拿到的都是 PaperBench 给 agent 的那一套，
不多不少：`paper/paper.pdf`、`paper/paper.md`、`paper/addendum.md`、`paper/blacklist.txt`、`paper/assets/`（本机只有
LFS 指针，三方都拿不到图），**不给 rubric**。指令 = PaperBench 官方 `code_only_instructions.txt` **逐字**，只换路径
（`PROMPT_official.txt`；8 行差异全是路径）。参考仓库靠自己上网找，黑名单挡住论文自己的代码。

之前那份 `PROMPT_codedev_tilted.txt`（rice/fre 用的，多一句"优先实现、别管可运行性"）**作废**，只留作记录；
用它跑出的 `submission/`（09-14 20:55，16 py / 2375 行）不判分，已移到 `submission_tilted/`。

口径：DeepSeek-V4-Pro @ Paratera，**思考关且有证据**（必须走代理，`proxy_requests.log` 每行 `thinking: disabled`），
单次会话不设时限，同 DeepCode / 我们的线。

## 跑

```bash
cd /Users/apple/Documents/env/paperbench-judge/validation/deepcode_test/sequential-neural-score-estimation/workspaces/bare_v4
rm -rf submission && mkdir submission
python3 paratera_proxy.py 8787 proxy_requests.log   # 另开一个终端留着
```
cc-switch 的供应商 Base URL 填 `http://127.0.0.1:8787`（key 不变，模型 DeepSeek-V4-Pro）。起 Claude Code，把
`PROMPT_official.txt` **整段**喂进去，不加别的话。记开始/结束时间。结束后确认 `proxy_requests.log` 非空且每行
`"thinking": {"type": "disabled"}`——没有这个文件的跑法不算数。

## 判分

```bash
rm -rf ~/pb_submissions/sequential-neural-score-estimation/bare_v4 && cp -r /Users/apple/Documents/env/paperbench-judge/validation/deepcode_test/sequential-neural-score-estimation/workspaces/bare_v4/submission ~/pb_submissions/sequential-neural-score-estimation/bare_v4 && cd /Users/apple/Documents/env/paperbench-judge/validation && PATH=/Users/apple/Documents/env/bin:$PATH PAPER=sequential-neural-score-estimation bash deepcode_test/scripts/run_grade.sh
```

约 ¥38。判完：`grades/` 新出的 grade.json 改名 `bare_v4.grade.json`；`submission` 拷到 `../../submissions/bare_v4`
（去 `.git`，另存 `GIT_HISTORY.txt`）；已判提交移到 `~/pb_submissions_archive/sequential-neural-score-estimation/bare_v4`。
分数和维度分解进 RESULTS.md。
