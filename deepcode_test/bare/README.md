# 裸跑臂（bare）的两个固定件

三方对照的第三臂是「编码 agent 裸跑」：Codex 桌面版 + 同底座模型，提示词 = PaperBench 官方
`code_only_instructions.txt` 原文 + 本目录的冻结后缀，只替换两处路径（`/home/paper`、`/home/submission`）。
完整口径与起跑前的三条核验命令见 `docs/INPUT_STANDARD.md`。

- `bare_prompt_suffix.txt`：冻结的两句后缀（不要停下来问；产物只放 submission/）。补的是 harness 条件，不含论文或判分信息。
- `paratera_proxy.py`：直通代理，请求体原样转发到 Paratera，只记录每次请求的 model 与 usage（含 `reasoning_tokens`），
  证明裸跑臂用的是哪个模型、思考开没开。key 不落日志。

本批（2026-09-17 起）先只做 DeepEvol 复现线 vs DeepCode 基线运行，裸跑臂暂不起；文件保留以便三方口径不变。
