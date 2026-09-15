# bam 判分结果

裁判：PaperBench Code-Dev，DeepSeek-V4-Pro @ Paratera，`PB_JUDGE_CONCURRENCY=20`，修过选文件根目录 bug（README §4.2）。

| 文件 | 臂 | 总分 |
| --- | --- | --- |
| `bam_ed22b0a2-*.grade.json` | 03_bare.gpt5-codex-high（gpt-5.5 high，不进主表） | 0.9073 |
| `bam_478de27b-*.grade.json` | 02_deepcode trial1（V4-Pro） | 0.8367 |
| `bam_2343bcdb-*.grade.json` | 03_bare Codex 桌面版（V4-Pro） | 0.7343 |

`void_judge_root_bug/`：修裁判前的三份分（b3c8351f 0.7644 gpt-5.5 / e598e3a4 0.6659 DeepCode / 239fba67 0.6530 Codex V4）与并发 16 对照（96c62370 0.6317 DeepCode），**作废**，只作修前修后对读。
01_deepevol 没有分（停跑），见 README §0.3。
