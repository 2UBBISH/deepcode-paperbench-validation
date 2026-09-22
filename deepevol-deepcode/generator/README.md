# generator — DeepEvol Paper2Code 线的独立快照

来源：DeepEvol `0916onmain-experiment` @ `4ed40890ee15`（`GENERATOR_COMMIT`）。拷了线本身（`apps/v2/agent/paper2code`、
`apps/v2/agent_engine/paper2code`）、它传递 import 到的 DeepEvol 模块（`agent_engine/{experiment,rsa,remote_relay}`、
`agent_shell`、`reliability`、`remote_compute`、`database`、几个单文件）、`scripts/paper2code_canary.py`、离线测试
`tests/v2_paper2code`、第 10 步的镜像 Dockerfile、`pyproject.toml` + `uv.lock`（锁定的依赖）。**只靠这个目录就能生成第 9 步的树**；
第 10 步（租机、试跑、修复）也在，但需要阿里云账号与镜像。验证过：`uv sync --frozen --extra agent-runtime` 后 224 项离线测试通过
（`test_footprint` 里 DeepEvol 独有的两条登记已去掉）。

线的说明：`apps/v2/agent/paper2code/README.md`（布局、命令、"生成到第 9 步并摆卷"）、`STATUS-2026-09-21.md`（现状一页）、
`HANDOFF.md`、`PITFALLS.md`、`adr/0004-the-blueprint-is-the-binding.md`（原文保真的决定）。

## 装

```bash
cd deepevol-deepcode/generator
uv sync --frozen --extra agent-runtime        # Python ≥3.11；uv 没有的话 pip install uv
.venv/bin/python -m pytest tests/v2_paper2code -q   # 224 passed，约 4 分钟，不联网
```

## 跑一篇到第 9 步（对比口径：不租机、不执行、不修复）

key 只通过 `--env-file` 传：一个文件里 `DEEPSEEK_API_KEY=...`（官渠 api.deepseek.com；`deepseek-flash` 别名 09-22 起为 V4.1-Flash），
另一个里阿里云的变量（第 9 步的 `compute` 只查价不租机，但 init 需要它存在；见 `apps/v2/agent/paper2code/README.md`）。

```bash
PY=.venv/bin/python; R=/path/to/runs; P=fre
$PY scripts/paper2code_canary.py init --run-dir $R/$P --paper-dir ../../materials/papers/$P \
  --compute aliyun --figures off --planning-fanout --repair-rounds 3 --run-hours 2 \
  --model deepseek-flash --figures-model DeepSeek-V4-Flash-Vision-Exp --experiment-model DeepSeek-V4-Flash-Vision-Exp \
  --provider-base-url https://api.deepseek.com/v1 --provider-key-env DEEPSEEK_API_KEY --thinking enabled
$PY scripts/paper2code_canary.py run --run-dir $R/$P --until compute --env-file deepseek.env --env-file aliyun.env
$PY scripts/paper2code_canary.py submit --run-dir $R/$P --paper $P --trial deepcode --dest-root ~/pb_submissions
```

批量：`../../paperbench/scripts/gen_batch.sh`。对照（保真关）：run 前 `export DEEPCODE_PAPER_FIDELITY=0`。
09-21/22 的 20 篇就是这个口径（思考开、65536/65536、1M、扇出开、保真开、语法检查开）。
