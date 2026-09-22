# Paper2Code line

Runs the vendored DeepCode engine (`apps/v2/agent_engine/paper2code/`) on a
paper as PaperBench hands it over, one file-backed run per directory, with
model calls going straight to Paratera and generated code executed on a
rented Aliyun machine. Vocabulary: root `CONTEXT.md` → *Paper2Code line*.
Build order and decisions: `PLAN.md` → `PLAN-2.md` → `PLAN-3.md` (the current
one: §0 the owner's decisions, §2.3 S1–S9, §2.4 the to-do); why it is not a
graft into the reproduction line: `adr/0001-separate-engine-not-a-graft.md`.

## State (2026-09-22)

Read `HANDOFF.md` §0.-2 (09-21/22 in one table) and `STATUS-2026-09-21.md` (the line as the code has it); the short form:

| | |
| --- | --- |
| branch | `0916onmain-experiment` on `HuigenYe/DeepEvol` (`2a94d3d38`); validation repository `~/Documents/env/paperbench-judge/validation` (master `9e48fbf`) holds the judge and the DeepCode baseline arm; its branch `0919-test` holds the desktop-arm rules; **three-arm submissions, run notes and grades are collected outside both repositories in `~/Documents/0919-test/`** (`deepcode/` · `codex/results/` · `claude/` · `grades/` · README with the summary table) |
| done | PLAN-3 S1–S9 (step 10 on a rented machine, CPU tier first), T1–T3b, T5, T13; **ADR 0004** paper fidelity (blueprint `Source:` pointers → `source_manifest.json` → `read_paper(file_path)` reads the whole bound section, every page → `write_file` refused until read → audit recorded, never a gate); static syntax check + repair after generation; no execution before step 10 |
| caliber (generation) | every slot `deepseek-flash` @ `api.deepseek.com` (the official `deepseek-flash` alias — **DeepSeek-V4.1-Flash** since the 09-2x switch, owner 09-22; Paratera's `DeepSeek-V4-Flash` is the older V4), **thinking on**, 1M context, planning fan-out on, 65536 output per call (planning and implementation), figures off, `DEEPCODE_PAPER_FIDELITY=1`. Same bytes in as the desktop arms (paper.md + addendum + blacklist) |
| **caliber (judging)** | validation `run_grade.sh` default since 09-21 16:40: SiliconFlow `deepseek-ai/DeepSeek-V4-Flash` as judge with the **whole submission tree** in every leaf's prompt (`PB_JUDGE_WHOLE_CODEBASE=1`, prefix cache ≈ 98 %, ¥32 a 306-leaf paper, half price 02:00–08:00) and **thinking off**; structured parser `deepseek-ai/DeepSeek-V4-Pro` (json_object, up to 3 attempts — its failures are transient). JudgeEval rice/0: 0.70–0.72 accuracy with 178/178 valid, level with the upstream top-10 / Paratera caliber (0.719). Scores judged before 09-21 (Paratera, top-10) are a different caliber and are not compared with these |
| **scores (same caliber, 09-21/22)** | 8 papers paired (line / Codex): fre 0.961 / 0.914 · rice 0.978 / 0.976 · pinn 1.000 / 1.000 · lbcs 0.987 / 0.993 · lca 0.925 / 0.898 · what-will 0.990 / 0.988 · robust-clip hydrated 1.000 / 0.941 (truncated md: 0.642 / 0.906). Line ahead 3, level 4, behind 0; four papers at the rubric ceiling on both arms. **fre ablation**: fidelity on 0.961 / off 0.862 / Codex 0.914 — the read-back machinery is worth +0.10. One tree per paper per arm, noise ≈0.025–0.05. Memorisation probe against the authors' repos: no copied code |
| comparison rule (owner 09-18) | **PaperBench comparisons stop at stage 9** (`compute`): the graded tree is the generated one; step 10 (environment / trial / repair) is validated on its own runs and does not enter the comparison. `submit` records `tree: stage9` |
| running | night grading only inside the SiliconFlow half-price window (validation `runs/grade_night_v4.sh`, ≈21 trees left, continues 02:00 nightly); the 18 t19 trees are in `~/Documents/0919-test/deepcode/`; owner re-runs Codex on the hydrated robust-clip (done) and the Claude desktop arm (none yet) |
| parked | T4 torch pre-installed images (T4b wheelhouse not started); T7 execution-port retirement; T9 criteria (colleague's paper → rubric JSON; stub phase); follow-ups O1–O5 in `STATUS-2026-09-21.md` §9 (glue misclassification — owner: collect more lost leaves first; addendum sub-headings; tree size vs judge cap; whole-section obligation untested live; optional-baseline triage) |
| machines | CPU tier `ecs.c7.xlarge` 1.39 CNY/h, escalation T4 `ecs.gn6i-c4g1.xlarge` 8.07 CNY/h; a stage-9 run rents nothing |

## 生成到第 9 步并摆卷（给同事 / 另一个 Claude 会话的最短路径）

"摆卷" = 把生成的代码树导出成 PaperBench 判分用的一份提交（`<目录>/<paper>/<trial>/` + `submission.json`）。
对比口径只跑到第 9 步：**不租机器、不判分、不动 git**。一篇论文 40–80 分钟，模型 token 3–9M，机器费 0。
（更细的冲突表、卡点和交回清单：`HANDOFF-STAGE9.md`。）

```bash
cd ~/Documents/search/DeepEvol-Paper_repro_0916/DeepEvol1.0      # 本 worktree；或按 HANDOFF-STAGE9 §1 方式 B 另开一个
PY=.venv/bin/python; R=~/Documents/search/paper2code-runs; E=~/Documents/env
$PY -c "import docker, apps.v2.agent.paper2code.driver" && git log --oneline -1   # venv 能用；记下提交

NAME=<运行名>                # 自己起，别复用 sapg* / pinn*
PAPER=<paperbench id>        # ~/Documents/search/paperbench/<PAPER>/paper.md 必须存在。--figures off 不读图；--figures on 前要把 assets/*.jpg 从 LFS 指针补成真图（见 HANDOFF-STAGE9 §3）：一张都没描述成功 intake 就失败（不让开图悄悄变成关图）
OUT=<摆卷目录>               # 任意目录，例 ~/my_submissions；默认 ~/pb_submissions（主会话的判分池，别往里放）

# 1. 初始化（口径与 S9 / T5 一致：Vision-Exp、思考关、单次 32768、上下文 1M、规划扇出开；--figures on 是另一种输入）
$PY scripts/paper2code_canary.py init --run-dir $R/$NAME --paper-dir ~/Documents/search/paperbench/$PAPER \
  --compute aliyun --figures off --planning-fanout --repair-rounds 3

# 2. 跑到第 9 步为止（--until compute：第 9 步只做静态估算 + 查价，不租机）
nohup $PY scripts/paper2code_canary.py run --run-dir $R/$NAME --until compute \
  --env-file $E/paratera.env --env-file $E/aliyun.env > $R/$NAME/console.log 2>&1 &
echo $! > $R/$NAME/console.pid

# 3. 看进度（phases 各阶段状态 + 四道闸；compute=completed、environment_run=pending 即到位）
$PY scripts/paper2code_canary.py status --run-dir $R/$NAME | head -60

# 4. 摆卷到指定位置（不需要 --env-file；四道闸都过 + compute 完成即接受，记 tree: stage9）
$PY scripts/paper2code_canary.py submit --run-dir $R/$NAME --paper $PAPER --trial <trial名> --dest-root $OUT
#    → $OUT/$PAPER/<trial名>/  = generate_code/ 的拷贝（去掉 __pycache__ / .pyc 等），旁边 submission.json（sha256 清单、口径、tree）
#    同名已存在加 --force；要拿代码历史里的某一状态用 --snapshot pre_repair|<commit>（第 9 步的运行没有历史，不用）
```

| 常见卡点 | 处理 |
| --- | --- |
| `references` 失败 `reached the maximum number of tool call iterations (80)` | 09-19 起阶段自己会在 40 次用尽时**自动用 80 再跑一次**（`phases/05_references.json` 记 `iterations_retry`，事件 `references.retry`）；80 也用尽才失败，那就 `DEEPCODE_REFERENCE_MAX_ITERATIONS=160 … rerun --phase references` |
| intake 失败 `--figures on but no figure was described: N figure references, 0 described (lfs_pointer)` | 资产是 LFS 指针（`version https://git-lfs…` 文本）。按 HANDOFF-STAGE9 §3 从 HF 补成真图后 `rerun --phase intake`；或者本来就想关图，`init` 时用 `--figures off` |
| `submit refused: gate … not passed` | `status` 看是哪道闸；`implementation_status` 的 `incomplete` / `no_tests_discovered` 不算失败（生成仓库没有测试）；`empty_files` 只记录不拦（T13） |
| `submit refused: environment_run is … and compute is …` | 没跑到第 9 步，或已经进了第 10 步却没完成；前者继续 `run --until compute`，后者不是对比口径 |
| 要判分 | 把 `$OUT/$PAPER/` 下的提交交给验证仓库的 `run_grade.sh`（下文 Commands），或交回主会话 |

规则：只通过 `--env-file` 传 key（不要 `cat`、不要 `bash -x`）；不要 `run --until environment_run`（会租机，租了必须 `release`）；不要改 `apps/v2/agent_engine/` 和 `apps/v2/agent/paper2code/`，发现问题记下交回。

## Layout

| file | role |
| --- | --- |
| `config.py` | `run.json` (`RunConfig`: the model slots `model` / `figures_model` / `experiment_model`, all `DeepSeek-V4-Flash-Vision-Exp` for now; `context_window`, default 1M → `DEEPCODE_PLANNER_CONTEXT_WINDOW`, the planner's segment budget; `planning_fanout`, `--planning-fanout` → `DEEPCODE_PLANNING_FANOUT`, VENDOR 11), run directory (`RunPaths`), env defaults (PLAN §6), kernel config, `install()` binding the ten seams |
| `provider.py` | `ParateraProvider`: httpx, `thinking: {"type": <run.thinking>}` on every call (`disabled` = the Paratera batches, guarded by `ThinkingNotDisabled`; `enabled` = the 09-19 deepseek-flash caliber, reasoning tokens recorded), persistent retries, `llm/<seq>.json` |
| `runner.py` | the tool-calling loop (`docs/INTEGRATION.md` §4) bound onto the seam's `AgentRunner` |
| `agent.py` | `Agent.__aenter__/__aexit__/attach_llm`, `AugmentedLLM.generate` bound onto the seam classes |
| `tools/` | the engine's seven "MCP servers" as in-process tools; denylist and repeat-fetch guards. Since S3 (PLAN-3 8i) the implement phase runs the engine's own `execute_python` / `execute_bash` — a local subprocess in the engine's write-fence sandbox — and its own local verification; `tools/execute.py` (the port-routed pair) is no longer registered |
| `execution/` | `port.py` contract, `job_executor.py` (one job per network-less container, image policy), `aliyun_lease.py` (ECS client; `GPU_CATALOG`; GPU types boot the GPU image `ALIYUN_GPU_IMAGE_ID`; both images baked by `scripts/paper2code_bake_gpu_image.py` — `--cpu` for the CPU one — S5 / T4), `remote_daemon.py` (SSH key, tar sync; its docker.sock tunnel is vendored but unused), `leased_runtime.py` (rent on first job, docker commands over SSH on the machine, release on close), vendored `remote_relay/`. The port is slated for retirement (T7) |
| `verification_hook.py` | the engine's mechanical verification routed through the port — not installed since S3; `phase_implement` only uninstalls it as a guard |
| `intake.py`, `gates.py` | paper bundle → `input/paper.md` (+ addendum), then the figure pass; the four `run_trial.sh` gates (preflight also runs the vision probe unless `figures=off`) |
| `phases.py`, `plan_review.py`, `driver.py` | the eleven phases (**no execution before step 10** since 09-20: the coding agent has no execute tools and upstream's post-generation `pytest` run is off unless `PAPER2CODE_IMPLEMENT_VERIFY=1`, event `implement.execution`; **static syntax check** after generation (`syntax_check.py`, owner 09-20 evening — the desktop arms may run `py_compile`, so the line gets the same: `compile()` every `.py`, nothing executed, up to `DEEPCODE_SYNTAX_ROUNDS`=2 repair rounds with `edit_file` and no container, `DEEPCODE_SYNTAX_CHECK=0` turns it off; under paper fidelity the repair round runs inside a `FidelitySession` — `read_paper(file_path)` before editing a pointed file, edits recorded in the trace — so the audit that follows sees the repaired bytes as recorded writes; `implement.syntax` events, `syntax` in the phase record; fre-t14's `prior.py:503` repeated keyword would have been caught); `plan` reads the Source pointers back at its end (`plan.source_manifest` event: files / paper_files / glue / pointers / unmatched) and **re-plans once** only when the plan has no `Source:` line at all (`DEEPCODE_PLAN_REPLANS`, default 1; the reason goes to the planner via `DEEPCODE_PLANNER_FEEDBACK`; event `plan.replan`); `plan` ends with the environment-spec extraction; `index` records per-repository `files_found / files_analyzed` and a `prefilter_fallback_suspected` flag from the engine's index metadata; `environment_run` with `--compute aliyun` is one call to main's experiment agent — see `experiment_step.py`; with local docker it keeps the record-only compileall + entry smoke), the `--ask` review loop, the run state machine |
| `figures.py` | intake's optional figure pass (`run.json.figures`: `auto` / `on` / `off`, `--figures` at init): the pass uses a separate vision model (`run.json.figures_model`, `--figures-model`, default `DeepSeek-V4-Flash-Vision-Exp`; the phase model stays pinned); the preflight probe sends a PNG of four random-colour quadrants and accepts only ≥3 of 4 named in order (a text model behind an endpoint that drops the image cannot pass); then every `![](assets/…)` of the paper is described from the image by one call (plot axes / tick values / curves / seeds, table images transcribed, pseudocode boxes copied) and inserted right after the reference as a marked blockquote; the benchmark bytes stay in `input/paper.raw.md` (`run.json.paper_sha256` is their hash), descriptions are cached in `input/figures.json`, image data is redacted from `llm/<seq>.json`. A run with figures described is a different input from the baseline's — compare like with like. With `figures=on`, intake fails when the pass described none of the paper's figures (LFS-pointer assets are skipped as `lfs_pointer`; a paper without figure references passes) |
| `environment_spec.py` | the environment spec (环境规格) read out of the blueprint after planning by one model call: language version, packages, GPU need, datasets, external tools, run commands; written to `<run>/environment_spec.json`, a failure is recorded on the plan phase, never raised |
| `compute.py` | the compute step: main's `agent_engine/experiment` static analysis of the generated code → compute spec → tier plan (GPU tiers `gpu-*` mapped to `GPU_CATALOG` machines by VRAM / GPU count / host RAM and rentable when the account has a GPU image, S6; CPU tiers on `CPU_CATALOG`; live stock and hourly price per machine) → the review point (`phases/09_compute.request.json` in DeepEvol's ask_user question shape, `09_compute.decision.json` back; without `--ask` the default tier is taken and recorded as `auto`). **Two-stage compute (T3b)**: the default is always the cheapest CPU tier and the decision names `escalation_type` (the smallest T4); a person may still pick a GPU tier up front. The chosen machine is set on the leased port before its first job |
| `entry_smoke.py` | the entry script (named in the blueprint, else `main.py` / `run*.py` / `train.py` / `experiment*.py` at the root or the unique child project); `find_entry` names it for the goal, `run_entry_smoke` is the record-only local-mode probe |
| `experiment_step.py` | step 10 on main's experiment agent (`adr/0002`): the one-line goal RSA compiles its criterion from (the basic template — entry **with its declared options and choices read statically (`entry_flags`, T2)**, GPU or not, minimal scale, unavailable tools, denylist, outputs under `/workspace/out`; nothing from the blueprint), the runner injected into `run_experiment_on_machine` (decision files in the ask_user shape or the unattended defaults; one automatic recompile with the falsifier's reasons; `--ask` hands the pending question back and the flow holds the machine — except G1's asset card, which is policy: the run fails and the machine is released), the machine facts appended to SetupX's per-process copy (SetupX is a black box; the experiment agent's model is `run.json.experiment_model`, `--experiment-model`), `environment.json` (container, committed image, SET_ENV, rounds ledger, SetupX logs, denylist audit), the import preflight before renting |
| `environment_controller.py` | S8: step 10's mechanical scheduler (the owner's four boxes): `ControllerState` + pure transitions (`after_environment` / `after_trial` / `after_repair` / `budget_check`) and `EnvironmentController` driving three fixed worker interfaces — 搭建环境 → `SetupResult`, 远程初步执行 → verdict dicts, 修复代码 → `RepairResult`; routing by `repair.classify`, the agent's attribution and the repeated-signature stop; environment and repair budgets separate; a failure that wants a GPU on a CPU machine stops with `GPU_REQUIRED` (T3b) and `phases._escalate_to_gpu` runs the phase again on the GPU tier at the current commit; no RSA import |
| `repair.py`, `repair_loop.py` | the repair agent (T2: `read_text_file` with line numbers, `edit_file` = one exact unique replacement, `write_file` for new files only, a syntax check on every write, a probe budget, a reminder at call 30 and `finish` only at call 40; the traceback frames' source is in the prompt), `classify` / `gpu_needed`, `failure_signature`, `normalise_ladder_artifacts` (T3: relative `workspace/out/…` artifacts become absolute before RSA freezes the ladder — installed as a `Freezer` subclass for the run), and the RSA-facing workers: `ControllerWorkers` (SetupX `run_round` with the grading contract and kickback; RSA's `Adjudicator` to G2; the agent with commit → move container → re-freeze) and `run_line_pipeline`, the drop-in the runner installs as `rsa.agent.run_pipeline` for the run (RSA's asset gate kept, its Router replaced) |
| `llm_loopback.py` | the loopback `/v1/chat/completions` RSA and SetupX talk to: every call goes through `ParateraProvider` (thinking off, logged under `llm/rsa/`), `response_format` dropped (Flash scrambles it), caliber violations answer 400 |
| `code_repo.py` | `generate_code/` as a git history at `<run>/code.git` (work tree untouched, byte-code excluded), one commit per judged state; `restore(commit)` puts the work tree back to a state (the rerun reset); what the machine's git daemon serves |
| `execution/machine_bootstrap.py` | Docker + git + RSA's two images on a fresh machine from `deploy/experiment-images` Dockerfiles, idempotent (seconds on the pre-baked HK images); the `setupx-base` image is torch-free (T4 parked: torch importable in the bare container defeats RSA's bare falsification); the variant machinery (`nvidia-smi` → cu121, else CPU; labelled, rebuilt on a mismatch) is opt-in with `PAPER2CODE_TORCH_PREINSTALL=1` until T4b; runs inside `PreparedLease.acquire` |
| `submit.py` | `submit`: copy `generate_code/` into `<--dest-root>/<paper>/<trial>/` (default `~/pb_submissions`, the judge's pool) when the four gates passed and either `environment_run` completed (`tree: repaired`) or step 10 never started (`tree: stage9`, the comparison caliber since 09-18); `--snapshot pre_repair|<commit>` exports a state from `code.git`; `submission.json` (sha256 manifest, caliber, tree) and `status.json.submitted_at` |
| (engine) `agent_engine/paper2code/workflows/paper_readback.py` + `source_fidelity.py` | T14 / ADR 0004 (`DEEPCODE_PAPER_FIDELITY=1`, the line's default): the blueprint is the binding; Section 2 paragraphs map files to `Source: §x.y` pointers, host compilation produces `source_manifest.json`, and `read_paper(file_path=…)` walks every page of every bound section. `write_file` is refused until all pages were read in an earlier model turn (`SOURCE_READ_REQUIRED`). The end-of-implement audit records receipt coverage and tree drift; it does not fail the phase. |
| `footprint.yaml` | the line's mentions outside its own paths (guarded by `tests/v2_paper2code/test_footprint.py`) |
| `PITFALLS.md`, `HANDOFF.md`, `HANDOFF-STAGE9.md`, `PLAN.md`, `PLAN-2.md`, `PLAN-3.md` | every pitfall with its knob or test; the state, open issues, to-do and run records; the recipe for another session to run a paper to step 9 without stepping on this one; the three batch plans (PLAN-3 is current) |
| `scripts/paper2code_canary.py`, `scripts/paper2code_bake_gpu_image.py` (repo root) | the CLI (`init / run / step / rerun / status / release / submit / relocate`; `rerun --phase environment_run` resets the tree to the stage-9 commit unless `--keep-tree`); the image bake (GPU by default, `--cpu` for the CPU image) |
| `vendor/paperbench-judge/` (repo root) | the PaperBench judge as vendored from the validation repository (`VENDOR.md`); `judge.sh setup` clones PaperBench at the pinned commit and applies the patch, `judge.sh grade` scores the pool |

## Commands

```bash
PY=.venv/bin/python
R=~/Documents/search/paper2code-runs      # keep run directories outside the repository
E=~/Documents/env                         # paratera.env (model key), aliyun.env (ECS + the two image ids); never printed

$PY scripts/paper2code_canary.py init --run-dir $R/<name> --paper-dir ~/Documents/search/paperbench/<paper> \
  --compute aliyun --run-hours 3 --repair-rounds 3 --figures off|on --planning-fanout     # models default to Vision-Exp; add --ask for the review points
# the 09-19 caliber (DeepSeek official, deepseek-flash, thinking ON — matches the desktop arms and the master baseline's deepseek profile):
#   … --model deepseek-flash --figures-model deepseek-flash --experiment-model deepseek-flash \
#     --provider-base-url https://api.deepseek.com/v1 --provider-key-env DEEPSEEK_API_KEY --thinking enabled   (run with --env-file $E/deepseek.env)
nohup $PY scripts/paper2code_canary.py run --run-dir $R/<name> --until environment_run \
  --env-file $E/paratera.env --env-file $E/aliyun.env > $R/<name>/console.log 2>&1 &
$PY scripts/paper2code_canary.py status  --run-dir $R/<name>
$PY scripts/paper2code_canary.py rerun   --run-dir $R/<name> --phase environment_run --env-file …   # repeat step 10 (an accepted phase is 'completed'; run --until skips it)
$PY scripts/paper2code_canary.py release --run-dir $R/<name> --env-file $E/aliyun.env                # backstop after any kill: proves nothing is rented
$PY scripts/paper2code_canary.py submit  --run-dir $R/<name> --paper <paper> --trial <label> [--dest-root DIR]   # the tree into DIR/<paper>/<label>/ (default ~/pb_submissions); stage-9 or repaired; --snapshot pre_repair|<commit> from code.git
```

Step 10 runs on the CPU tier first; when the run's own evidence says it needs a GPU (a CUDA-class failure, SetupX's
FINISH, the repair agent's word) the phase releases the machine, rents the T4 and starts the boxes again at the
current commit (`phases/10_environment_run.escalation.json`).

**Validating step 10 from a stage-9 tree** (the phase is a separate step; nothing before it has to run again):

```bash
# a run that stopped at stage 9 (this session's or another's, see HANDOFF-STAGE9): step 10 is simply the next phase
$PY scripts/paper2code_canary.py run --run-dir $R/<name> --until environment_run --env-file $E/paratera.env --env-file $E/aliyun.env
# a run that went through step 10 already: rerun it from the same starting point — generate_code is put back to the
# first commit of code.git (the tree as first judged = the stage-9 tree), the repair rounds stay in the history,
# rsa/ and environment.json move aside as .superseded.<stamp>; --keep-tree skips the reset
$PY scripts/paper2code_canary.py rerun --run-dir $R/<name> --phase environment_run --env-file …
# a copied or moved run directory: the absolute paths inside it (phases, dir_info, status, …) still name the old place
$PY scripts/paper2code_canary.py relocate --run-dir $R/<copy>          # then run / rerun as above
```
Each step-10 run rents a machine (CPU tier ≈ 1.4 CNY/h; SetupX 8–20 min, then the trial and up to `--repair-rounds` repairs); `release` after any kill.
With `--figures on` the paper's assets must be real bytes (PaperBench ships LFS pointers; sapg and pinn were hydrated from HF): a figures-on run that describes no figure fails at intake (`phases.figures_on_but_undescribed`) instead of going on as an unlabelled figures-off run; `auto` and `off` never refuse.

Judging lives in the validation repository: put one tree per arm under `~/pb_submissions/<paper>/<arm>/`, then
`PATH=$HOME/Documents/search/.tools/bootstrap/bin:$PATH PAPER=<paper> bash paperbench/scripts/run_grade.sh`
(defaults = the 09-21 caliber above: SiliconFlow V4-Flash whole tree thinking off, V4-Pro parser; key from `~/Documents/env/siliconflow.env`; Docker up;
`num_invalid_leaf_nodes ≤ 2` or the score is void). Copy the `grade.json` to `~/Documents/0919-test/grades/<paper>_<arm>_<runid>…` and the number into the
validation repository's `docs/RESULTS-HISTORY.md`. The DeepCode baseline arm is `paperbench/scripts/run_trial.sh` there (HANDOFF §2).

`--compute local` runs jobs on this machine's Docker and exists for tests.
`--ask` stops at the review points: `plan_review` (request `phases/04_plan_review.request.json`, decision `04_plan_review.decision.json`), `compute` (`09_compute.request.json` / `09_compute.decision.json`, `{"action": "approve" | "cancel", "answers": ["economy" | "standard" | "gpu-economy" | "ecs.*"], "run_hours": <optional>}`) and step 10's questions (`10_environment_run.request.json` / `.decision.json`; the machine is held meanwhile). All requests use DeepEvol's `ask_user` question shape so the product can render them with the same card later (T8). Without `--ask`: plan as is, the CPU tier, `repair_review` accepts.

## Run directory

```
run.json  status.json  events.jsonl  canary.log  dir_info.json
input/paper.md          paper.md + "# Addendum" (sha256 in run.json); with figures described: the enriched text, benchmark bytes in paper.raw.md, cache figures.json
workspace/.deepcode... engine workspace; task dir = workspace/tasks/paper_<run-id>/
phases/<nn>_<name>.json every phase attempt (+ .attempts.jsonl history)
llm/<seq>.json          every model call (no key)
jobs/<seq>/             every remote job: job.json, docker.txt, stdout/stderr, result.json
lease.json  secrets/    the machine (instance id, timeline); root password store and SSH key (0600)
code.git/               generate_code/ as a git history: round 0, then one commit per repair round (submit --snapshot reads it)
environment.json  rsa/  step 10's record (container, image, SetupX rounds, controller rounds with the agent's calls / probes / edits) and RSA's store, adjudications, SetupX logs
llm/rsa/  llm/repair/   the loopback's calls (RSA + SetupX) and the repair agent's
```

## Tests

```bash
.venv/bin/python -m pytest tests/v2_paper2code -q
```

The suite is offline (213 tests; `tests/v2_paper2code/conftest.py` sets `DEEPCODE_PAPER_FIDELITY=0` for every test because the driver's `setdefault` would otherwise leak the line default into the offline fakes, which write free-text plans — the fidelity tests switch it on themselves) except `test_job_executor.py`, which needs a local
Docker daemon and a `python:3.11` image (it picks a present one or
`PAPER2CODE_BASE_IMAGE`). `tests/data_contracts` holds the footprint and provider-egress guards.
