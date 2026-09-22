# HANDOFF — Paper2Code line (updated 2026-09-22 afternoon)

Read this first; then `STATUS-2026-09-21.md` (the line as the code has it, one page), `PLAN-3.md` (§0 decisions, §2.3 the
build order), `PITFALLS.md` (every trap with its knob or test), `README.md` (layout, commands). Terms are the root `CONTEXT.md`'s.

## 0. Where things are on 2026-09-22 (start here; everything below §0.-1 is the earlier record and still holds)

Line head `3e9552360` on `0916onmain-experiment` (pushed, tree clean, 224 offline tests). Nothing rented. Validation repository
`~/Documents/env/paperbench-judge/validation`: master `5890c13` (judge), branch `0919-test` `c23ba85` (desktop-arm rules; owner's
Codex clone of it at `~/Documents/0919-test/codex`). Git identity is now `Yuwei Qiu <84381164+2UBBISH@users.noreply.github.com>`
(global config; every earlier "apple" commit was rewritten and force-pushed on 09-21 — other checkouts must `git fetch && git reset --hard origin/<branch>`).

### 0.-2 09-21/22 in one table (newest first)

| What | Where |
| --- | --- |
| **Model identity (owner 09-22)**: the official `deepseek-flash` alias now serves **DeepSeek-V4.1-Flash**, not V4-Flash (the API does not expose the version; `/models` lists only `deepseek-flash` / `deepseek-v4-pro`). Every 09-19+ run on api.deepseek.com — fre/rice t17, the 18 t19 trees, fre-t18, robust-clip-t20, and the owner's Codex desktop runs — is V4.1-Flash; the two arms of the PaperBench comparison are therefore on the same model. The 09-19 SA-Bench three-arm run was Paratera `DeepSeek-V4-Flash` (V4) — a C-arm re-run on the official channel would be a different model from its A/B arms, so SA-Bench must be re-run three arms together on one serving (stopped 09-22 14:59 for this reason; scripts in `paper2code-runs/sabench-t2/`). Also: SiliconFlow has no GLM-5.3-Flash and its `zai-org/GLM-5.3` ignores thinking-off | this table |
| **Validation repository restructured (09-22 afternoon, `d2da0ad`)** into four top-level directories: `paperbench/` (judge = frontier-evals + patches + `scripts/run_grade.sh` / `run_judge_eval.sh` / `run_trial.sh`, `baseline-deepcode/`, docs, `runs/` ignored), `deepevol-deepcode/<paper>/{submission,RUN_NOTES.md}` (this line's 20 stage-9 trees + `fre-nofid`), `result/` (grades, Codex trees + audits, score table via `update_readme.py`, RESULTS-HISTORY, probes), `materials/` (the shared paper inputs — robust-clip hydrated — and the desktop prompt kit). The judge's own robust-clip `paper.md` is the hydrated one too. `~/Documents/0919-test/` stays as the local archive (session logs, datasets) | validation `README.md` |
| **Three-arm archive**: every submission, run note and grade now lives outside both repositories in `~/Documents/0919-test/` — `deepcode/<paper>/` (18 t19 trees + fre/rice t17 + `fre-nofid` control, each with `RUN_NOTES.md`), `codex/results/<paper>/codex/` (owner's desktop runs: submission, AUDIT, session logs), `grades/<paper>_<arm>_<runid>_sf_tree_thinkoff_pro.grade.json`, `README.md` (layout, judge caliber, JudgeEval table, score table rebuilt by `update_readme.py`). Codex trees done: 14 papers + robust-clip re-run | `~/Documents/0919-test/` |
| **Judge caliber, final (09-21 16:40, RESULTS-HISTORY §1.5)**: SiliconFlow `deepseek-ai/DeepSeek-V4-Flash`, whole submission tree in every leaf prompt (`PB_JUDGE_WHOLE_CODEBASE=1`, prefix cache ≈98 %), thinking off; parser `deepseek-ai/DeepSeek-V4-Pro` (json_object, instance-not-schema guide, 3 attempts). JudgeEval rice/0 0.70–0.72 with 178/178 valid = level with the old top-10 caliber. ¥32 a 306-leaf paper, half price 02:00–08:00. Scores judged before 09-21 (Paratera top-10) are not compared with these | validation `run_grade.sh` defaults; `runs/judge_eval/0921c_*` |
| **Same-caliber scores, 8 papers paired** (deepcode / Codex): fre 0.961 / 0.914 · rice 0.978 / 0.976 · pinn 1.000 / 1.000 · lbcs 0.987 / 0.993 · lca 0.925 / 0.898 · what-will 0.990 / 0.988 · robust-clip (truncated md) 0.642 / 0.906 · **robust-clip† (hydrated md) 1.000 / 0.941**. Codex-only: bam 1.000, ftrl 0.602. Line ahead 3, level 4, behind 0; four papers pinned at ≈1.0 on both arms (rubric ceiling). Single-paper noise ≈0.025–0.05 — one tree per paper per arm, so direction not magnitude | `0919-test/README.md`, RESULTS-HISTORY §1.6 |
| **fre ablation** (same day, same caliber, only `DEEPCODE_PAPER_FIDELITY`): on 0.961 / off 0.862 / Codex 0.914 → the blueprint pointers + whole-section read-back + write gate are worth **+0.10**; without them the line trails Codex. Loss with fidelity off sits in the "training & evaluation" subtree (0.68) | `0919-test/deepcode/fre-nofid/`, `paper2code-runs/fre-t18-nofid` |
| **robust-clip**: the official `paper.md` is truncated mid-sentence in §1 (no §2, §3 method, §4 head; §4.1/4.4 and appendix present). Hydrated from the arXiv LaTeX source (2402.12336v2 = the ICML version; `paperbench/robust-clip/hydration/` holds the official copy, `tex2md.py`, the three converted parts and a record). Both arms re-run on the hydrated md: line 0.642 → **1.000** (230 read-backs vs 62), Codex 0.906 → 0.941. Owner's rule kept: **the coding agent never fills gaps from memory** — fixing the input beats loosening the rule. The other 19 papers' top-level sections are contiguous | `paperbench/robust-clip/hydration/README.md`, `paper2code-runs/robust-clip-t20`, STATUS O7 |
| **Memorisation probe** (owner's question "did the model just remember the authors' code?"): authors' repos (`opt_for_pinns`, `GSM-VI`) vs both arms' trees — 0 shared code lines ≥40 chars, shared names only torch/LBFGS argument names and algorithm names, tree sizes and structures unrelated. High scores come from rubrics that ask for paper-stated specs, not from copied code. Probe script: scratchpad `mem/sim.py` (not in the repo; worth keeping as a per-paper check) | this table |
| **t19 batch**: 18 papers to stage 9 overnight 09-21/22 (waves of 6–8 on the official DeepSeek key; index of a 930-file reference repo (adaptive-pruning/LoRA) took 3.5 h — pre-filter fallback, PITFALLS), 18/18 exported, 0 failures. Per-paper stats in `0919-test/deepcode/<paper>/RUN_NOTES.md` | `paper2code-runs/*-t19`, `batch-t19/batch.log` |
| **Night grading** (SiliconFlow half-price window only, 4 workers, `validation/runs/grade_night_v4.sh`, pid in `runs/grade_night_v4.pid`): first night 09-22 banked 13 trees, 0 × 429 at 100 requests in flight; sleeps until 02:00 and continues (≈21 trees left). Three script generations died on: per-paper `runs/<paper>/` missing, deepcode source moved before rsync, and `run_grade.sh` [4/4] copying the newest run group *of any paper* — the last one is fixed in `run_grade.sh` (`5890c13`) | validation `runs/grade_night_0923.log` |
| **Follow-ups O1–O7** recorded, none blocking (owner): glue misclassification (no re-plan — over-correction), addendum sub-headings unmatched, tree size vs judge cap, whole-section obligation untested live, optional-baseline triage, planned files written one directory too shallow (pinn `model.py`, bam `matrix_equations.py`), truncated-input papers | `STATUS-2026-09-21.md` §9 |
| Step 10 is still outside the comparison and has not run on any t17/t19 tree. Prerequisites before it can: T4b wheelhouse (torch not pre-installed), entry smoke on nested packages, data download (lca lost 29 leaves for "loader but no download"), O6 relocation. Suggested first pair: fre, pinn (small, spec-complete, CPU-runnable at reduced scale) | PLAN-3 T4b, `entry_smoke.py`, HANDOFF §0.0c |

### 0.-1 09-20 evening in one table (newest first; the 09-19 sections below still hold)

| What | Where |
| --- | --- |
| fre scores: line1 (t14) **0.8756**, line2 (t15, ADR 0003) **0.8389**, Codex desktop with experiments 0.7847 (void) — the t15 gap is the training node: FB/SF baselines planned as "optional … if time permits", left out of `source_files`, not written | validation `docs/RESULTS-HISTORY.md` §1.3 / §1.4 |
| ADR 0003 exercised live (fre-t15: 43/43 files, 78 receipts, audit passed) and then stress-tested on 50 papers' plans: every failure was in its own layer → **retired the same night for ADR 0004** (row above); the relaxations made on the way (fuzzy quote location, kinds, YAML repair, JSON binding) are gone with it | `source_fidelity.py`, `phases.py` (`plan.replan`, `plan.source_manifest`), `paper_readback.py` |
| Static syntax check + repair after implement (`compile()`, ≤ 2 `edit_file` rounds, inside the FidelitySession) | `syntax_check.py`, `repair.py` (`fidelity=`, `read_paper` tool) |
| Execution rule for the desktop arms, final: commands allowed, only long CPU/GPU training or evaluation is out; audit 10 min / 60 min | validation branch `0919-test` `00d1947` |
| criteria (T9): the owner's colleague builds a paper → rubric-shaped JSON; study of its effect recorded (priority signal, not coverage) — **no code written** | PLAN-3 T9 row |
| Planner stress test: 50 papers (PaperBench 20 + SA-Bench 30) `run --until plan`, 2 at a time, results `~/Documents/search/paper2code-runs/plan-batch/<pb|sa>__<paper>/` (`batch.log`) | scratchpad `plan_one.sh` (not in the repo) |
| **ADR 0004 first live runs (09-21 00:00–01:40)**: rice-t17 stage 9 clean — 59/59 files, 204 receipts, audit passed, syntax 0 → pool `rice/line2`; fre-t17 died once on the implement output cap (19.5k reasoning + 52 KB write_file at 32768 → `DEEPCODE_IMPLEMENT_MAX_TOKENS` now 65536), `rerun --phase implement` → 37/37, 224 receipts → pool `fre/line3`. Pool also holds `fre/codex` (desktop, 09-20) and `rice/line1` (t14). **Grading blocked: the Paratera judge key is out of balance (403 team_model_access_denied, 8 models left)** | `paper2code-runs/{fre,rice}-t17`, `~/pb_submissions/` |
| Plan-only validation on the final code: sapg / bbox / lbcs / lca 4/4 first try, 0 re-plans; pointer parser extended to what planners actually write (number + words, Eq./Table → the section stating it, ranges, parentheses, directory paragraphs); paper-file binding 24/44, 44/60, 20/33, 24/53 | `plan-batch/pb3__*`, `0c5571830` |
| fre-t16 stopped at index (owner: run the planner batch first); Codex desktop fre run graded 09-20 evening → void, judge key out of balance | `paper2code-runs/fre-t16`, `~/pb_submissions_archive/fre/` |

### 0.0 What changed on 09-19 (this line)

| # | what | where |
| --- | --- | --- |
| T13 done | zero-byte planned files: `repair.empty_files`, recorded by `implementation_status`, named to the repair agent; pinn ×3 flag `opt_for_pinns/src/pdes.py`, sapg clean | `306951555` |
| figures-on refusal | `--figures on` fails intake when no figure was described (LFS-pointer assets); `auto`/`off` never refuse | `b2e82023a` |
| step 10 standalone | `rerun --phase environment_run` resets `generate_code` to the first commit of `code.git` (the stage-9 tree; `--keep-tree` opts out); `relocate` rewrites a copied run's absolute paths; a stage-9 run from another session is just `run --until environment_run`. Verified on a copy of `sapg-s9-off` (`paper2code-runs/sapg-s9-off-copy`, ready for step 10, nothing rented) | `ca1b02174` |
| references cap | 40 iterations exhausted → one automatic retry at 80 (`iterations_retry`, event `references.retry`); default stays 40 (baseline parity) | `4a84366b2` |
| bundle upload | `default_serve_repo` retried 3× with a fresh runtime (the pinn-on `GitDaemonError` case) | `4a84366b2` |
| README | state 09-19, quick start "generate to stage 9 and export with `submit --dest-root`", step-10 validation recipes | `298756f12`, `ca1b02174` |
| **No execution before step 10** (owner 09-20) — **09-20 evening: quick checks yes, experiments no, and the line gets a static syntax check** | The Code-Dev comparison rule. Evening version: the desktop agents run normally, the prompt tells them the code runs remotely later and no experiment can run locally (quick checks fine), the audit lists what ran with wall time (> 5 min one command / > 30 min total = void); the line's equivalent of `py_compile` is `syntax_check.py` after generation (compile-only, ≤ 2 `edit_file` repair rounds without a container, `DEEPCODE_SYNTAX_CHECK` / `DEEPCODE_SYNTAX_ROUNDS`; fre-t14 shipped `prior.py:503` `device=` twice and the judge docked the leaf). Daytime version (superseded, kept for the record): the desktop agents were blocked mechanically (Codex execpolicy rules file, Claude workspace deny-Bash; the 09-19 fre / rice Codex runs that trained on CPU for 177 min are void); the line's coding agent never had execute tools (`write_file` / `search_code_references` / `read_paper`; the structure agent's mkdir/touch is not execution) and upstream's post-generation `python3 -m pytest` is now off by default (`PAPER2CODE_IMPLEMENT_VERIFY=1` restores it), reported as `inner_status: unverified` / `verification_disabled`. Step 10 (rent + run + repair) stays outside the comparison. `fre-t14` / `rice-t14` were produced under this rule already (their one pytest attempt found no pytest and changed nothing) | `phases.py` |
| **T14 paper fidelity — ADR 0004 (09-20/21, supersedes 0003)** | The blueprint is the binding: Section 2 paragraphs name their files and `Source: §x.y`; host compilation produces `source_manifest.json` with files, sections, glue and unmatched pointers. `read_paper(file_path)` walks every page of every bound section; `write_file` requires all pages to have been read in earlier model turns. The audit records receipt coverage and tree drift in `implement.paper_fidelity`; it does not fail the phase. A plan with no `Source:` line is re-planned once. **Live t17:** fre 37/37 files, 224 receipts; rice 59/59, 204 receipts; syntax 0 and audits passed. `4b5cac6ab` tightened the obligation from one page per section to every page. | `source_fidelity.py`, `paper_readback.py`, `phases.py`, `adr/0004-*.md` |
| parked / not doing (owner) | T4b wheelhouse (bare images stay: SetupX round 1 = 12–66 min of torch), T9 criteria-into-blueprint (design kept in the PLAN-3 row), T3c, "read the PDF with Flash" (no V4.1 Flash exists; official `deepseek-flash` does take images — a product-side transcription experiment for later) | PLAN-3 §2.4 |

### 0.0b The validation repository on 09-19 (the other half of the day)

- **PaperBench vendored** into master as plain files (`frontier-evals/`, upstream `51052ce`, 5-file patch, all 20 official
  papers hydrated and hash-verified by `patches/verify_paperbench.sh` → `VERIFY_OK`); all 20 registered in the judge
  (`paper_split` Literal + splits) and in `run_trial.sh`. `check_paper_md.py`: only `robust-clip`'s official md is
  incomplete (no §2–§3).
- **`deepcode_config` `deepseek` profile** (api.deepseek.com, `deepseek-flash`, `DEEPSEEK_API_KEY`; switch with
  `DEEPCODE_REGEN_CONFIG=1 DEEPCODE_CONNECTION=deepseek DEEPCODE_MODEL=deepseek-flash bash setup.sh`).
- **Branch `0919-test`** (collaboration, 74 files): the 20 papers' `paper.md`/`addendum.md`/`blacklist.txt` only,
  official Code-Dev instructions, `desktop/` (prep → the owner drives the Codex app / Claude desktop Code tab → finish
  audits from the apps' own session logs), `cli-reference/` (CLI + thinking-off proxy, not the batch caliber), plain
  README. Caliber: `deepseek-flash` @ api.deepseek.com, **thinking ON** (nothing can turn it off in the desktop apps
  without a body-rewriting proxy; DeepSeek ignores Codex's `effort=none` when the UA / `x-codex-*` headers are Codex's),
  same bytes for every arm (no PDF/assets/rubric; one sentence of the official prompt changed to "in markdown
  format"), official `ADDITIONAL NOTES` with the official 3-hour `time_limit_template`, continue message ≤5. Results
  `results/<paper>/<arm>/`, handed back as a tarball. The DeepCode arm of that comparison runs on master
  (`run_trial.sh`, deepseek profile, `compat.thinking` there still `disabled` — flip to `enabled` before running it,
  the branch's copy already is).
- `docs/CODEDEV-ARMS.md` = the one-page caliber with code citations (instruction file vs rubric, how each arm plugs in).

### 0.0c Step-10 to-do (owner 09-19 night; PLAN-3 §0 table)

| # | what | state |
| --- | --- | --- |
| ① references retry | done | |
| T14 paper fidelity | done offline and exercised on fre-t17 / rice-t17; next: normalize judge caliber, finish pending grades, then expand stage-9 sample | |
| ② T4b wheelhouse | **not doing** | |
| ③ bundle upload retry | done | |
| ④ pinn step 10 from round 0 | **next**: `rerun --run-dir paper2code-runs/pinn-off --phase environment_run --env-file paratera.env --env-file aliyun.env`; watch whether T13's hint gets `pdes.py` filled and whether G2 passes. Waiting for the owner's go (Paratera key is shared with SA-Bench; step 10 is ~150–250 sequential calls over 1–2 h) | |
| ⑤ stop rule review | after ④: is "same signature → stop" too early when the agent changed the root cause? | |
| ⑥ step-10 sample | 5 of the 20 stage-9 trees through step 10; record SetupX time, G2 rate, repair rounds, escalations; keep the artifacts here | |
| ⑦ T3c | **not doing** | |

## 0x. Where things are on 2026-09-18 evening (the previous state; §0.4–0.5 the day's real runs, §0a / §0b the overnight record)

**PLAN-3 §2.3 S1–S9 and §2.4 T1–T4 are `done`** (T3b two-stage compute included; T3c deferred). Branch
`0916onmain-experiment` on `HuigenYe/DeepEvol`, worktree `search/DeepEvol-Paper_repro_0916/DeepEvol1.0`, pushed, tree
clean. Validation repository `~/Documents/env/paperbench-judge/validation` master `939f8fb`, pushed. No Aliyun
machine rented (`release` backstop after every run); no background job running. Machines today ≈ 35 CNY.

### 0.1 State

| item | state |
| --- | --- |
| tests | `tests/v2_paper2code` 191 offline (+5 Docker-gated), ruff clean; footprint / egress guards green (`tests/data_contracts`); validation `patches/verify_deepcode.sh` green (16 files) |
| models | every slot defaults to `DeepSeek-V4-Flash-Vision-Exp` (`run.json.model` / `figures_model` / `experiment_model`); `DeepSeek-V4.1-Flash` still `403 team_model_access_denied` on both keys; judge `DeepSeek-V4-Flash`, parser `DeepSeek-V4-Pro` |
| keys | `~/Documents/env/paratera.env` (= `paratera_backup.env`), `aliyun.env` (now also `DEEPEVOL_API_ALIYUN_GPU_IMAGE_ID=m-j6c6925byfwdjv97nokt`); only via `--env-file`, never printed |
| images | **in use**: CPU `m-j6c7r9v1zfzlibaknmmd`, GPU `m-j6c6925byfwdjv97nokt` (no torch). Parked T4 images with torch 2.1.2 baked in: CPU `m-j6c5c9qjwym10zee9gol`, GPU `m-j6cem6p6pg13c5rg9dvo` (they defeat RSA's bare falsification; T4b will replace them with a wheelhouse). Oldest GPU `m-j6c9bys3x83hji55jfha` — **owner: leave them all** |
| step 10 | two-stage compute: starts on the CPU tier (`ecs.c7.xlarge`), escalates to the smallest T4 on run-time GPU evidence (T3b, verified §0.5); SetupX round 1 8–12 min on the CPU tier (torch CPU wheel 2 min; T4 parked); the owner's four-box controller (`environment_controller.py` + `repair_loop.run_line_pipeline` installed as `rsa.agent.run_pipeline` for the run); SetupX black box (60 steps × 3 rounds), trial = adjudicate ≤ G2, repair agent ≤ 40 calls / 15 probes / 3 rounds with `edit_file` (T2: exact unique replacement, read-before-edit, `write_file` new files only, `.py` writes parsed, call 30 reminder, call 40 = `finish` only, traceback-frame source in the prompt); the goal carries the entry's real options and choices (`entry_flags`); a stop that is not a pass = `repair_review`, default accept; **owner: the repaired tree is the product** (`submit` default; `--snapshot pre_repair` is the comparison caliber only) |
| runs | `sapg-2` (S7 shakedown, six attempts, last one accepted at G0 with two re-pinned ladders); `sapg-s9-off` and `sapg-s9-on` (S9, four gates green, environment_run = 1 environment + 3 repair rounds each, G2 not passed, accepted, released); all in `~/Documents/search/paper2code-runs/` |
| scores (S9 + T1, Flash judge, 77 leaves) | baseline `vexp1` **0.7156** / `vexp2` **0.6910** · line figures-off pre-repair **0.6677** / post **0.6365** · figures-on pre **0.6414** / post **0.6709** (§4; validation `docs/RESULTS-HISTORY.md` §1.2 has the leaf overlaps). **Run-to-run noise ≈ 0.025; every S9 delta sits inside it.** All six graded copies in `~/pb_submissions_archive/sapg/`; `~/pb_submissions/sapg/` empty |
| spend overnight | machines ≈ 60 CNY (T4 9.53 CNY/h: six S7 attempts ≈ 2.5 h, S9 four attempts × 2 ≈ 3.5 h, two bakes 0.7); judge four submissions ≈ 15M Flash tokens; model tokens for the three full runs ≈ 27M |

### 0.2 Open issues (found overnight, not fixed)

1. ~~**Repair rounds cost score** (off: 0.668 → 0.637).~~ **Withdrawn after T1**: the figures-on run went the other
   way (0.641 → 0.671, +0.030) and the baseline's two identical runs differ by 0.025 — the repair delta is noise, and
   the "rewrote the entry files, rubric counts it as regression" story was hindsight. What stands: the loop's *behaviour*
   (whole-file rewrites, 28–34 probes per round before `MAX_PROBES=15`, junk files, G2 never reached) is still worth
   fixing, but its acceptance is mechanical (G2, diff size, probe count), not a score.
2. **G2 never passed on sapg** in any of the eight controller runs. Failures move each round (module missing → CLI
   argument → shape mismatch → artifacts absent), i.e. the generated code is far from running, not a loop defect.
3. **RSA's compiler ignores the goal's output rules**: it declared the output directory an asset (now dropped before G1)
   and writes artifact paths relative to the repository (`workspace/out/…`, `figures/…`, `runs/…`). The trial `mkdir -p`s
   them and routing ignores artifact tails now, but the criterion's shape is still the compiler's whim (RSA source is
   off limits, PLAN-3 §3).
4. **SetupX's first round is 10–25 min** on a T4, mostly torch's ~3 GB of CUDA wheels (tuna 3–11 MB/s). A pre-installed
   torch in `setupx-base` (or a wheelhouse on the image) would cut every run by ~10 min; it changes main's Dockerfile.
5. **Figures on vs off shows nothing on sapg**: pre-repair −0.026 (0.668 vs 0.641, leaves 38 shared / 15 vs 14 own),
   post-repair +0.034 — both inside the noise; sapg's four figure subtrees are 0 / 0.25 / 1 / 1 on every submission, only
   Fig. 5 moves. A verdict on figure descriptions needs a paper whose rubric depends on the figures (T5).
6. **Run-to-run noise ≈ 0.025** (`vexp1` 0.7156 vs `vexp2` 0.6910: same code, same input, no truncation in either log;
   leaves 41 shared, 6 / 16 own — `vexp1` passes *fewer* leaves and scores higher because Fig. 5's subtree weighs 0.75
   vs 0.25). Any single-run delta under ≈ 0.05 on sapg is not a signal.
7. `sapg-s9-on`'s repair agent left junk files in the tree (`main.py.flatpatch`, `_patch_import.txt`) that went into the
   product; the write tool could refuse non-source paths.
8. Monitors: `tail -f` + `cut` buffers; `tail -f` over two files prints headers — use `tail -q` and no `cut` (PITFALLS).

### 0.3 To-do (owner's order 09-18 22:30: T13 → T4b → pinn noise sample → the rest)

| # | what | size | why / acceptance |
| --- | --- | --- | --- |
| T1 | ~~Grade `s9on_pre` and `vexp1`~~ **done 09-18 10:28** (5 min, 0 invalid leaves): `s9on_pre` 0.6414, `vexp1` 0.7156 | small | §4 and RESULTS-HISTORY §1.2 updated; grade files also at `archive/deepcode_test/sapg/grades/{s9on_pre,vexp1}_flash.grade.json` |
| T2 | **done 09-18 12:31** — real-machine result below (§0.4). Offline (owner's decisions): the goal carries the entry's real option list (`experiment_step.entry_flags`, ast, no execution); the agent gets `edit_file` (exact unique `old_string` → `new_string`, read-before-edit, numbered reads) and `write_file` creates new files only; extension allowlist; `.py` writes are `ast.parse`d and refused whole on a syntax error; a reminder to `finish` at call 30, `finished` / `probes_after_write` / `created` / `edited` in the round record; prompt order read → change → verify with the traceback frames' source (±20 lines) in the prompt; `MAX_PROBES` stays 15. Facts behind it: PITFALLS §D (four new rows) | medium | **mechanical acceptance only** on one `sapg-s9-off` rerun (`rerun --phase environment_run`, ≈ 45 min T4): every round `finished`, no non-source files, diff per round recorded, the failure moves forward (no agent-made NameError / ModuleNotFound), G2 reached or not; scores are not a criterion on one paper (noise 0.025) |
| T3 | **done 09-18 18:34** (verified on the same real run, §0.5). Offline: `repair_loop.normalise_ladder_artifacts` runs inside a `Freezer` subclass installed for the run (`install_artifact_normaliser`, restored after) — relative `workspace/out/…` → absolute, and a relative artifact whose basename twins an absolute one under the command's output directory is dropped; RSA untouched. Event `controller.artifacts_normalised` | medium | no `workspace/out` relative paths in `rsa/store/*/frozen.json`; verified on the same real run as T3b |
| T3b | **done 09-18 18:34** (real run below, §0.5). **Two-stage compute** (owner's decisions 09-18 afternoon, all as recommended): `compute` defaults to the cheapest CPU tier whatever `needs_gpu` says and records `escalation_type` (the smallest T4); the controller stops with `GPU_REQUIRED` on run-time evidence only — a CUDA-class failure text (`repair.GPU`), SetupX's FINISH saying GPU-only, the agent's environment attribution naming a GPU — never on the spec; on a CPU machine that stop is not a review point: `phases._escalate_to_gpu` releases, sets `rsa/` + `environment.json` aside as `.cpu_stage.<stamp>`, writes `10_environment_run.escalation.json` and runs the phase again on the GPU machine at the current commit (goal recompiled with GPU facts); on a GPU machine the same signal is a code failure. Fractional GPUs (`sgn7i-vws` 1/12 A10, 2.51 CNY/h) are not sold in cn-hongkong and need GRID drivers → T4b, not doing | medium | offline 191 tests; real: a copy of `sapg-s9-off` with the entry forced to `cuda` — CPU stage fails on CUDA, escalates, GPU stage passes |
| T4 | **parked 09-18 21:50** — torch in the image makes any G0 of "torch + own package" pass RSA's bare falsification (pinn: criterion rejected twice, the controller never started; the compiler derives G0's imports from the entry and ignores the goal). The torch layer and the bootstrap variant are opt-in (`PAPER2CODE_TORCH_PREINSTALL=1`); `aliyun.env` is back on the torch-free images `m-j6c7r9v1zfzlibaknmmd` / `m-j6c6925byfwdjv97nokt`; the torch images stay. Next shape = **T4b wheelhouse** (wheels cached in the image, torch not installed). What was measured before parking: `Dockerfile.setupx-base` pre-installs torch 2.1.2 + torchvision 0.16.2 behind `TORCH_INDEX_URL` (cpu default, cu121); `machine_bootstrap` decides the variant on the host (`nvidia-smi -L`), labels it (`paper2code.torch_variant`) and rebuilds on a mismatch; the machine facts name the pre-installed build; `paper2code_bake_gpu_image.py --cpu` bakes the CPU image and both bakes run the line's bootstrap so the snapshot carries the labelled image. Images: GPU `m-j6cem6p6pg13c5rg9dvo` (container `torch 2.1.2+cu121`, sees the T4), CPU `m-j6c5c9qjwym10zee9gol` (`2.1.2+cpu`); in `aliyun.env`; the three older images stay (owner). 2.1.2 because nine of the ten torch pins the generated repositories wrote admit it | small | measured on `sapg-s9-off` (CPU tier): machine prepared in 41 s (`built=no`), SetupX round 1 **7.7 min with 0 min of torch** (12–66 min before); the remaining minutes are SetupX reading the import layout (28 steps) — the black box, left alone. That run then went env → repair (import path + artifacts, +124 −12) → `mat1 and mat2` → repair (+18 −2) → same signature → stop, accepted; every round `finished`, no junk — the loop's third real sample, one of three not reaching G2 |
| T5 | **first pair done 09-18 22:16** (validation `docs/RESULTS-HISTORY.md` §1.3): pinn baseline 0.6700 / line figures-off pre-repair 0.7083 / figures-on pre 0.6696 (Flash, 126 leaves) — same band as sapg, the line +0.038 over the baseline (a shade above sapg's 0.025 noise; pinn's own noise unmeasured), figures on −0.039 again, Fig. 4/5 subtree 0.16 on all three. Step 10 reached G2 on neither line run (`off`: repair round correctly diagnosed a **zero-byte `src/pdes.py`** but the trial repeated → stop; `on`: `GitDaemonError: connection lost` uploading the bundle after the repair — this Mac's network); owner stopped both at 22:25, machines released. Prep that stays: pinn registered in the judge (`paper_split`, `splits/pinn.txt`, `run_trial.sh` title key + blocked repo), assets hydrated from HF | large | remaining: post snapshots (optional), a pinn noise sample (baseline ×1) |
| T13 | **done 09-18 23:20** — zero-byte planned files: `repair.empty_files(code_dir)` lists 0-byte source files with the non-empty same-named files elsewhere in the tree (the engine's habit: content at one path, the planned twin left empty); `gates.implementation_status(result, code_dir=…)` records them as `detail.empty_files` (recorded, never failing; event `implement.empty_files`), and the repair prompt names them with the instruction to fill the file or move the twin's content in. Checked on the real trees: pinn ×3 (baseline too) → `opt_for_pinns/src/pdes.py` (twin `src/pdes.py`), baseline also an empty `README.md`; sapg ×3 clean | small | 194 tests; a real sample comes with the next step-10 run on pinn |
| T6 | When `DeepSeek-V4.1-Flash` is granted: flip the three defaults, re-probe the vision path (`figures.py` probe), one figures-on run | small | HANDOFF §0 + PLAN-3 §0 "模型" |
| T7 | Retire the execution port (PLAN-3 §0 "退役": `job_executor.py`, `leased_runtime.py`, the lease half of `aliyun_lease.py`, `entry_smoke.run_entry_smoke`, `tools/execute.py`, `verification_hook.py`), keep `release` and `find_entry`; the local docker mode's record-only path goes with it | medium | fewer moving parts before item 9 |
| T8 | PLAN-3 item 9: product integration (workflow run, checkpoint, `ask_user` cards from the decision files, Gateway provider) | large | owner's call when to start |
| T3c | Two-stage compute, deferred by the owner (09-18 afternoon: "not now, record it, keep CPU first / GPU when it fails"): ① let item 8's *hard* static evidence start on the GPU directly (cuda literals with no `cuda.is_available()` anywhere in the tree; or a spec tool that is `installable=false` and needed at import) — the analyser today marks `is_available()` itself as GPU use, so this needs its own detector; ② **time as GPU evidence**: a CPU trial longer than X (proposed 300 s; sapg trials take 22 s) escalates instead of repairing, a CPU stage longer than Y (proposed 30 min) escalates at the next decision, a slow *pass* is recorded (`duration_s` is in every verdict) but not escalated. The owner's worry: a paper that passes on CPU slowly stretches evaluation and product time; the bigger lever for wall clock is T4 (torch pre-installed) | small | thresholds to be set with the owner |
| — | Not doing (owner): `optimize` stub; deleting the old GPU image; changing the product rule |

### 0.4 T2 real-machine result (sapg-s9-off, `rerun --phase environment_run`, tree reset to round 0, T4, 11:45–12:31)

**G2 passed 6/6 on the third repair round** — the first pass on sapg in nine controller runs. Mechanical metrics
(`environment.json.repair.rounds[*].agent`, diffs from `code.git`):

| round | calls | probes (after 1st change) | files / diff | finished | trial after it |
| --- | --- | --- | --- | --- | --- |
| env (SetupX, 19 min) | 66 steps | — | — | — | `IndexError` in `networks.latent` (0/6) |
| repair 1 | 40 | 13 (13) | 3 files, +14 −12 | no | `mat1 and mat2 shapes cannot be multiplied` (0/6) |
| repair 2 | 40 | 14 (8) | 3 files, +118 −13 | no | run completes, artifacts absent (1/6) |
| repair 3 | 17 | 6 (2) | 1 file, +47 −21 | **yes** | **PASS 6/6** |

Zero whole-file rewrites, zero non-source files, zero agent-made errors (S9 had `NameError` / `ModuleNotFound`
of its own making in two of six rounds); every failure moved forward. Two things the run taught, both landed:
the compiler invented a *value* once it could not invent an option (`--task dummy` → `entry_flags` resolves
`choices=`, `f37f6b0bd`; the first attempt was killed at 35 min for it), and the call-30 reminder did not stop
rounds 1–2 (the 40th call is now `finish` only). Two things left open: SetupX spent 20 of its 19+ minutes
patching code in the container that the trial's checkout discarded (candidate addendum fact "code defects are
repaired by another agent after you; do not edit repository files" — owner's call, PLAN-3 §0 keeps the addendum
to machine facts); and round 3's fix mirrors the artifacts into the repository-relative path the compiler wrote
(`/workspace/repo/workspace/out/smoke/…`) — the T3 shape problem, now visible in a product. Not graded (T1:
one-run scores are noise); `submit` would put the first G2-passing sapg tree in the pool if wanted.
Machines: 46 min + 35 min (aborted) ≈ 13 CNY; `release` backstop clean.

### 0.5 T3 + T3b real run (`sapg-t3-esc`: a copy of `sapg-s9-off` at round 0 with a CUDA gate on import, 16:31–18:34)

Seven launches to get one clean end-to-end sample; the first six each taught one thing (PITFALLS §D):
a copied run directory carries absolute paths in its phase records (rewrite them); a soft `--device` override
falls back to CPU; the repair agent deletes an unconditional CUDA gate it sees in a probe ("spurious hard-gate",
twice); SetupX's machine facts were written once per process (fixed, `fd797457a`); a `CUDA` failure text must outrank
the code's own missing-module text in `classify` (fixed, `324234fe1`); G0's declared imports (`experiments`) resolve
from the repository root and SetupX sometimes papers over it with `PYTHONPATH` — a criterion-shape lottery.

The seventh: **CPU stage** `ecs.c7.xlarge` 19 min (SetupX 12 min incl. CPU torch 2 min, G0 trial → `gpu required:
the code wants CUDA and this machine has no GPU`, no review point, released, 0.45 CNY) → `experiment.escalate` →
**GPU stage** `ecs.gn6i-c4g1.xlarge` at the same commit, 1 h 43 min (SetupX 66 min — 3 GB of CUDA wheels at ~1 MB/s
plus its fight with the test fixture's `experiments` alias; the machine facts said "NVIDIA GPU" and it installed
`torch 2.1.2+cu121`, `CUDA available`; G0 passed; G2 failed on the original `IndexError`) → repair ① 40 calls /
14 probes / 3 files +96 −8 / `finish` → `flat_all` missing → ② 38 / 11 / 4 files +61 −21 / `finish` → 5/6 (only
`smoke_history.json` lacked the compiler's `iterations` key) → ③ 17 / 12 / 1 file +15 −2 / `finish` → **G2 6/6**.
Every round finished; artifacts in the frozen criterion were absolute `/workspace/out/smoke/…` (T3) and the code
wrote them where the command said — no mirroring into the repository this time. Cost of the day's verification line:
CPU stages ≈ 2 CNY, GPU stages ≈ 17 CNY (the 66-min torch install is the T4 item's argument).

Rules that stay: **PaperBench comparisons on V4-Flash stop at stage 9** (`run --until compute`; `submit` takes the
stage-9 tree, `tree: stage9`; step 10 is exercised in its own runs, never in the comparison samples — owner 09-18 22:30);
run only through `scripts/paper2code_canary.py`; never read the credential files; kill = `kill -TERM` then
the `release` backstop; `rerun --phase environment_run` (not `run --until`) to repeat an accepted step 10; never edit a
running script; git identity is unset here — commit with `-c user.name="apple" -c user.email="apple@appledeMacBook-Pro.local"`;
another Claude session may commit on this worktree — `git log` before assuming.

## 0a. Overnight 2026-09-18: S1–S6 landed, S7 first GPU attempt, S8 offline (the record)

| step | result |
| --- | --- |
| S1 `ba3e25ab7` | repair loop: `repo_modules()` over the whole tree, `finish(summary, attribution)`, repeated-signature stop, requirements rule gone |
| S2 `bbe272fe9` | goal = basic template; asset card = policy (fail + release, no question); SetupX addendum = machine facts; `run.json.experiment_model` / `context_window`; every model default `DeepSeek-V4-Flash-Vision-Exp` |
| S3 `0bb500498` | implement runs upstream's local sandbox; `implementation_status` passes on all files written (`local_tests_failed` recorded) |
| S4 `8b26900e5` / validation `0f41ff1` | VENDOR 11: fan-out behind `DEEPCODE_PLANNING_FANOUT` (`run.json.planning_fanout`), planner budget behind `DEEPCODE_PLANNER_CONTEXT_WINDOW` (from `context_window`), appendix its own segment. sapg real paper: upstream 4 segments / 24k → all 10 / 59.5k |
| S5 `33d08fef9` | GPU image `m-j6c9bys3x83hji55jfha` via `scripts/paper2code_bake_gpu_image.py` (main's script imports a missing `apps.api`): T4, 5.5 min, ≈0.41 CNY; the CPU image already carried driver 595. **Rebaked 2026-09-18 01:19** with the tuna pip index (see the pitfall) — the new id is in `aliyun.env` `DEEPEVOL_API_ALIYUN_GPU_IMAGE_ID` |
| S6 `33d08fef9` | `GPU_CATALOG` (gn6i T4 / gn7i A10), `image_for(type)`, stock + live price at the review point, `gpu-*` tiers, `compute_tier` accepts `gpu-*` / `ecs.*`; sapg-2 re-decided `gpu-economy` → `ecs.gn6i-c8g1.2xlarge` 9.53 CNY/h |
| S7 attempt 1 | 00:54–01:20, `i-j6chclgvw9x140c54fif`, released. Machine ready in 45 s on the GPU image, RSA compiled and falsified in 2 calls, no asset card, SetupX started `pip install -r requirements.txt` — and torch (554 MB) crawled at 0.25 MB/s. **Cause found**: `mirrors.aliyun.com` serves HTTP/1.1 (pip) at 0.25 MB/s from cn-hongkong while HTTP/2 (curl) gets 14 MB/s; tuna and download.pytorch.org do 12 MB/s over HTTP/1.1. Stopped (would have blown the 3 h cap on ~3 GB of nvidia-* wheels), fixed in `25eb43ec8` (Dockerfile.setupx-base + facts + PITFALLS), image rebaked, relaunch follows |
| S7 = six attempts, `801b42b16` | all on T4 `ecs.gn6i-c8g1.2xlarge` (9.53 CNY/h), every machine released. ① pip 0.25 MB/s (aliyun mirror throttles HTTP/1.1) → tuna, image rebaked `m-j6c6925byfwdjv97nokt` (now in `aliyun.env`); ② artifact FileNotFound misclassified as a missing asset → classify judges the run's own failure first; ③ RSA's rollback silently lost the container → diagnosed, recovered by rebuilding the environment; the denylist audit scans commands only; ④ `run --until` skips an accepted phase → `rerun`; ⑤ the compiler declared the output directory an asset → dropped before G1, trial mkdirs it; ⑥ **03:11–03:53 full chain**: 搭建环境 10 min → trial → 修复 (commit `6fe80933`, re-pin, same container) → trial (run completes, artifacts absent) → 修复 spent 40 calls on 34 probes, wrote nothing → stop → `repair_review` accept → released. Probe budget `MAX_PROBES=15` added afterwards |
| S8 | offline `eef89b175` `c030a4f3d`; real machine = S7 ⑥ (the controller drove all four boxes on the T4) |
| S9 in flight | baseline `vexp1` (02:16, 33 files, 40 min) and `vexp2` (03:32, same + `DEEPCODE_IMPLEMENT_MAX_TOKENS=32768`, VENDOR 12) staged in `~/pb_submissions/sapg/` (old `deepevol_2` / `trial2` moved to `~/pb_submissions_archive/sapg/`); line `sapg-s9-off` (24 files) and `sapg-s9-on` (12 figures described, 25 files; its first implement aborted on a truncated 30 KB write_file at 8192 → VENDOR 12, rerun) both passed the four gates, compute → `gpu-economy`, environment_run launched 03:58 on two T4s. Next: `submit` pre_repair / post for both, `PAPER=sapg bash run_grade.sh` (Flash judge, V4-Pro parser), scores into §4 |
| port fix `b5c9c893d` | `LeasedExecutionPort.configure` refuses only an *active* lease (a released one blocked the compute rerun) |

## 0b. State on 2026-09-17 late evening (§1 below is the earlier record)

| item | state |
| --- | --- |
| branch | **`0916onmain-experiment`** on `HuigenYe/DeepEvol` (worktree `search/DeepEvol-Paper_repro_0916/DeepEvol1.0`), merged with `origin/main` tonight (`2df0920e5`; main's nine new commits touch the rsa child env allowlist, host keys and the SSH circuit — none of them the vendored rsa/setupx, all neutral to this line). `Paper_repro_0916` is the pre-rebase branch (recipe route; its PLAN-3 is stale) |
| venv | `.venv` (uv, CPython 3.12); `uv sync` removes the docker SDK — reinstall with `uv pip install docker` (SetupX needs it; `missing_modules()` refuses to rent without it) |
| tests | `tests/v2_paper2code` 141 offline + 6 Docker-gated, ruff clean; `tests/v2_experiment/test_rsa_child_process.py` green after the merge |
| models | owner: **everything on `DeepSeek-V4-Flash-Vision-Exp` for now** — `run.json.model`, `figures_model`, and the loopback's `experiment_model` (the last one is S2). `DeepSeek-V4.1-Flash` is in Paratera's catalogue but both keys get `403 team_model_access_denied`; apply for it, then switch the three defaults. Judge `DeepSeek-V4-Flash`, structured parser `DeepSeek-V4-Pro` |
| keys | `~/Documents/env/paratera_backup.env` (93 models), `aliyun.env` (ECS + image id); only via `--env-file`, never printed |
| runs | `~/Documents/search/paper2code-runs/sapg-2`: phases 1–9 completed; product unchanged since it was scored (27 files, `submission.json` manifest matches); `environment_run` had seven real-machine attempts (§4), none through G2; every lease released (`release` backstop after each kill); `code.git` holds round 0 |
| landed tonight | intake figure pass (`figures.py`: four-colour vision probe, one call per figure, description inserted after the reference, benchmark bytes kept in `input/paper.raw.md`; verified live on Paratera); compute re-decides on a released lease; PLAN-3 §0 rows for every review decision |
| tonight's review | phases 1–10 reviewed with the owner. Kept upstream-aligned: references, acquire, index, plan retry, plan review. Changed or planned: input is PaperBench `paper.md` (baseline logic — the scores so far were on it); figures described when the model takes images; implement runs the engine's local sandbox (S3); planner must see the whole paper within the context window (7b) and the fanout comes back (7); step 10 = mechanical controller, basic goal from the plan, SetupX as a black box, asset gate fails the run, repair agent edits code only; GPU tier pulled forward |
| next | **PLAN-3 §2.3 S1 → S9** in order: S1–S4 offline code (controller fixes, owner's step-10 inputs, implement local, engine patches 7/7b on both sides), S5 bake the GPU image (approved, ≤ 50 CNY), S6 GPU tier, S7 real-machine step 10 on sapg-2 (GPU), S8 controller reshaped to the owner's four-box diagram, S9 the paired rerun (baseline ×1, line figures-off ×1 and figures-on ×1, judged by V4-Flash) |

Rules that stay: run only through `scripts/paper2code_canary.py`; never read the credential files; kill = `kill -TERM` then the
`release` backstop; never edit a running script; git identity is unset here — commit with
`-c user.name="apple" -c user.email="apple@appledeMacBook-Pro.local"`; another Claude session may commit on this worktree — `git log` before assuming.

## 1. Where things stand

| item | state |
| --- | --- |
| branch | `Paper_repro_0916` on `HuigenYe/DeepEvol`, rebased onto `origin/main` (`8bf892299`) on 2026-09-17: 32 linear commits, no merge commits, ready for a PR. Worktree `search/DeepEvol-Paper_repro_0916/DeepEvol1.0`, venv `.venv` (uv, CPython 3.12) |
| first batch (PLAN.md C0–C10) | done: engine vendored (`apps/v2/agent_engine/paper2code`, VENDOR.md entries 1–10), the line (`apps/v2/agent/paper2code`), CLI `scripts/paper2code_canary.py`, 90-odd tests in `tests/v2_paper2code` |
| second batch (PLAN-2 D0–D5) | done: defaults 40/12, entry smoke, validation repository rebuilt on DeepCode main `21ebc57f`, judge vendored + `submit`, PITFALLS, environment_run re-run on Aliyun |
| third batch (PLAN-3) | items 1–3 done; items 4–5 (step 10 on main's experiment agent + repair rounds, `adr/0002`) **paused on 2026-09-17 by the owner's call**: code complete and offline-green (140 line tests), real-machine run not yet through — main's environment agent is freshly merged and not stable; six temporary seam adjustments recorded in PLAN-3 §0. Resume when main stabilises, revisit the adjustments against main's new version, then the full sapg-2 run (round 0 + repair rounds). Items 6–9 unchanged |
| scores | one scored pair on sapg under the caliber, Flash judge: baseline `trial2` 0.3374 vs line `deepevol_2` 0.3180 (77 Code-Dev leaves, gap inside run-to-run noise). No further sapg runs until the later phases exist (the user's call) |
| machines | no Aliyun instance left running; every lease in `lease.json` shows `released_at` (checked after the pause: sapg, sapg-2, ema-glm, sapg-glm, the bootstrap check) |
| keys | `~/Documents/env/paratera.env` is valid again (93 models, same key as `paratera_backup.env`); `aliyun.env` for the machine account. Never read, print or commit either |

## 2. How to run (every slot DeepSeek-V4-Flash-Vision-Exp, thinking off, 32768 per call; step 10 on a T4)

```bash
cd ~/Documents/search/DeepEvol-Paper_repro_0916/DeepEvol1.0
PY=.venv/bin/python; R=~/Documents/search/paper2code-runs; E=~/Documents/env
$PY scripts/paper2code_canary.py init --run-dir $R/<name> --paper-dir ~/Documents/search/paperbench/<paper> \
  --compute aliyun --run-hours 3 --repair-rounds 3 --figures off|on --planning-fanout    # add --ask for the review points;
  # the paper's assets must be real bytes (sapg's were LFS pointers — hydrated from HF on 2026-09-18); the compute
  # phase picks the GPU tier by itself when the code needs a GPU and ALIYUN_GPU_IMAGE_ID is set
nohup $PY scripts/paper2code_canary.py run --run-dir $R/<name> --until environment_run \
  --env-file $E/paratera.env --env-file $E/aliyun.env > $R/<name>/console.log 2>&1 &
$PY scripts/paper2code_canary.py status  --run-dir $R/<name>
$PY scripts/paper2code_canary.py release --run-dir $R/<name> --env-file $E/aliyun.env          # backstop: proves nothing is left rented
$PY scripts/paper2code_canary.py submit  --run-dir $R/<name> --paper <paper> --trial <label>   # into ~/pb_submissions/<paper>/<label>/ (repaired tree; --snapshot pre_repair for the comparison caliber)
$PY scripts/paper2code_canary.py rerun   --run-dir $R/<name> --phase environment_run --env-file … # repeat step 10 (an accepted step is 'completed'; run --until skips it)
```

Baseline pair (validation repo): `DEEPCODE_EXPECT_MODEL=DeepSeek-V4-Flash-Vision-Exp DEEPCODE_PLANNING_FANOUT=1 DEEPCODE_PLANNER_CONTEXT_WINDOW=1000000 PAPER=sapg TRIAL=<t> ENV_FILE=~/Documents/env/paratera.env bash paperbench/scripts/run_trial.sh`; judge: archive graded copies out of `~/pb_submissions/<paper>/` first, then `PATH=$HOME/Documents/search/.tools/bootstrap/bin:$PATH PAPER=sapg PB_JUDGE_MODEL=DeepSeek-V4-Flash bash paperbench/scripts/run_grade.sh` (Docker must be up).

Review points with `--ask`: `phases/04_plan_review.request.json` → write `04_plan_review.decision.json`;
`phases/10_environment_run.request.json` (RSA's clarification / asset / approval / criterion_review / escalation, machine held meanwhile) → write `10_environment_run.decision.json` (`{"kind": "clarification", "action": "answer", "message": "…"}`);
`phases/09_compute.request.json` → write `09_compute.decision.json` (`{"action": "approve", "answers": ["economy"]}`).
Both requests are in DeepEvol's `ask_user` question shape. Without `--ask` both auto-approve (plan as is, the
cheapest CPU tier) and record `auto`.

Baseline run (original DeepCode, same caliber), in the validation repository
`~/Documents/env/paperbench-judge/validation` (origin `2UBBISH/deepcode-paperbench-validation`, master):

```bash
PAPERS=sapg bash setup.sh                                                                # once; clone-and-run verified
PREFLIGHT_ONLY=1 PAPER=sapg ENV_FILE=~/Documents/env/paratera.env bash paperbench/scripts/run_trial.sh
PAPER=sapg TRIAL=trial3 ENV_FILE=~/Documents/env/paratera.env nohup bash paperbench/scripts/run_trial.sh > runs/sapg/console_trial3.log 2>&1 &
PAPER=sapg DRY=1 PB_JUDGE_MODEL=DeepSeek-V4-Flash bash paperbench/scripts/run_grade.sh   # then without DRY=1 (≈ ¥5–10 per submission)
```

Rules that cost money or runs when broken: never edit a running bash script; never rent through anything but the
line's lease; check `release` after every run; the judge's structured parser stays `DeepSeek-V4-Pro` (Flash corrupts
`response_format` output); `paper_split` must list the paper (`sapg` is in the patch; a new paper needs a Literal entry + split file).

## 3. What exists, and where

| thing | where |
| --- | --- |
| the line | `apps/v2/agent/paper2code/`: `config.py` (run.json, env defaults, seams), `provider.py`, `llm_loopback.py`, `runner.py`, `agent.py`, `tools/`, `execution/` (port, docker executor, Aliyun lease + `PreparedLease`, machine bootstrap, leased runtime), `intake.py`, `gates.py`, `phases.py`, `plan_review.py`, `environment_spec.py`, `compute.py`, `entry_smoke.py`, `code_repo.py`, `experiment_step.py`, `submit.py`, `driver.py`, `footprint.yaml`; the venv needs main's `agent-runtime` extra (the docker SDK) for step 10 |
| the engine | `apps/v2/agent_engine/paper2code/` (DeepCode business layer at `21ebc57f`, VENDOR.md lists every deviation; nothing else may change) |
| the judge | `vendor/paperbench-judge/` (PaperBench patch + `run_grade.sh` + `judge.sh`, copied from the validation repository at the commit in its VENDOR.md) |
| plans and records | `PLAN.md` (first batch + §10 deviations), `PLAN-2.md`, `PLAN-3.md`, `PITFALLS.md`, this file; root `CONTEXT.md` (glossary), `adr/0001-*` and `adr/0002-*` (line decisions) |
| validation repository | `~/Documents/env/paperbench-judge/validation`: DeepCode `21ebc57f` full copy + `patches/deepcode_local_changes.patch` (15 files; `verify_deepcode.sh` proves copy = upstream + patch), caliber config template, `run_trial.sh` / `run_grade.sh`, `docs/RESULTS-HISTORY.md` (every number since 2026-08-25 with void marks), `docs/PITFALLS.md`, `docs/INPUT_STANDARD.md` |
| archive (local, not tracked) | `~/Documents/env/paperbench-judge/archive/`: old submissions, grades, logs, JudgeEval outputs (Pro ×2, Flash ×1), the old DeepCode `e0767d0` copy, sapg grade files |
| run directories | `~/Documents/search/paper2code-runs/`: `sapg` (C9 acceptance + D5 reruns), `sapg-2` (second run, scored; its `environment_run` re-run on the experiment agent — `environment.json`, `rsa/` report + SetupX logs, `llm/rsa/` calls, `code.git`), `ema-glm` and `sapg-glm` (plumbing on GLM, never to be scored), `inputs/ema-detect` |
| submissions pool | `~/pb_submissions/sapg/{deepevol_2, trial2}` (graded); `~/pb_submissions_archive/sapg/trial1_maxtok8192` (ungraded) |

## 4. Results so far

| run | side | model | outcome | key numbers |
| --- | --- | --- | --- | --- |
| `ema-glm` | line | GLM-4.5-Flash | smoke passed | 145 calls; compileall job on Aliyun; plumbing only |
| `sapg-glm` | line | GLM-4.5-Flash | phases passed, job + release died on a network hiccup | rehearsal of segmentation, cloning, indexing |
| `sapg` (C9) | line | V4-Flash | all gates passed; ungraded | 346 calls, reasoning 0; 4 repos; 26/26 files; entry smoke failed (nested package) |
| `trial1` | baseline | V4-Flash | passed its gates; ungraded | 413 calls; per-call `max_tokens` clamped to 8192 by DeepCode's catalog |
| `sapg-2` | line | V4-Flash | all gates passed; **0.3180** | 544 calls; 5 repos; 26/26; requirements ok, compileall ok, entry smoke ok |
| `trial2` | baseline | V4-Flash | passed; **0.3374** | 702 calls; IsaacGymEnvs pre-filter overran and fell back to all 263 files |
| JudgeEval rice/0 | judge | V4-Flash as judge | acc 0.719 (= V4-Pro), F1 0.716, lenient 2.2 pp | 9.5 min; Pro parser kept |
| **S9 (2026-09-18)** `vexp2` | baseline | **Vision-Exp**, fan-out + whole-paper planning on, impl max_tokens 32768 | **0.6910** | 308 calls, 40 min; SAPG 0.979 / setup 0.917 / Fig.2·5·7·8 = 0·0.25·1·1 |
| S9 `sapg-s9-off` pre_repair | line (figures off) | same | **0.6677** | 714 calls / 8.9M tokens for the whole run; 24 files; SAPG 0.910 / setup 0.846 / 0·0.25·1·1 |
| S9 `sapg-s9-off` post | line (figures off) | same | **0.6365** | after 1 environment + 3 repair rounds on a T4 (G2 not passed, accepted); SAPG 0.969 / setup 0.850 / 0·0·1·1 |
| S9 `sapg-s9-on` post | line (figures **on**: 12 figures described by Vision-Exp) | same | **0.6709** | 811 calls / 9.1M; 25 files; 3 repair rounds (G2 not passed, accepted); SAPG 0.896 / setup 0.630 / 0·0.5·1·1 |
| **T1 (09-18 10:28)** `vexp1` | baseline | same, but implement max_tokens at upstream's 8192 (no truncation in the log) | **0.7156** | 33 files; the same run as `vexp2` — the noise sample; SAPG 0.816 / setup 0.728 / 0·0.75·1·1 |
| T1 `sapg-s9-on` pre_repair | line (figures on) | same | **0.6414** | the snapshot before the 3 repair rounds; SAPG 0.951 / setup 0.647 / 0·0.25·1·1 |

Reading S9 + T1 (validation `docs/RESULTS-HISTORY.md` §1.2 has the leaf overlaps): all six sit at 0.64–0.72, twice the
09-17 pair — the planning patches (fan-out, whole paper, appendix) and Vision-Exp lifted both sides alike (Figure 7/8
subtrees from 0 to 1). **The noise scale is 0.025** (`vexp1` vs `vexp2`, nothing changed between them). Against it:
baseline vs line-before-repair 0.023; repair rounds −0.031 (off) and +0.030 (on); figures on vs off −0.026 (pre) and
+0.034 (post). None of these is a signal; the earlier reading that "repair rounds cost score because the agent rewrote
the entry files" is withdrawn. **Owner's call (2026-09-18 morning): the repaired tree stays the product**; the old GPU
image stays too. Consequence for the to-do: T2's acceptance is mechanical (G2, diff size, probes), and a verdict on
figures or on the line vs the baseline needs more papers and repeats (T5), not more sapg runs.


Reading: same engine, same caliber, same result; the pair does not rank the two. Both lose the Figure 2/5/7
subtrees, PQL/DexPBT, the off-policy equations (Eq. 3/6/8) and IsaacGymEnvs imports — engine behaviour
(plan fixed once, no execution), the same mechanism seen on fre/rice. The DeepCode paper reports no sapg number.

## 5. Decisions that shape the next work

- Engine stays upstream-aligned: no replacement of the coding loop (A3 dropped). Engine patches, both sides,
  with rationale comments: pre-filter paths-only + `length` retry + domain-neutral wording (VENDOR 10);
  next, restoring upstream's own planning fan-out (removed in `c9090c1a`, gated, default off).
- Repair rounds live in the line (step 10), never inside the engine loop; max 3, 0 for comparison runs.
- Compute: static resource analysis from main's `agent_engine/experiment`; no billing; no time estimates;
  GPU tiers shown, CPU rented; the review point precedes any rent.
- Review points (plan, compute) use the `ask_user` question shape; product integration (workflow run +
  ask_user card) is item 9; the baseline never gets one.
- Judge: DeepSeek-V4-Flash with the V4-Pro structured parser; scores from different judges never share a table.

## 6. Next

See §0.3 (the to-do as of 2026-09-18 morning). The earlier list (main's agent stabilising, item 6, item 7, item 8, more samples) is
superseded: items 4–8 landed overnight as S1–S9; what remains is the repair loop's quality, the criterion's shape, a second
paper, the V4.1-Flash switch, the port's retirement and item 9.

---

# Appendix — run records (chronological)

## sapg acceptance (`sapg`) — passed under the agreed caliber

Run `09161610d37a`, paper `paperbench/sapg` (PaperBench Markdown + addendum,
blacklist `https://github.com/jayeshs999/sapg`), model `DeepSeek-V4-Flash`
via `paratera_backup.env`, `--compute aliyun --compute-tier enough --run-hours 6`,
engine commit `c1ed298b`. Launched 2026-09-17 02:08 UTC as `rerun --phase plan`
(after the reasoning-content guard fix below) followed by `run --until
environment_run`; both exited 0 at 03:01 UTC.

| criterion (PLAN.md C9) | result |
| --- | --- |
| `status.json` completed through `environment_run` | yes (`optimize` pending by design) |
| plan generated, not coerced | `plan_source` gate: `source=generated`, mode `segmented`, 9,672 chars, 33 s |
| references → repositories | `reference.txt` 13,098 chars naming 4 GitHub repos; all 4 cloned under `code_base/` (`IsaacGymEnvs`, `legged_gym`, `pql`, `pytorch-a2c-ppo-acktr-gail`); the blacklisted `jayeshs999/sapg` never appears |
| indexes | 4 (`indexes/<repo>_index.json`), 28 min |
| implementation | 26/26 files written in 10 min (27 files on disk incl. summary); `implementation_status` gate passed as `unverified` / `no_tests_discovered`; ownership gate 27/27 |
| every `llm/*.json` has `reasoning_tokens == 0` | yes — 346 calls, 1,836,573 prompt / 334,254 completion tokens, 0 errors. 20,631 chars of `reasoning_content` arrived with `reasoning_tokens == 0` (logged as warnings, see the guard note) |
| at least one job on an Aliyun instance | job 0001 `environment_run:compileall`, exit 0, `aliyun:i-j6c7n67quclolg13iqrb@47.83.136.192`, 409 s. It ran on the base image `paper2code-base:py311-7b88c7e1`: the requirements image build failed (see below) and `compileall` needs no dependencies |
| `lease.json` has `released_at` | yes — `ecs.c7.xlarge`, 02:53:43 → 03:01:09 UTC; `release` backstop afterwards: "instance already gone" |
| no residual instance | confirmed by the backstop's DescribeInstances |

Wall clock 53 min for the model phases plus 8 min on the machine. Nothing
here is a score: judging is deferred (PLAN.md §8) and the original-DeepCode
baseline still has to be re-run under this caliber before any comparison.

What changed on the way (each also in the fixes list): the first attempt
died at `plan` because the provider raised `ThinkingNotDisabled` on a
non-empty `reasoning_content` that came with `reasoning_tokens == 0`
(the segmentation call), and the engine silently fell back to traditional
planning. The guard now aborts only on non-zero `reasoning_tokens`,
records `reasoning_content_chars` in `llm/<seq>.json`, and
`PAPER2CODE_STRICT_REASONING_CONTENT=1` restores the strict behaviour. The
run was killed and restarted from `plan`.

The requirements image (`torch` and friends from the generated
`requirements.txt`) was not built: the `docker run` carrying the six-minute
`pip install` was held open on one ssh session, the session dropped, the
daemon's ssh retry re-ran the same `docker run` and hit a container-name
conflict (`exit 125`, `jobs/0001/pip.log`); the executor fell back to the
base image as designed (`jobs/image.json: status failed`). The fix —
containers started detached and polled with short docker calls, unique
container names — landed after this run and is covered by the local-Docker
tests only; the next Aliyun run is its first real exercise.


## EMA-Detect smoke (`ema-glm`) — passed on the substitute model

Run `091616046dfe`, paper `inputs/ema-detect` (77 lines, no references,
empty blacklist), model `GLM-4.5-Flash`, `--compute aliyun --compute-tier
enough --run-hours 2`.

| criterion (PLAN.md C8) | result |
| --- | --- |
| `status.json` completed through `environment_run` | yes (`optimize` pending by design) |
| every `llm/*.json` has `reasoning_tokens == 0` | yes — 145 calls, 975,874 prompt / 58,254 completion tokens, 0 errors |
| at least one job on an Aliyun instance | job 0003 `environment_run:compileall`, exit 0, `aliyun:i-j6chhnpyvp32nsi07t2t@47.243.184.171`, 386 s (includes the requirements image build) |
| `lease.json` has `released_at` | yes — `ecs.c7.xlarge`, 18:10:28 → 18:30:50 UTC, 1.39 CNY/h |
| no residual instance in the console | `DescribeInstances` shows none named `p2c-*` |

Four gates passed: preflight, plan_source, implementation_status
(`unverified` / `no_tests_discovered`, 8/8 files — the generated repository
has no tests, so the compile-check job is the only mechanical run),
ownership (14 files under `generate_code/`).

What it took to get there (each fix is in the list below): three
`implement` attempts — the first wrote 9/9 files in 15 min and was refused
by the old status gate; the second, started on top of the first's files,
drifted for an hour (GLM kept rewriting files and inventing checker
scripts; killed by hand); the third, after `rerun --phase implement`
archived the tree, wrote 8/8 in 13 min. Two `environment_run` attempts
died before the job ran: an attribute error masking a 20-minute
`docker build` hang through the docker.sock tunnel, then two 5-minute
pull timeouts through the same tunnel; the third attempt, with docker
commands over SSH, built the base image in 13 s and finished.

Job directories 0001 and 0002 belong to the failed attempts (no
`result.json`). The machine was reused across the last two attempts
(`lease.json` stayed `running`), which is why `created_at` precedes the
successful attempt.


## sapg rehearsal (`sapg-glm`) — the C9 plumbing on the substitute model

Run `09161922d522`, `paperbench/sapg` (56,540 bytes with the addendum;
denylist `https://github.com/jayeshs999/sapg`), `GLM-4.5-Flash`,
`--compute aliyun --compute-tier enough --run-hours 4`, implement wall
clock capped at 3,600 s by env. Not the caliber — a rehearsal of what the
EMA smoke could not reach: segmented planning, reference mining with real
fetches, cloning under the denylist, indexing, indexed-mode implementation.

| PLAN.md C9 criterion | result |
| --- | --- |
| all phases through `environment_run` completed | yes (`optimize` pending by design); 176 model calls, 1,137,152 prompt / 114,813 completion tokens, 0 reasoning tokens, 0 errors |
| `initial_plan.txt` `source == generated` | yes — segmented mode (`document_segments/` built in-process), 10,115-character plan in 209 s |
| `code_base/` ≥ 1 repository | `sac`, `baselines` (IsaacGym / DexPBT / parallel-ql do not exist or are private; refused at once with `GIT_TERMINAL_PROMPT=0`) |
| `indexes/` ≥ 1 index | `sac_index.json`, `baselines_index.json` (pre-filter chose 10 of 151 files; 767 s) |
| `generate_code/` ≥ 5 files | 31 files, 30/30 planned written in 2,893 s, `unverified` (no tests) |
| one job on an Aliyun instance, machine released, no residual | **no**: the compile-check job died on the SSH transport (`exit 255` from the daemon's five ssh attempts, then `CANARY_REMOTE_TUNNEL_FAILED` from the vendored sync's `ensure()`), and `DeleteInstance` failed on `[SSL: UNEXPECTED_EOF_WHILE_READING]` — the instance kept running and was deleted by hand 10 min later (`DescribeInstances` then empty). Both symptoms appeared in the same minute; this Mac's network hiccuped. |

The four fixes below make that failure mode survivable; they are covered
by tests but have not been through another paid run. Phase attempts:
`references` 2 (budget 8 → 40), `acquire` 5 (two runs cloned into the
repository root before the target fix, one fail-fast), `index` 3,
`implement` 3.


## Fixes made during C8 and the sapg rehearsal (folded into the C-commits they belong to)

- preflight denylist gate: an empty `blacklist.txt` is not a failure; the
  check is now "every blacklist entry is in the run's denylist" (C6).
- CLI reports a failed gate as JSON with exit 2 instead of a traceback (C7).
- `acquire` skips cloning, with a report, when the reference report names
  no GitHub repository at all (EMA-Detect has no references); the engine's
  fail-fast stays for the "agent narrated instead of cloning" case (C7).
- `llm/<seq>.json` numbering continues across processes (a rerun used to
  overwrite the earlier logs) (C4).
- `environment_run` runs one model-free `python -m compileall -q .` job on
  the run's machine: the engine only verifies when it discovers a test
  command, so a repository without tests never touched the execution port
  and the lease went unexercised (C7).
- ruff: the line's sources and tests lint clean (the vendored engine stays
  excluded).
- implementation_status gate: the engine reports `unverified` /
  `no_tests_discovered` when every planned file is written but no test
  command exists (EMA-Detect: 9/9 files, no tests — the same verdict the
  user's earlier DeepCode run got). The gate now passes that case and
  records `verified: false`; `test_failed` and early stops still fail (C6).

- Docker over SSH, not through the tunnel (C5, deviates from PLAN.md C5 item
  6): with the vendored docker.sock tunnel a `docker pull python:3.11-slim`
  from this Mac hung for the full timeout on a machine that pulls the same
  image in seven seconds when asked over SSH. `SshDockerHost` in
  `execution/leased_runtime.py` now runs every docker command on the machine
  through `RemoteDaemon.run` (stdin carries the Dockerfile); the tunnel code
  stays vendored but is never started.
- Base image acquisition (C5): `docker pull` first (5-minute budget), then
  the `docker.1ms.run` / `daocloud` / `dockerproxy` mirrors, all logged to
  `jobs/image-build.log`; the build runs with `--pull=false`. A job that
  cannot start reports its real exception (an attribute error used to mask
  it) and the JobResult carries `aliyun:<instance>@<ip>` as the machine.

- Provider egress guard (`scripts/data_contracts/check_provider_egress.py`):
  V2 code may not dial a provider directly, and findings under `apps/v2`
  cannot be allowlisted. The line's interim direct Paratera transport is
  pinned there as `PAPER2CODE_INTERIM_PROVIDER_FINDINGS` (two symbols of
  `provider.py`), justified inline and pinned by
  `tests/data_contracts/test_provider_egress_guard.py`; both are registered
  in `footprint.yaml`. When `GatewayProvider` lands the set goes away and
  the guard fails on any leftover direct call (C4).

- `references` fails loudly on a degenerate report (C7): the engine writes
  the analyzer's last message to `reference.txt` even when it is the
  runner's "I reached the maximum number of tool call iterations" text, and
  `acquire` then reads that as "no repositories". On sapg the analyzer
  re-read the 56k-character paper six times and exhausted the plan's default
  budget of 8 iterations (PLAN.md §6). The phase now names that outcome and
  points at `DEEPCODE_REFERENCE_MAX_ITERATIONS`; the sapg rehearsal reran
  it with 40 (the validation repo used 40–80 for the same reason).

- `fetch` on a developer machine with a fake-IP proxy DNS (C3): this Mac
  resolves github.com to 198.18.0.61 (RFC 2544 range, Clash/Surge fake-IP
  mode) and the engine's anti-rebinding resolver refused it, so the
  reference analyzer could not open any GitHub page. When resolution lands
  in 198.18.0.0/15 the tool now falls back to the system resolver (URL
  validation and the denylist unchanged; real private addresses are still
  refused). Cloud machines are not affected.
- `acquire` recognises `Repository: owner/repo` shorthand in the reference
  report as a repository to clone, not only full GitHub URLs (C7).

- `git_clone` targets are forced under the task's `code_base/` (C3): the
  engine's tool resolves an empty or relative `target_path` against the
  process cwd, and on the sapg rehearsal the download agent passed `""` —
  two repositories were cloned into the repository root (removed by hand).
  The wrapper now takes only the last path component the model gave, or the
  name inferred from the URL, and always places it in `code_base/`;
  `GIT_TERMINAL_PROMPT=0` is set so a missing or private repository fails
  at once instead of waiting for a username.

- Release must not lie (C5): a failed `DeleteInstance` is now recorded as
  `release_failed` (the machine is still yours), `EcsClient` retries
  transport errors four times for every action, and the `release`
  subcommand asks `DescribeInstances` first and deletes whatever it finds,
  whatever `lease.json` says. The daemon used on the machine is
  `SshOnlyDaemon` (no tunnel to open on a sync retry), and an ssh exit 255
  on a job is reported as a transport error, not as the job's exit code.

- Thinking guard on the token count only (C4): on the first C9 attempt
  Paratera's DeepSeek-V4-Flash filled `reasoning_content` on one call while
  billing `reasoning_tokens=0`; the provider treated that as thinking and
  raised, and the engine silently fell back from segmented to
  full-document planning. The caliber is defined on the token count
  (PLAN.md §0), so the provider now aborts only when `reasoning_tokens != 0`,
  records `reasoning_content_chars` in `llm/<seq>.json` and warns;
  `PAPER2CODE_STRICT_REASONING_CONTENT=1` restores the abort.

- Containers run detached (C5): a `docker run` held open on one ssh
  session dies with the session — the C9 run lost a torch install that
  way, and the daemon's ssh retry re-ran the same `docker run` into a
  "container name already in use" (exit 125). Both the requirements build
  and every job now start with `docker run -d`, are polled with short
  `docker inspect` calls (10 s), have their logs collected with
  `docker logs`, and are removed explicitly; names carry a timestamp.


## Second batch (PLAN-2, 2026-09-17)

### D5 — `rerun --phase environment_run` on the C9 run `sapg` (run id 09161610d37a)

| attempt | machine | requirements image | compileall | entry smoke (`sapg/main.py`, named in the plan) | lease |
| --- | --- | --- | --- | --- | --- |
| 1 · 12:49–13:11 | `i-j6ccdl9upisjo2gwv3di` (ecs.c7.xlarge, HK) | **failed: timed out after 1200 s** — the PyPI `torch` wheel pulls the CUDA 13 stack (nccl 216 MB, triton 248 MB, cublas 423 MB, …) at 1–2 MB/s onto a machine without a GPU; `jobs/0002/pip.log` | exit 0 on the base image (1247 s incl. the pip wait) | `--help` and bare: exit 1, `No module named 'yaml'` (base image has no requirements) | released 05:11:26Z, nothing left in the account |
| 2 · 13:13–13:18 | `i-j6c3z176gl3r9xtabvxk` | **ok in 2 min** with `--extra-index-url https://download.pytorch.org/whl/cpu` (commit 7a583e13; `PAPER2CODE_TORCH_CPU_INDEX=0` turns it off, `PAPER2CODE_PIP_TIMEOUT_S` / `PAPER2CODE_PIP_EXTRA_ARGS` added); `paper2code-run-09161610d37a:2b452a86c6a9` | exit 0, 148 s | `--help` and bare: exit 1 in 23 s — `from sapg.aggregation import …` fails because the generated tree is nested (`sapg/main.py`, `sapg/train.py` **and** `sapg/sapg/{aggregation,ppo,…}.py`): `sapg.train` resolves to the outer package, which has no `aggregation`. The repository compiles but does not start; that is what the smoke is for | released 05:18:21Z, `release` found nothing |

The detached containers (C5 fix) held through both attempts; the ssh "Permission denied" on the first
connect is the machine still authorising the key (retried and fine). Both attempts are `jobs/0001–0007`;
`phases/10_environment_run.json` is attempt 2 (`requirements_installed: true`, `entry_smoke.status: failed`).


### D2 — baseline run of sapg (validation repo, `PAPER=sapg TRIAL=trial1`, DeepCode 21ebc57f + patch, V4-Flash, thinking off)

12:36–13:24 on this Mac, task `paper_83d09fcd`, console `runs/sapg/console_trial1.log`, submission at
`~/pb_submissions/sapg/trial1/` (30 files). Same input bytes as the C9 run (`paper.md` + addendum, sha256
`04790c3f…`), same denylist, same model and thinking state. **No score** (PLAN-2: self-test only).

| | line, C9 run `09161610d37a` (2026-09-17 02:xx–03:01 + D5) | baseline run, `sapg/trial1` (DeepCode 21ebc57f + patch) |
| --- | --- | --- |
| plan | 9,672 chars, segmented (9 segments), `generated`, 33 s | 9,733 chars, segmented, `generated`, 17 s (attempt 1/3) |
| reference report | 13,098 chars, 4 GitHub URLs, 85 s | 14,707 chars, 5 GitHub URLs, 85 s |
| repositories cloned | 4: IsaacGymEnvs, legged_gym, pql, pytorch-a2c-ppo-acktr-gail (79 s) | 5: the same four + pytorch_sac (73 s) |
| indexing | 4 indexes, 1,711 s | 5 indexes, 2,103 s |
| implementation | 26/26 planned files, 599 s, engine verdict `unverified` (no test command discovered) | 29/29 planned files, 518 s, `verification: []` (upstream discovered no test command; nothing executed) |
| generated code | 22 py / 6,349 lines / 27 files | 24 py / 7,319 lines / 30 files |
| model calls | 346; prompt 1,836,573, completion 334,254, **reasoning 0** | 413; prompt 1,726,481, completion 419,385, **reasoning 0**; no `finish_reason=length` |
| per-call `max_tokens` | 32768 | **8192** (catalog clamp, see below) |
| environment | Aliyun ecs.c7.xlarge: requirements image ok (CPU torch), `compileall` exit 0, entry smoke failed (nested package, see D5) | none: the baseline runs on this Mac and only executes what it discovers (nothing here) |
| wall clock, intake → implement | 33 + 85 + 79 + 1,711 + 599 ≈ 2,507 s (+ 267 s environment_run) | 2,825 s (segmentation 17 s + plan 17 s + references 85 s + acquire 73 s + index 2,103 s + implement 518 s) |
| gates | preflight, plan_source, implementation_status, ownership: passed | caliber gate, plan-source gate, status + ownership gate: passed (run by hand, see below) |

Two things to know about this trial:

- **`max_tokens` asymmetry.** DeepCode 21ebc57f resolves the per-call limit through its model catalog; with the
  model listed as a bare name the deepseek family default is 8192, so this trial ran at 8192 while the line uses
  32768. No response was cut (`finish_reason=length` never occurred), so the numbers above stand, but the next
  baseline trial runs at 32768: the template now declares `manualModels: [{id, contextWindow, maxOutputTokens: 32768}]`
  and the preflight gate checks it (validation repo f9aa253).
- **The post-run gates were run by hand.** I edited `run_trial.sh` (the preflight check above) while this trial's
  bash process was still executing it; bash reads a script incrementally, so when the driver returned the shell
  hit a parse error at the tail and the gates + submission step did not run. I extracted that tail unchanged
  (from `STATUS=$(cat "$STATUS_FILE")` to the end) into a scratch script and ran it with the same variables at
  13:24:48; it passed every gate and placed the submission. Lesson already in `PITFALLS.md` §F ("运行中的脚本不改");
  it happened anyway.

Reading of the pair (no judging): same plan size and mode, same four repositories plus one the baseline found
in addition, ~10 % more generated code on the baseline side, 20 % more model calls, comparable wall clock;
the line's only extra is the environment step, which is where it learned that its own output does not start.


### Second pair, both scored (2026-09-17 13:41–15:22)

Judge switched to DeepSeek-V4-Flash after the JudgeEval trial (same accuracy as V4-Pro on rice/0, lenient
instead of strict; the structured parser stays V4-Pro because Flash corrupts `response_format` output).
Caliber on both sides: V4-Flash, thinking off (reasoning 0 throughout), `max_tokens` 32768, same input bytes.

| | line `sapg-2` (run 09170541607c) | baseline `trial2` (DeepCode 21ebc57f + patch) |
| --- | --- | --- |
| plan | 11,010 chars, segmented, 29 s | 9,687 chars, segmented, 15 s |
| references / clones | 5 URLs → 5 repos (161 s / 73 s) | 5 URLs → 4 repos (65 s / 108 s) |
| index | 5 repos, 3,409 s; every pre-filter parsed (263→101, 151→57, 23→19, 42→29, 12→11) | 4 repos, 5,057 s; **IsaacGymEnvs pre-filter JSON overran 32k and fell back to all 263 files** (the one `finish_reason=length` of the run) |
| implement | 26/26, 590 s | 24/24, 469 s |
| generated | 21 py / 6,279 lines / 27 files | 19 py / 5,143 lines / 24 files |
| model calls | 544; prompt 2,290,202, completion 536,512, reasoning 0 | 702; prompt 2,066,472, completion 708,860, reasoning 0 |
| environment | requirements ok (CPU torch), compileall ok, **entry smoke ok** (`python main.py --help` prints the argument table) | not executed (upstream discovered no test command) |
| wall clock | 76 min incl. 5 min environment | 96 min |
| **score (Flash judge, 77 Code-Dev leaves)** | **0.3180** — SAPG implemented 0.382, experimental setup 0.526, Fig 2/5/7 0, Fig 8 1.0 | **0.3374** — 0.413, 0.611, 0 / 0 / 0 / 1.0 |

Leaf level: 22 leaves pass on both sides, 9 only on the line, 11 only on the baseline. The gap (0.019) is far
inside the run-to-run spread seen on every earlier paper (0.09–0.19), so this pair says "same engine, same
caliber, same result", which is what the embedding was meant to show; it does not rank the two. Both lose the
whole Figure 2/5/7 subtrees (no experiment scripts for the plots), "five seeds", "six policies" and the 0.001
entropy setting; the line also loses the entropy choice and half of the hard tasks.
Grading took 4 minutes for both; judge tokens ≈ 3.1–3.2 M in / 0.28–0.30 M out per submission (Flash) plus
≈ 0.1 M in for the V4-Pro parser. Grade files: `~/Documents/env/paperbench-judge/archive/deepcode_test/sapg/grades/`.


### PLAN-3 item 2 — environment spec, first real extractions (2026-09-17)

One V4-Flash call per blueprint (`environment_spec.py`, prompt = the blueprint's `environment_setup` +
`implementation_components` + `validation_approach`, fixed JSON shape, "copy facts, never guess"):

| blueprint | tokens (in / out / reasoning) | python | packages | gpu | external tools | datasets | run commands |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sapg C9 | 1,800 / 640 / 0 | 3.8+ | torch, numpy, gym, gymnasium, isaacgym, mujoco 3.0, pyyaml, tensorboard | required ("IsaacGym requires NVIDIA GPU") | isaacgym (installable: true, needs driver + CUDA), mujoco, PhysX (unknown) | "autoencoder training data" (generated) | none stated |
| sapg-2 | 1,989 / 471 / 0 | 3.8+ | torch>=1.13, numpy, scipy, matplotlib, pyyaml | required (same reason; "CPU fallback for small-scale debugging") | IsaacGym, MuJoCo, PhysX | none | `pip install …`, "install IsaacGym per NVIDIA docs" |

Read: the spec says what the environment step must decide explicitly — `needs_gpu`, and IsaacGym as a tool
that pip cannot install (the model calls it installable "with driver + CUDA"; the recipe treats every
non-pip tool as a manual asset). CUDA version is null on both, as it should be: the blueprints never name one.

## Experiment agent on sapg-2 (PLAN-3 items 4–5, 2026-09-17, paused)

Six real-machine attempts of `environment_run` on the sapg-2 product, all on `ecs.c7.xlarge` in cn-hongkong from the
pre-baked image `m-j6c7r9v1zfzlibaknmmd` (bootstrap 6 s: docker 29.7.2, `setupx-base:py310-proxy`, `rsa-grader:py311-v1`
present). Each attempt cost minutes of machine time; every machine was released (the last two by the `release` backstop
after `kill -TERM`, because a nohup'd process ignores SIGINT).

| attempt | how far | what stopped it | fix |
| --- | --- | --- | --- |
| 1 | machine ready, code served | `ModuleNotFoundError: docker` (SetupX imports the docker SDK; main's `agent-runtime` extra) | SDK installed; `missing_modules()` preflight before renting |
| 2 | RSA compiled the criterion (1 loopback call, 3.9k in / 542 out, reasoning 0) | `criteria_G0.py` `SyntaxError` — the multi-line goal is rendered as one `# goal:` comment | goal is one line |
| 3 | compiled, frozen | falsifier: `command references 'runs/smoke_sapg', which does not exist` (`--output_dir` not in its output-flag list) | outputs under `/workspace/out` (absolute); one automatic recompile with the reasons |
| 4 | compiled, falsified OK; SetupX configured the environment (CPU torch 2.14 after 20 min of CUDA downloads) | SetupX then spent 70+ steps (116 calls) diagnosing code bugs it may not fix — `int(None)` in the config, `policy.sample` returns 4 values where `SAPG.act` unpacks 3 — and started patching a site-packages copy; killed at 51 min | machine notes appended to SetupX's prompt copy (CPU index, background installs, absolute outputs, denylist); round-0 budgets 60 steps × 3 rounds; the repair loop for the code bugs |
| 5 | (docs/plan commits between) | — | — |
| 6 | SetupX installed CPU torch directly (the notes worked), G0 checks at 12 min | **paused by the owner**: main's environment agent is not stable enough to iterate on | resume when main stabilises |

Every RSA / SetupX call went through the loopback (`llm/rsa/`, 148 calls across attempts, `reasoning_tokens == 0` on all).
The line's own executor (`job_executor.py`, `leased_runtime.py`) is still in place — retirement waits for the real-machine pass.
