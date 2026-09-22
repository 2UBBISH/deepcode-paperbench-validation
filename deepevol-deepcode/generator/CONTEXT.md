# Context

Domain vocabulary for DeepEvol. Terms only — no implementation detail, no decisions.
Decisions live in `docs/architecture/adr/`.

## Artifact lines（产物线）

**Artifact line**（产物线）— one kind of thing the system produces for a user on
request and keeps as theirs: paper, report, presentation, reproduction. Each line
may keep its own tables; all lines share one skeleton.

**Artifact skeleton**（产物骨架）— the seven roles every artifact line is built
from: subject, source reference, outline row, generation request, version,
version asset, idempotency receipt. The skeleton names roles, not tables — two
lines with the same shape may share a table, a line-specific concept gets its own.

**Subject**（主体）— the artifact itself as the user sees it: title, status,
workflow status. One per paper, per deck, per reproduction of a paper.

**Source reference**（来源引用）— what a subject was generated from: a research
session, an uploaded file, a published asset — with a content hash so a later
revision can tell whether the source moved. Distinct from the *source* senses
used in ARA compilation (see that section in the other worktree's glossary).

**Outline row**（大纲行）— one ordered unit of the subject's plan, kept as a row
with its own status: a chapter of a paper, a slide of a deck, a stage of a
reproduction. Never a blob inside the subject.

**Generation request**（生成请求）— one attempt to advance the subject through
one phase; exactly one per run. Carries the phase, the outcome, and the run it
was executed as. Failed requests stay: they are the record of what was tried.

**Artifact version**（产物版本）— an immutable snapshot of the subject created
only when a generation succeeded. Intermediate state is not a version; it lives
on the generation request.

**Version asset**（版本产物）— one file that belongs to a version, with a role
saying what the file *is* to the line. A version usually has several.

**Asset role**（产物角色）— the closed vocabulary a line uses to say what each
version asset is: the deliverable, a preview, the source, a figure, a project
file, the material snapshot, a carry. Closed means an unknown role is refused
rather than stored; each line owns its own vocabulary.

**Material snapshot**（输入物料包，作为角色）— the version asset holding the
copy of the material a version was generated from. Same meaning as the
existing *Material snapshot* term; as a role it makes "which inputs produced
this version" a fact on the version rather than on the subject.

**Project file**（工程文件）— a version asset that the source needs in order
to build but that is not the source itself: a paper's bibliography, class and
style files, its figures. Part of what a version *is*; a later revision
starts from the previous version's project files, unchanged.

**Carry**（搬运件）— intermediate state handed from one phase to the next so
the next phase can continue where this one stopped: a deck's spec lock.
Belongs to no version and is never shown to the user as a deliverable. A
paper's project files are not carry — they are *Project files* of a version.
Not *Material*, which names what a writer was given.

**Working copy**（工作副本）— the one mutable project state a subject has:
what the editor changes, what a compile renders, and what the next local
revision starts from. It is the authority on "the paper as it is now". Each
time a generation succeeds, the new version's files replace it wholesale;
edits made in between live only here and are not versions. Its shape follows
the line, and for both document lines it is a file set: a paper's is its
LaTeX project; a report's is `report.md` plus its figures. A report has no
chapter rows — its sections live inside the one markdown file.

**Idempotency receipt**（幂等回执）— the stored answer to an operation the
client may send again; a repeat gets the receipt, not a second effect.

### Two things called 阶段

**Chapter**（章节）— a content section of the subject. A paper's chapters are
unrelated to how it was generated. Since ADR 0034 neither document line keeps
chapter rows: a paper's sections live inside its LaTeX project, a report's
inside `report.md`; a confirmed outline is recorded on the document
(`metadata.document_outline.sections`) and replayed by the full draft, never
materialised as rows. The chapter tables remain for older documents and manual
edits only.

**Version**（版本）— a deliverable: one row per successful generation (or an
explicit rendered version), always carrying an artifact. Creating a document,
changing its metadata, pinning material, confirming an outline or editing a
chapter advances the document's CAS token, not its version list; a document
has no version until its first artifact lands (ADR 0034).

**Stage / phase**（阶段）— a step of the workflow that generates the subject:
outline, full draft, local revision for documents; plan, style preview, full
deck, revise pages for presentations. Recorded on generation requests, never on
chapters. Revising a deck by a free-text instruction is not a phase of its own:
it is a full-deck run given a new brief. *Revise pages* is the deck's local
revision — named pages redone, the rest kept, a new version minted.

In a reproduction the two coincide — environment, data, training, evaluation
are both the content sections and the workflow steps — so its stages are
outline rows and each advance is a generation request. That coincidence is a
property of reproduction, not of the words.

### Ownership vs execution

**Product-owned**（用户拥有）— anything that must go when the user goes: subjects,
outline rows, generation requests, versions, version assets, and the sessions
and files they came from. Lives on the product side.

**Execution record**（执行记录）— how a run was actually carried out: leases,
attempts, checkpoints, model invocations, diagnostic events. Lives on the agent
side, kept for a retention period, and never the home of anything a user owns.
A generation request and an execution record describe the same run from the
two sides and are joined by the run id, never by a foreign key.

### Engine and adapter

**Engine**（引擎）— the program that turns a brief and materials into a line's
content: a deck, a LaTeX project, a markdown report. One per line. An engine
knows nothing about runs, billing, versions or users; it reads files in a
workspace and writes files back. An engine is *live* when an adapter reaches
it, and dead otherwise — the V1 server reaching it once does not count.

**Adapter**（适配器）— the layer between a run and an engine: it translates the
run's phase into engine calls, injects the model client, and publishes what
the engine wrote as version assets under the line's asset roles. The phase
vocabulary and the asset-role vocabulary are the adapter's to keep closed;
the engine never sees either.

## Paper2Code line（论文转码线）

**Paper2Code line**（论文转码线）— an artifact line that turns a paper into a
code repository through the vendored DeepCode engine. It sits beside the
reproduction line and shares nothing with it but the artifact skeleton: its own
subject, its own phases, its own asset roles. Only one of the two is expected
to survive a scored comparison against upstream DeepCode.
_Avoid_: DeepCode line, 复现线（that is the reproduction line）, 论文复现（ambiguous between the two lines）

**Paper2Code phase**（转码阶段）— one of the eleven workflow steps below, each
run as one generation request. The engine's own glossary calls these 步骤;
in this repository they are phases. The engine's "阶段" (the DeepCode paper's
three phases) is never used here.

- **intake**（论文输入）— the paper directory as the benchmark hands it over
  becomes the engine's single Markdown input; the rubric never enters.
- **criteria**（评判标准）— a pass-through: the line records that a rubric
  exists but does not read it.
- **plan**（代码规划）— the engine's blueprint: the five-section YAML plan.
- **plan_review**（计划审阅）— the review point on the blueprint.
- **references**（参考挖掘）— repositories named by the paper's references.
- **acquire**（仓库获取）— those repositories cloned into the task directory.
- **index**（代码库索引）— the engine's code retrieval index over them.
- **implement**（代码实现）— the engine's tool loop writing the repository,
  then the mechanical verification.
- **compute**（算力）— the machine chosen from the compute spec at a review
  point; nothing is rented before that decision.
- **environment_run**（搭环境与运行）— the experiment agent proves the
  generated repository runs against a criterion, with the line's repair
  rounds in between; one phase, one loop. Today only the verification record.
- **optimize**（按标准优化）— iteration against the evaluation standard; a
  stub.

**Environment spec**（环境规格）— what a generated repository needs before
it can run, read out of the blueprint after planning: language version,
system and Python packages, datasets and their sizes, whether a GPU is
required. It informs the compute spec and the goal handed to the experiment
agent; nothing is built from it directly.
_Avoid_: environment_setup（the blueprint section it is read from）, requirements（one package list inside it）

**Compute spec**（算力规格）— what the generated code will demand of a
machine, read from the code itself with evidence: memory, storage, whether
and how much GPU. It ranks machine tiers for the compute review point; it
deliberately does not predict running time.
_Avoid_: 资源估算（the activity, not the result）, 机器配置

**Repair round**（修复轮）— one pass of the line's own loop inside
environment_run: a criterion the experiment agent could not satisfy is handed
to a repair agent, the code is changed, the same frozen criterion is judged
again in the same environment. Bounded by a count the run fixes in advance;
zero for a comparison run.
_Avoid_: 闭环修复（the mechanism, not one pass）, 自动修复

**Controller**（调度器）— the mechanical scheduler of environment_run: a
state and its transition rules, never a model call. From the input repository
on it decides which box runs next — building the environment, the trial, or
the code repair — from the last failure's signature or the repair agent's
attribution, and stops when the criterion passes, a budget is spent, or the
same failure comes back unchanged.
_Avoid_: 编排 agent（it is not an agent）, Router（RSA's own, which the
controller replaces for this line）

**Building the environment**（搭建环境）— the box that makes a container in
which the criterion can be judged: SetupX as a black box, given the
repository at a commit, the container so far, the failure evidence, the
machine facts and a budget; it returns what it did and the container it left.
_Avoid_: 配方（the abandoned deterministic recipe）, 装依赖（one of its causes）

**Trial**（远程初步执行）— the box that judges the frozen criterion on the
container as it is, rung by rung up to the minimal-scale run, and returns
verdicts with their failure text. It changes nothing.
_Avoid_: 验证（overloads the criterion）, 跑实验（the full-scale run it is not）

**Code repair**（修复代码）— the box that changes the generated code only:
the repair agent reads the evidence and the files, edits, probes in the
container, and ends with an attribution — code, or environment when the
cause is not in the code. Never installs into the judged environment.
_Avoid_: 修环境（the other box）, 改判据（forbidden）

**Experiment agent**（实验 Agent）— DeepEvol's own component that takes a
repository and a one-sentence goal, configures an environment for it on a
rented machine and proves, by a criterion, that it runs. It promises that the
code runs, never that the paper's numbers come out. This line's
environment_run phase is one call to it; the line adds only the repair rounds.
_Avoid_: RSA, SetupX（two of its parts, not the whole）, 配环境 agent（one part）

**Criterion**（判据）— the experiment agent's definition of "it runs" for one
repository and goal, written down and frozen before anything is configured,
then judged mechanically: a ladder of rungs from "imports and the entry
parses" through "a minimal-scale run leaves its artifacts" up to a full-scale
run. Rungs beyond the minimal-scale run start only on a person's approval.
_Avoid_: 测试（the criterion is rendered as tests, but a repository's own
tests are not it）, 验证标准（overloads the evaluation standard）

**Evidence**（证据）— what an environment_run leaves behind so that "it ran"
can be checked: the frozen criterion, every command the experiment agent
issued with its exit code, each judgement, the output tails. Without it a
pass is a claim.
_Avoid_: 日志（evidence is selected and structured; a log is neither）,
报告（the rendering of evidence, not the evidence）

**Review point**（审阅点）— a place where a run stops and waits for a
person's decision, which comes back as a file; the run resumes from there.
Distinct from a gate, which the line decides by itself. Every review point
has a non-blocking default so an unattended run never waits: usually
"approve", but "do not start" for anything that would spend hours or weaken
a criterion. The blueprint (plan_review), the machine (compute) and the
experiment agent's questions (environment_run) have one.
_Avoid_: human gate（overloads 闸门）, 交互（too broad）, 确认（the decision, not the place）

**Denylist**（黑名单）— the paper's own `blacklist.txt`: repositories and URLs
the engine must not fetch or clone during a run. Enforced by the line's tools,
not by the model.
_Avoid_: blacklist（the file's name, not the concept）, 封锁

**Run directory**（运行目录）— where one Paper2Code run keeps everything it
produced: the frozen configuration, the phase records, the engine's task
directory, every model call, every remote job, the machine lease. The
authority on a run until the line has Product tables.
_Avoid_: task dir（that is the engine's directory inside it）, workspace

**Caliber**（口径）— the conditions under which a Paper2Code run's output
may be compared with another's: the model, whether its thinking is on or
off, and the endpoint that served it. A run records its caliber in
`run.json`; a score is only ever read against a baseline of the same
caliber. Today's caliber is DeepSeek-V4-Flash with thinking off.
_Avoid_: 配置（the caliber is a comparison rule, not the run's settings）, model（only one of its three parts）

**Machine lease**（机器租期）— one Aliyun instance a Paper2Code run holds
exclusively from creation to deletion; every remote job of the run executes
on it. A lease that the run could not delete is recorded as such, never as
released; the `release` command is the backstop that asks the account.
_Avoid_: 服务器, VM（the line rents and returns; it never keeps a machine）

**Comparison method**（对比方法）— an algorithm a paper measures its own
method against; the rubric scores whether a reproduction implements the
comparison methods and runs the comparison, separately from the main
method. The word "baseline" is not used for this.
_Avoid_: 基线, baseline（ambiguous with the baseline run below）

**Baseline run**（基线运行）— one run of the original DeepCode engine on a
paper under a given caliber, kept only so a Paper2Code run's score has
something to be read against. It carries none of the line's prompt
changes; a baseline that knows the rubric measures the rubric, not the
engine.
_Avoid_: 基线（bare; say which of the two), 对照组

**Source pointer**（Source 指针）— a `Source: §x.y` line in a blueprint component's paragraph: the paper section
the component comes from, named by number or heading. It locates, it does not quote; the paper is read where it points.
_Avoid_: 引用 / quote（a pointer carries no text）, source obligation（retired: no unit smaller than a section）

**Reading obligation**（阅读义务）— what a planned file must do before it is written: read every section the
pointers of every paragraph naming that file resolve to, at least one page each, in an earlier model turn. A file no
paragraph points at (glue) has none; a pointer that names no heading creates none.
_Avoid_: binding（the obligation is the blueprint read back, not a separate artifact）

**Read receipt**（回读回执）— the host's record that one `read_paper` page was returned for one file: section,
page, bytes, turn. The write of a file cites the receipts that authorised it.
_Avoid_: paper read count（too coarse; say which file and section）

**Fidelity audit**（保真审计）— the replay of receipts against the manifest and the final bytes at the end of the
implementation: which writes cited earlier-turn reads of the right sections, which planned files are missing or
unplanned files present. A record of findings, never a verdict on the phase.
_Avoid_: gate（retired: nothing fails on the audit）, quality score, judge
