# 论文复现 Agent 架构 v0.3：MVP 实施与验证计划

> 日期：2026-09-03。
> 本文细化 `ARCHITECTURE_v0.2_OPTIMAL.md` 的实施顺序，并修正其评审中发现的三个阻塞问题：
> 科学对照与 Agent 进步混用、度量体参与优化、A/B 之间的产物合同不闭合。
> v0.2 继续保留为目标形态与证据汇总；**实际建造与验收以本文为准**。

---

## 0. MVP 的唯一目标

先证明一条最小纵切真实闭合：

```text
论文来源
  → 冻结 StudySpec
  → 冻结 CandidateBundle
  → 冻结 RunPlan（同一最终版本内，treatment vs comparator）
  → 隔离执行
  → evaluator 自己产生 ObservationSet
  → 确定性生成 Verdict
```

MVP 完成不等于论文主张已经复现。MVP 完成只表示：

1. 输入、候选代码、运行计划、原始观测和结论都有版本与内容哈希；
2. evaluator 能在不信任 candidate 自报分数的情况下运行两个实验臂；
3. `agent_progress` 与 `paper_replication` 是两套不同结论；
4. smoke / 降配运行只能得到 `directional` 或 `na`，不能冒充 `pass`；
5. 篡改文件、缺产物、环境伪装、执行失败都 fail-closed；
6. 最终 evaluator 不向写码循环回灌结果。

### 0.1 MVP 明确不做

- 不接 LLM 写码循环；
- 不做 B1 的 LLM 自动抽取；
- 不做 B2 的 LLM 自动测试生成；
- 不做 DeepCode A1 语料前端；
- 不做跨论文技能库；
- 不做远程租卡；
- 不做 PaperBench Code-Dev 优化；
- 不用 smoke 结果宣称论文主张 `pass`。

这些能力只有在纵切稳定后才逐层接入。

---

## 1. 先固定五份外部合同

文件数量不是目标。接口是否小，取决于调用者还需要知道多少隐含事实。MVP 用五份版本化合同把目录、checkpoint、镜像、运行臂和统计规则全部显式化。

四份 JSON envelope（study、candidate、run plan、verdict）都必须包含：

```json
{
  "schema_version": "0.3.0",
  "created_at": "RFC3339 UTC",
  "content_digest": "sha256:..."
}
```

每份 envelope 另有一个类型专属稳定 ID，例如 `study_id` 或 `candidate_id`。`content_digest` 的计算规则固定为：移除自身字段 → UTF-8 → JSON key 排序 → 紧凑序列化 → SHA-256。任何读取者先验 hash，再做语义校验。

`observations.jsonl` 的每条记录包含 `schema_version` 与 `observation_id`；整个文件按原始字节计算 set digest，并由 ledger 与 Verdict 引用，不给每行重复添加 content digest。

### 1.1 `study.json`：FrozenStudySpec

它表达“论文要求比较什么”，不表达 Agent 如何实现。

MVP 必需字段：

```json
{
  "study_id": "study_rice_v1",
  "paper": {
    "paper_id": "rice",
    "source_files": [
      {"path": "paper.md", "sha256": "..."},
      {"path": "addendum.md", "sha256": "..."}
    ]
  },
  "methods": [
    {"id": "rice", "role": "treatment"},
    {"id": "statemask", "role": "comparator"}
  ],
  "metrics": [
    {
      "id": "mask_training_wall_seconds",
      "unit": "seconds",
      "direction": "minimize",
      "evaluator_adapter": "wall_clock_v1"
    }
  ],
  "claims": [
    {
      "id": "rice_exp2_efficiency_hopper",
      "statement": "RICE trains the mask network faster than StateMask for a fixed sample count on Hopper.",
      "source_evidence": {
        "path": "paper.md",
        "start_line": 618,
        "end_line": 635,
        "quote_sha256": "..."
      },
      "comparison": {
        "treatment_method_id": "rice",
        "comparator_method_id": "statemask",
        "metric_id": "mask_training_wall_seconds",
        "estimand": "paired_median_ratio",
        "expected_direction": "lower",
        "null_value": 1.0,
        "practical_margin": 0.0
      }
    }
  ],
  "freeze": {
    "status": "approved",
    "approved_by": "human",
    "approved_at": "RFC3339 UTC"
  }
}
```

约束：

- 每条 claim 必须同时引用 treatment、comparator 和 metric；
- 每条 claim 必须有论文证据位置及证据内容哈希；
- `paper_effect_sigma` 只可作为描述性字段，不能作为可测性硬闸；
- metric 使用 evaluator 内已登记的 adapter，不允许 `parse` 或任意 shell 字符串；
- MVP 的 StudySpec 由人工填写并人工冻结；自动抽取以后只能产生 `draft`。

### 1.2 `candidate.json`：CandidateBundle

它表达“将被执行的候选到底是什么”。

```json
{
  "candidate_id": "candidate_rice_trial2_001",
  "study_digest": "sha256:...",
  "source": {
    "tree_sha256": "...",
    "archive": "candidate.tar",
    "archive_sha256": "..."
  },
  "runtime": {
    "image_digest": "sha256:...",
    "dependency_lock_sha256": "...",
    "python_version": "3.11"
  },
  "entrypoints": [
    {
      "id": "train_mask",
      "argv": ["python", "experiments/train_mask.py"],
      "working_directory": "/workspace/candidate"
    }
  ],
  "artifacts": [
    {
      "id": "pretrained_hopper_agent",
      "relative_path": "models/hopper_agent.zip",
      "sha256": "...",
      "format": "sb3_zip",
      "required": true
    }
  ]
}
```

约束：

- evaluator 只按 bundle 中的 digest 取代码、镜像和 checkpoint；
- 所有路径是 bundle 根目录下的相对路径，解析后不得逃逸根目录；
- 禁止 symlink 指向 bundle 外；
- `argv` 是字符串数组，不接受 shell command 字符串；
- checkpoint 只在隔离执行容器内加载，宿主进程不得反序列化；
- 任何必需 artifact 缺失都不得进入运行阶段；
- tar 按路径排序并归一化 uid/gid/mtime，保证相同代码树得到相同 tree digest。

### 1.3 `run_plan.json`：RunPlan

它表达“如何公平比较”。同一 claim 的两个实验臂必须来自同一个 CandidateBundle。下面是目标接口示意，不表示 trial2 当前已经支持这些参数；M2/M5 正是用来发现这种 candidate 合同缺口。

```json
{
  "run_plan_id": "runplan_rice_exp2_smoke_001",
  "study_digest": "sha256:...",
  "candidate_digest": "sha256:...",
  "claim_id": "rice_exp2_efficiency_hopper",
  "scale": {
    "name": "smoke",
    "paper_equivalent": false,
    "reason": "Pipeline validation only",
    "sample_count": 1000
  },
  "paired_seeds": [0],
  "arms": [
    {
      "id": "treatment",
      "method_id": "rice",
      "entrypoint_id": "train_mask",
      "args": ["--method", "rice", "--env_id", "Hopper-v4"]
    },
    {
      "id": "comparator",
      "method_id": "statemask",
      "entrypoint_id": "train_mask",
      "args": ["--method", "statemask", "--env_id", "Hopper-v4"]
    }
  ],
  "resources": {
    "needs_gpu": true,
    "timeout_seconds_per_run": 1800,
    "memory_mb": 8192,
    "network": "none"
  },
  "verdict_policy": "smoke_direction_only_v1"
}
```

约束：

- paper claim 只能比较 `arms[]`，不能比较 `_seed` 与 `_final`；
- 两个实验臂使用相同 seeds、scale、环境和资源上限；
- smoke 的 `paper_equivalent=false`，其 policy 永不产生 `pass`；
- full 计划另建文件，不得运行中修改 scale；
- 运行前同时冻结 study、candidate、run plan 三个 digest。

### 1.4 `observations.jsonl`：SealedObservationSet

这是 evaluator 的原始输出，不是 candidate 的自报结果。每行对应一个实验臂 × seed × metric。

```json
{
  "schema_version": "0.3.0",
  "observation_id": "obs_...",
  "run_plan_digest": "sha256:...",
  "arm_id": "treatment",
  "method_id": "rice",
  "seed": 0,
  "metric_id": "mask_training_wall_seconds",
  "value": 12.34,
  "unit": "seconds",
  "status": "ok",
  "runtime": {
    "started_at": "...",
    "finished_at": "...",
    "exit_code": 0,
    "image_digest": "sha256:...",
    "container_id": "..."
  },
  "environment_identity": {
    "distribution": "gymnasium",
    "version": "...",
    "module_origin": "/usr/local/lib/...",
    "origin_inside_candidate": false
  },
  "artifact_digests": []
}
```

约束：

- candidate 写出的 `metrics.json` 只能作为 diagnostic attachment；
- wall time、退出码、资源信息由 executor 记录；
- evaluator 在 seal 前检查行数、arm/seed 完整性和父 digest；
- seal 后任何一字节变化都使 ObservationSet 无效。

### 1.5 `verdict.json`：FinalVerdict

```json
{
  "verdict_id": "verdict_...",
  "study_digest": "sha256:...",
  "candidate_digest": "sha256:...",
  "run_plan_digest": "sha256:...",
  "observation_set_digest": "sha256:...",
  "pipeline_status": "completed",
  "claim_results": [
    {
      "claim_id": "rice_exp2_efficiency_hopper",
      "status": "directional",
      "reason_codes": ["SMOKE_SCALE_NOT_PAPER_EQUIVALENT"],
      "estimate": {"name": "paired_median_ratio", "value": 0.81},
      "evidence_observation_ids": ["obs_...", "obs_..."]
    }
  ]
}
```

状态语义：

- `pass`：full、预注册样本量满足、统计 policy 满足；
- `fail`：执行有效但效果未满足 policy；
- `directional`：执行有效，仅能报告方向；
- `na`：资源、scale、数据或论文信息不足，不能判；
- `error`：合同、执行或 evaluator 本身失效，本轮不得进入成功率分母。

科学 `fail` 不对应非零进程退出码；它表示评测成功完成但主张未复现。

---

## 2. 六个深模块及其测试面

| 模块 | 外部接口 | 隐藏在实现中的复杂性 | MVP adapter | 测试面 |
| --- | --- | --- | --- | --- |
| `StudySpec` | `load_and_freeze(source_dir, draft) -> study.json` | 来源白名单、证据哈希、引用闭合、冻结 | 人工 draft；暂不设 LLM port | 冻结后的 JSON 与错误码 |
| `CandidateBundle` | `pack(source_dir, study) -> candidate.json` | 规范路径、tree hash、archive、artifact 清点、防 symlink 逃逸 | 本地文件系统 | bundle 及篡改检测 |
| `RunExecutor` | `execute(EvaluationRequest) -> observations` | 容器、超时、环境身份、日志、artifact | `FixtureExecutor` + `DockerExecutor` | ObservationSet，不测试容器内部类 |
| `Evaluator` | `evaluate(study, run_plan, observations) -> verdict` | 完整性检查、estimand、status policy、理由码 | 无外部依赖 | Verdict；这是主要测试面 |
| `RunLedger` | `transition(run_id, event) -> state` | 事务、幂等、lease、恢复 | SQLite WAL | 状态与非法转换 |
| `DevHarness` | `check(candidate) -> dev_gate_result` | compile/import/help/smoke 的详细错误 | Docker adapter | gate 结果；不与 FinalVerdict 共用 |

依赖分类：

- 哈希、合同验证、统计 policy：进程内依赖，直接并入相应模块；
- 文件系统和 SQLite：本地可替换，测试使用临时目录与内存数据库；
- Docker：在 `RunExecutor` seam 放 `FixtureExecutor` 与 `DockerExecutor` 两个 adapter；
- Paratera：MVP 不引入。自动抽取阶段再定义模型 port 与 fake adapter；
- 远程 GPU：本地 Docker adapter 稳定后再增加，不能提前抽象一个只有单实现的租卡接口。

---

## 3. 两条完全分离的反馈路径

```text
开发路径：
Candidate draft → DevHarness → 可回灌 DevGateResult → Candidate draft

最终路径：
Frozen CandidateBundle → FinalEvaluator → FinalVerdict
                                   └─ 运行结束前不回灌
```

规则：

1. DevHarness 和 FinalEvaluator 使用不同目录、不同进程和不同 artifact root；
2. A3 将来只能调用 DevHarness；
3. FinalEvaluator 不提供“只跑某个隐藏断言”的接口；
4. 一次 final evaluation 创建后，candidate digest 不可变化；
5. 留出集的 FinalVerdict 不进入技能库；
6. 技能库只能消费开发集 DevGateResult 和开发集实验结果。

---

## 4. MVP 运行状态机

RunLedger 是唯一权威状态，不从日志文本或输出目录猜阶段。

```text
CREATED
  → CONTRACTS_VALIDATED
  → CANDIDATE_VERIFIED
  → EXECUTION_LEASED
  → EXECUTING
  → OBSERVATIONS_STAGED
  → OBSERVATIONS_SEALED
  → EVALUATED
  → PUBLISHED
```

终止状态：

```text
CONTRACT_INVALID
CANDIDATE_INVALID
EXECUTION_FAILED
SANDBOX_VIOLATION
EVALUATION_ERROR
CANCELLED
```

实现约束：

- SQLite WAL；每个 transition 在单事务中写入；
- `(run_id, event_id)` 唯一，重复提交幂等；
- executor 领取 lease 后记录 container id，再真正启动命令；
- 产物先写 staging，校验完成后原子 rename；
- `docker --rm` 只在 ledger 已记录终态后清理；
- recovery 只依据 ledger、container label 和已 seal artifact，不依据“目录看起来完整”。

---

## 5. 最终执行的最小沙箱合同

FinalEvaluator 启动 candidate 时至少使用：

```text
非 root 固定 UID/GID
只读 rootfs
candidate 代码只读挂载
单独可写 /outputs
tmpfs /tmp
--network none
--cap-drop ALL
--security-opt no-new-privileges
--pids-limit
CPU / memory / wall-clock 限额
不挂 Docker socket
不挂 LLM credential
不挂 harness 源码
不挂工作区根目录
```

运行身份检查：

- evaluator 在容器中通过 `importlib.metadata` 取得 distribution/version；
- 解析 `module.__file__` 后确认不在 candidate 挂载目录；
- 记录镜像 digest、Python 版本、distribution 版本和 module origin；
- candidate 自己写入的 `provenance.env_module` 不参与硬闸；
- SB3/PyTorch checkpoint 只在该隔离容器内加载。

资源获取是另一种运行模式，允许受控网络，但不能复用 FinalEvaluator 容器与凭据。

---

## 6. 分步实施：每一步都可在本步终止并验收

### M0：建包与合同锁定

**目标**：不执行论文代码，先让五份合同可创建、校验、哈希和互相引用。

位置：`/home/deepevol/deepevol/repro/`，Python 3.12，使用 `uv`，初版只依赖标准库。

最小目录：

```text
repro/
├── pyproject.toml
├── src/repro/
│   ├── cli.py
│   ├── contracts/
│   │   ├── canonical_json.py
│   │   ├── validation.py
│   │   └── reason_codes.py
│   ├── study.py
│   ├── candidate.py
│   ├── run_plan.py
│   └── verdict.py
├── schemas/
│   ├── study.schema.json
│   ├── candidate.schema.json
│   ├── run_plan.schema.json
│   ├── observation.schema.json
│   └── verdict.schema.json
└── tests/
    ├── fixtures/
    └── test_contracts.py
```

验收：

```bash
uv run python -m unittest tests.test_contracts
uv run repro validate tests/fixtures/valid/study.json
uv run repro validate tests/fixtures/invalid/claim_without_comparator.json
```

必须证明：

- 合法 fixture 返回 0；
- 缺 comparator 返回 10；
- 父 digest 不匹配返回 11；
- 修改冻结文件一个字符后返回 11；
- 绝对路径、`../`、逃逸 symlink 返回 12。

本步不做 Docker、GPU、统计和 LLM。

### M1：纯进程内纵切

**目标**：用 `FixtureExecutor` 证明 treatment/comparator → observations → verdict 的语义，不依赖外部环境。

fixture 固定返回：

```text
seed 0: treatment=80, comparator=100
seed 1: treatment=82, comparator=101
seed 2: treatment=79, comparator=99
```

验收：

```bash
uv run python -m unittest tests.test_evaluator tests.test_run_ledger
uv run repro evaluate \
  --study tests/fixtures/valid/study.json \
  --candidate tests/fixtures/valid/candidate.json \
  --run-plan tests/fixtures/valid/run_plan_full.json \
  --executor fixture \
  --output tmp/m1/
```

必须证明：

- evaluator 比较的是两个 arm，不是 candidate revision；
- smoke policy 只生成 `directional`；
- full fixture policy 可以生成 `pass`；
- 缺一个 seed 时生成 `error/INCOMPLETE_PAIRED_OBSERVATIONS`；
- candidate 自报一个伪造高分不改变 Verdict；
- 重跑同一个 run id 不会追加重复 observation。

本步完成后，核心 evaluator 已可通过其外部接口测试。

### M2：CandidateBundle 打包与防篡改

**目标**：验证可重复打包，并对 `rice/submissions/trial2` 形成结构化缺口报告；不运行训练。

输入：

```text
/home/deepevol/deepevol/deepcode_test/rice/submissions/trial2/
```

验收：

```bash
uv run repro inspect-candidate \
  --study examples/rice_hopper_efficiency/study.json \
  --source ../deepcode_test/rice/submissions/trial2 \
  --output runs/m2/candidate_gap.json

uv run repro pack-candidate \
  --study tests/fixtures/valid/study.json \
  --source tests/fixtures/valid/candidate_source \
  --output runs/m2/toy_candidate/

uv run repro verify-candidate runs/m2/toy_candidate/candidate.json
```

必须证明：

- 对一个完整 toy candidate 打包并通过防篡改测试；
- 同一代码树重复打包得到相同 tree digest；
- 修改任一 `.py` 后旧 bundle 校验失败；
- `inspect-candidate` 检查 trial2；缺少声明的 pretrained checkpoint 时生成 `candidate_gap.json`，随后对 trial2 执行 `pack-candidate` 明确返回 12；
- 不因 checkpoint 缺失伪造空 artifact 或继续执行；
- archive 展开后没有路径逃逸和外部 symlink。

M2 对 toy candidate 应全绿；对 trial2 的检查预期很可能以 `MISSING_REQUIRED_ARTIFACT` 结束。这是有效验收结果，不是管线失败，它会明确下一步需要真实获取或生成哪些 checkpoint。缺口消除前不得生成“看似有效”的 trial2 CandidateBundle。

### M3：容器 adapter 的确定性测试

**目标**：先用无 GPU 的 toy candidate 验证 Docker 隔离、ledger、超时和 observation seal。

toy candidate 只用 Python 标准库，两个 arm 各输出固定结构的数值结果；`numeric_output_v1` adapter 计算最终 metric，executor 计时只作运行 diagnostic。

验收：

```bash
uv run python -m unittest tests.integration.test_docker_executor
uv run repro evaluate \
  --study tests/fixtures/docker/study.json \
  --candidate tests/fixtures/docker/candidate.json \
  --run-plan tests/fixtures/docker/run_plan.json \
  --executor docker \
  --output runs/m3/
```

必须证明：

- candidate 不能读取 harness canary 文件；
- candidate 不能访问网络；
- 超时被杀死且 ledger 为 `EXECUTION_FAILED`；
- 重启 controller 后可依据 ledger 恢复；
- staged observations 篡改后不能 seal；
- candidate 进程成功但未产出必需 artifact 时不算成功。

### M4：rice 环境与入口 smoke

**目标**：第一次运行 trial2 源码，但只验证真实环境与入口，不判断论文主张。

范围：

- Hopper；
- 真实 `gymnasium`/MuJoCo；
- compile、import、`--help`、`make/reset/step`；
- 不要求 pretrained checkpoint，不训练 300k samples；
- 不调用 FinalEvaluator，不产生 Verdict；里程碑报告只记 `NO_SCIENTIFIC_RUN`。

验收：

```bash
uv run repro dev-check \
  --source ../deepcode_test/rice/submissions/trial2 \
  --profile rice_hopper \
  --output runs/m4/dev-gate.json
```

必须证明：

- `Hopper-v3 → Hopper-v4` 映射被记录为 scope deviation，不静默替换；
- distribution、version 和 module origin 来自镜像而非 candidate；
- clean container 中重复运行结果一致；
- DevGateResult 可以给未来 A3 使用，但不产生 FinalVerdict。

### M5：rice 单 claim 的 smoke 科学纵切

**目标**：运行 Experiment II 的两个实验臂，验证 treatment/comparator 的真实执行合同。

第一轮使用：

```text
claim: rice_exp2_efficiency_hopper
methods: RICE vs StateMask
metric: fixed-sample mask-training wall time
scale: smoke
sample_count: 1,000（若不足以走完初始化，可上调到最小可执行值）
paired_seeds: [0]
```

验收要求：

- 两个 arm 都完成相同 sample count；
- wall time 由 executor 测量；
- 输出 `observations.jsonl` 与 `verdict.json`；
- 无论方向如何，claim 只能是 `directional` 或 `na`；
- candidate 自报训练时间与 executor 时间不一致时，只记录 diagnostic；
- 费用、墙钟、镜像和环境身份完整入账。

如果 trial2 缺 pretrained checkpoint 或没有真正可执行的 StateMask 训练入口，M5 必须以 `CANDIDATE_CONTRACT_GAP` 结束并列出缺口；不得由 evaluator 猜测或临时拼接 comparator。

### MVP 完成门槛

M0–M5 全部完成，且至少满足以下之一：

1. M5 产生有效的 smoke `directional/na` Verdict；或
2. M5 以结构化 `CANDIDATE_CONTRACT_GAP` 终止，且 M0–M4 全绿、缺口可由后续写码循环直接消费。

这两个结果都证明架构纵切成立。第二种结果说明 candidate 尚不能复现论文，而不是度量管线失败。

---

## 7. MVP 之后才开始的增量层

### P1：A3 写码循环，只接 DevHarness

- 输入 `study.json`、candidate draft 和 DevGateResult；
- 不能调用 FinalEvaluator；
- 每次 DevHarness 新绿即创建不可变 checkpoint commit；
- rollback 只回到已登记的绿色 commit；
- 验收：从 M5 的 `CANDIDATE_CONTRACT_GAP` 修到 smoke 可执行。

### P2：B1 自动 StudySpec draft

- LLM 只能输出 draft；
- 每个 claim 必须附来源证据位置；
- 人工 approve 后才能 freeze；
- 自动稿与人工稿在 fre/rice 上逐字段对照；
- 不引入“自动生成即自动通过”。

### P3：B2 开发测试生成

- 初期只进入 DevHarness；
- 每条测试带证据位置与 expected derivation；
- 对正确 fixture 与 mutation fixture 做校准；
- 未经人工批准不得进入 sealed final harness。

### P4：A1 语料前端

- 只通过 `ReferenceCorpus` 合同把内容交给 A3；
- 下载、索引、空结果和截断全部显式；
- 以“有前端 vs 无前端”的执行级 A/B 决定是否保留。

### P5：调度与远程 GPU

- 本地 DockerExecutor 稳定后再增加 RemoteExecutor adapter；
- 两个 adapter 共享同一个 EvaluationRequest/ObservationSet 接口；
- remote adapter 必须实现 submit、reconcile、cancel、fetch sealed artifacts，而非暴露供应商细节。

### P6：技能库

- 单设 `SkillCurator` 模块；
- 只消费开发集结果；
- 技能有来源、适用条件、验证记录和版本；
- 留出集与最终评测结果永不自动写入技能库。

---

## 8. 统计 policy：MVP 与正式复现分开

### 8.1 MVP policy

- `smoke_direction_only_v1`；
- 允许一个 seed；
- 只报告原始差异或 ratio；
- 结果只能是 `directional`、`na` 或 `error`；
- 不计算论文“复现成功率”。

### 8.2 full policy

每个 claim 在运行前冻结：

- estimand；
- null value 与 practical margin；
- paired/unpaired 设计；
- seed 数；
- 置信区间方法；
- 多 claim 处理；
- 停止规则；
- 缺失 seed 的处理。

默认优先使用配对 seed 的逐 seed 差值。ratio 类 estimand 同时声明 `null_value`；例如耗时 ratio 的 null 为 1.0，若 practical margin 为 0，则 minimize claim 要求置信区间上界小于 1.0。`pass` 必须同时满足：

1. scale 与论文等价，或 deviation 已被预注册并允许；
2. 预注册样本量全部完成；
3. 效果置信区间越过由 null value 与 practical margin 定义的阈值；
4. guardrail 没有越过预注册退化阈值；
5. 没有合同、sandbox 或 evaluator 错误。

论文报告数只能作为独立 reference observation，不能与本机 measured observation 混成同一个 baseline。

---

## 9. 统一退出码

| 退出码 | 含义 | 是否生成 Verdict |
| --- | --- | --- |
| 0 | 管线正常完成；科学结果可为 pass/fail/directional/na | 是 |
| 10 | schema 或语义合同无效 | 否 |
| 11 | digest 或父引用不匹配 | 否 |
| 12 | candidate artifact/路径无效 | 否 |
| 20 | candidate 执行失败或超时 | error Verdict |
| 21 | sandbox violation | error Verdict |
| 22 | observation 不完整或无法 seal | error Verdict |
| 30 | evaluator 内部错误 | error Verdict |
| 40 | 用户取消 | error Verdict |

科学 `fail` 返回 0，因为 evaluator 已正确完成。只有管线失效才返回非零。

---

## 10. 第一轮真实工作的推荐顺序

严格按以下顺序，不并行扩张范围：

1. 创建 `repro/` 与 M0 合同测试；
2. 实现 FixtureExecutor、Evaluator 和 SQLite RunLedger，完成 M1；
3. 检查 trial2 并打包完整 toy candidate，完成 M2；
4. 实现 DockerExecutor 的 toy candidate 测试，完成 M3；
5. 构建 rice Hopper 环境并跑 DevHarness，完成 M4；
6. 仅当 M0–M4 全绿，才尝试 M5；
7. M5 暴露的第一个真实 candidate 缺口，才成为 A3 的首个任务。

任何阶段出现失败，只修当前 seam，禁止同时启动 LLM 编译、语料前端、技能库或远程调度。这样每次失败都能定位到一个模块的外部接口，而不是在整条 Agent 流水线里猜测。
