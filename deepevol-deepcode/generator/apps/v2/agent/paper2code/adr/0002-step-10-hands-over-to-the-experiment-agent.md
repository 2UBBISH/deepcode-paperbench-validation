# ADR 0002 · 第 10 步交给 main 的实验 Agent，本线只加修复轮

日期：2026-09-17
状态：accepted

## 背景

本线第 10 步（`environment_run`）今天是自有的一套：`aliyun_lease.py` 直连 ECS 租机、`job_executor.py` 在一次性容器里跑作业（首个作业把 `requirements.txt` 烤进镜像，装失败静默退回基础镜像）、compileall 与入口冒烟只记录。PLAN-3 原来的第 4 项打算在这上面继续叠：从 `environment_spec.json` 推一份确定性"配方"（apt → pip → 数据集 → import 自检），再加修复轮与缩减规模跑。

翻 main 时发现 `apps/v2/agent_engine/experiment/run_flow.py` 的 `run_experiment_on_machine` 已经是同一件事的完整产品链路：租机 → RSA（`agent_engine/rsa`：把一句目标编译成冻结的 pytest 判据 → vendored SetupX 在容器里 ReAct 配环境 → 确定性裁决）→ 报告 + 证据 → 按策略还机。它接受本地目录（机器上起 git daemon 供 bundle），真机跑过 micrograd（8 次模型调用配好并跑通）与 nanoGPT（环境装好、训练跑完，判据的正则没对上日志）。RSA 的判据是一架梯子 G0（import 图、入口能解析）→ G1（资产在不在）→ G2（最小规模整跑 + 产物清单）→ G3（缩减配置短实验，要批准）→ G4'（全量，要批准），与本线第 10 步的入口冒烟 / 缩减规模跑 / 全量跑一一对上。

owner 的判断：生成的 `requirements.txt` 和环境规格"不一定对"，一个按目标去装、装完用判据证明的 agent 比一份从生成物推出来的配方可信；本线该复用 DeepEvol 已有的零件（算力估算、ask_user 已如此），不该在同一产品里养第二套配环境实现。

## 决定

`environment_run` 阶段整体调用 `run_experiment_on_machine`；本线的贡献只有两件：

1. **注入点上的循环**：`rsa_runner` 是本线的——第 0 轮让 RSA 编判据、配环境、裁决 G0–G2；未通过则在**同一个容器**里进修复轮（本线自己的 ReAct agent 读/写 `generate_code/`，可在容器里 exec 探针，探针前 checkpoint、裁决前 rollback），失败归因为环境性的（非本仓模块 `ModuleNotFound`、`command not found`、资产缺）就走 SetupX 的 `run_round(container_id=…)`，否则改代码；改了 `requirements.txt` 下一次强制先环境轮；每轮重新 bundle、容器拉新代码、用**同一份冻结判据**裁决。轮次由 `repair_rounds` 定（默认 3，对比运行 0），用尽后是审阅点（continue / accept / abort）。
2. **两个薄适配**：本线的 ECS 直连包成 `LeaseBackend`（产品接入时换成 API 侧的）；进程内起一个回环 `/v1/chat/completions` 转给本线的 provider，SetupX / RSA 的 `.env.small` 指向它——thinking 关、每次调用落 `llm/`、`reasoning_tokens` 计入口径、key 不落盘。

RSA 与 SetupX 源码一字不改（沿用 `rsa/DEEPEVOL_VENDOR.md` 的纪律），本线只调 `run_flow.py` 已经在用的两个内部入口 `setup_loop.run_round` 与 `Adjudicator`。租机、还机、`needs_user` 持机 + 告警、OOM 升档、硬上限、证据收集、各项预算全部继承 main，本线不另定。

判据的目标句由本线用固定模板生成（蓝图 `validation_approach` + 入口命令 + 约束：CPU、最小规模、不可安装的外部工具），不再另做一次结构化抽取；RSA 的 `clarification` / `criterion_review` / 升级卡接成本线的审阅点（决定文件，`--ask` 之外自动通过）。

复用的"环境"是**容器**：RSA 成功后 `docker commit` 成 run 镜像，`SET_ENV` 值与每轮命令账本记进 `environment.json`；账本只作记录，不重放。

sapg 真机走通后，本线自己的执行端口（`job_executor.py`、`leased_runtime.py`、`aliyun_lease.py` 的租机部分）退役，只留 `release` 备份命令。

## 备选

1. **本线自有配方**（PLAN-3 原第 4 项）：0 次模型调用、口径最干净。否决：配方是从"不一定对"的生成物推出来的，装不上只能列 `missing[]` 交给修复 agent，等于把配环境这件事再发明一遍，而 main 已经有一个会按目标装、装完能证明的；同一产品两套配环境实现必然分叉，第 9 项产品接入时还得收敛。
2. **配方 + SetupX 兜底**：两套"智能"叠加，出了问题分不清是谁的。否决。
3. **每轮修复都整跑一次实验 Agent**（新容器、SetupX 从头配）：和产品一模一样、最简单。否决：每轮环境可能配得不一样，信号变化分不清是代码改好了还是环境变了；判据冻结而环境不冻结，与 RSA 自己的原则相悖。
4. **等人时释放机器、答复后整机快照恢复**：省钱。否决：偏离 main 的释放策略又是一条分叉，且快照本身 34–81 分钟；`--run-hours` 硬上限兜底，产品里 `needs_user` 持机 + 告警是既有设计。
5. **让本线在 `implement` 后自己写 `tests/`，G2 = `pytest`**：测试会进提交物且是同一个模型写的，自证；可另立一项，不在这批。

## 后果

- 第 10 步的"环境"不再是一份可读的配方，而是一个容器 + 一份账本 + 一份冻结判据；跨机器（换 GPU 档）不移植容器，由 SetupX 在新机上按账本提示重配——这与 main 升档时"新容器、重新 clone、重配"的做法一致（`run_flow.py:954` 的注释）。
- 本线多出对 `agent_engine/experiment`、`rsa`、`setupx` 三个包的依赖，以及机器上两个预烘镜像（`setupx-base:py310-proxy`、`rsa-grader:py311-v1`，`deploy/experiment-images/`）；CLI 模式下没有 `deepevol-linux-*` 镜像时现建。
- 口径规则延伸到 RSA / SetupX 的调用：它们只有经回环端点才算合规；直接写厂商端点进 `.env.small` 视为违反口径。
- 成功后外壳按策略**还机器**，commit 出的镜像随机器消失；第 8 项全量跑 GPU 要在还机前整机快照（main 的 `upgrade` 已有此路），届时再定。
- 原第 6 项"缩减规模跑结果"并入 G2，不再单列；PaperBench Code-Dev 裁判不执行代码，执行只服务修复轮与第 8 项。
- ADR 0001 的"执行必须远程"不变；变的是执行的载体：从本线的一次性作业容器换成实验 Agent 的常驻容器。
