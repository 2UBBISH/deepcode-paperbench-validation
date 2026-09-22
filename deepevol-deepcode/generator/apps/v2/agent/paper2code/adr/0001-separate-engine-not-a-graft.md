# ADR 0001 · Paper2Code 线是独立引擎，不嫁接进复现线

日期：2026-09-16
状态：accepted

## 背景

仓库里已经有一条论文复现产物线（`feature/reproduction-c`，十阶段，Product 表 `reproduction.*`，迁移 0044 到 0057）。它对 DeepCode 做过对齐（`M3_DEEPCODE_ALIGNMENT.md`），但架构上有两条硬规则：阶段内模型不持有工具、只对冻结证据输出结构化 JSON；Agent 进程从不执行生成的代码，一切执行是远程持久任务。

我们从 HKUDS/DeepCode 切出的业务层（`paper2code-kernel`）恰好建立在相反的机制上：模型持有 `write_file` / `execute_python` 等工具跑多轮循环，每写完一个文件用代码记忆清空上下文，验证在本地子进程里跑。

owner 的验证仓库（`2UBBISH/deepcode-paperbench-validation`）在 PaperBench bam 上给原装 DeepCode 打出 0.8367，复现线那一臂停跑无分。owner 的要求是：在 DeepEvol 里有一条能跑的 DeepCode 臂，只和原装 DeepCode 比分。

## 决定

Paper2Code 是一条**独立的产物线**：kernel 作为引擎 vendor 到 `apps/v2/agent_engine/paper2code/`，接缝实现与驱动器在 `apps/v2/agent/paper2code/`，phase 词汇按 owner 的十一段表，run 状态先落文件不落 Product 表。不修改复现线的任何文件；需要它的零件（论文包读取、远程 relay、Aliyun 租约、footprint 守卫）按文件手工搬入本线目录并记录来源。

## 备选

1. **嫁接**：把 kernel 的规划器和实现循环装进复现线的 Stage 5 / 8 / 9 执行器，替换它的纯 JSON 适配器。否决：与"模型不持工具、不本地执行"互斥，要么改掉复现线的规则，要么阉割 kernel 的循环，两者都失去"和原装 DeepCode 比"的意义。
2. **作为复现线的第二个 program adapter**：保留复现线的 Product 骨架，只换 Stage 8 的实现。否决：Stage 8 的输入输出契约（`TYPED_VERSIONS[8] = 12`，冻结的 bundle 与 readiness 文档）是为无工具模型设计的，kernel 的产物形态对不上；而且复现线的迁移号与 main 冲突，尚未合并，也还没跑通一篇真论文，在它上面叠东西等于同时背两份未验证的债。
3. **等复现线合并后再说**。否决：owner 需要对照臂的时间点在复现线之前。

## 后果

- 仓库里并存两条复现线，各自评分。谁吸收谁，等两边在同口径下有分数再定。
- 本线的 Product 表、迁移、API 推迟；届时按产物线骨架建自己的 schema，不复用 `reproduction.*`。
- 从复现线搬来的文件在本线目录下独立演化，不回流；`footprint.yaml` 记录每一处对线外文件的触碰。
- 执行必须远程（owner 的要求：产品化后多个本地跑会崩），因此 kernel 的本地沙箱执行在本线里被执行端口替换，这一点和复现线的"不本地执行"一致，只是执行的粒度不同：kernel 保留工具循环，执行调用逐次远程。
