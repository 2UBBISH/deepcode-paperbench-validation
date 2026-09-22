# rsa (research_setup_agent) 迁入基线

- 上游仓库：`https://github.com/zyt1253679098/research_setup_agent`
- 基线分支：`main`
- 基线提交：`0531abc85b47f99c6e9228fa2b50754c28087ac6`
- 上游包版本：`0.1.0`
- 许可证：上游未提供 LICENSE 文件（第一方内部组件）
- 迁入日期：`2026-08-22`
- 迁入范围：`rsa/` 包目录 25 个 .py。未迁入：`pyproject.toml`、`README.md`、
  `docs/`、`bench/`（评测客户端）。上游 15 个测试文件中的 13 个迁入到
  `Agent/tests_vendor/rsa_upstream/`，原样不改；`test_bench_oracle.py` 与
  `test_dockerutil.py` 未迁入——它们测的是没有迁入的 `bench/`，在收集期就 import 失败。
  `test_measured_defects.py` 里另有 7 个用例在函数体内 `import bench`，由
  `tests_vendor/conftest.py`（DeepEvol 新增，不动上游文件）在运行期翻译成 skip。

## 迁入时实测基线

| 环境 | 结果 |
| --- | --- |
| Linux / py3.11（`python:3.11-slim` + git） | **154 passed, 7 skipped** |
| macOS / py3.11 与 py3.14 | 另有 ~13 个失败，**全部是 macOS 特有** |

macOS 上多出来的失败已定位到单一根因：**上游用了 GNU coreutils 独有的 flag**——
`asset_gate.py:116` 的 `du -sb`、`prelude.py:113` 的 `stat -c %s`，BSD/macOS 的
`du`/`stat` 都不认。这两条命令在生产路径上是**在远端 Linux 容器里执行**的，
所以并不是 bug；只有测试走 `LocalBridge` 在 macOS 开发机上跑才会炸。
已在原始 clone（同 commit、py3.11 与 py3.14）复现同样的失败，**确认迁入没有引入任何回归**。

结论：**CI（ubuntu）与远端执行都不受影响；本地 macOS 上跑上游测试请预期这批失败。**

rsa 负责实验 Agent 链路的「配环境 + 用冻结的确定性判据证明能跑通」这一段：
给一个科研仓库和一句自然语言目标，生成 pytest 判据、在远端 Docker 里配好环境、
并判定它是否真的跑起来了。它**不承诺复现论文数值**。

设计与集成见 `DeepEvol1.0/docs/experiment-agent-design.md`。

## 名字冲突警告

PyPI 上存在同名发行包 `rsa`（纯 Python RSA 加解密，历史上是 `google-auth` /
`oauth2client` / `python-jose[rsa]` 的依赖）。迁入时两个 uv.lock 与两个 .venv 都
**没有**该包（现代 google-auth 已改用 `cryptography`）。一旦未来某个传递依赖把它
带回来，`sys.path` 顺序会决定谁赢，而且是**静默出错**。
守护测试 `test_no_pypi_rsa_distribution_installed` 会在 CI 当场拦下。

同理，本仓**不注册** `[project.scripts] rsa = "rsa.cli:main"`——产品后端不该多一个
叫 `rsa` 的可执行文件，徒增混淆面。要用 CLI 就 `python -m rsa.cli`。

## 与上游的分叉纪律

- P0 **不改** rsa 任何源码。
- 后续给 `static_facts.py` 加资源维度（batch_size / precision / params_b 等）时，
  改动集中在**新增函数与新增字段**，尽量不改既有函数体，保住可 diff 性。
- `RSAAgent.run()` 的 `finally`（`agent.py:127-133`）会对 `config.remote_backend`
  调 `close()` 并置 `None`。**DeepEvol 不改这段**，改为「每次 run 注入一个一次性
  backend，自己另持一条控制连接做记账与释放」。理由：`close()` 可能同时做远端清理
  （kill 作业组、清 jobs 目录），用 no-op 代理挡住会静默泄漏远端状态。
  守护测试 `test_rsa_run_closes_injected_remote_backend` 把这个行为钉住——
  它若在未来变了，DeepEvol 的编排假设也要跟着改。

## 外部前置

- **SetupX**：`setupx_interop.py:40` 要求磁盘上有 SetupX 检出（`<root>/src/agent.py`）。
  已一并 vendored 到 `Agent/setupx/`，由 `Agent/run.sh` 的 `RSA_SETUPX_ROOT` 指路。
- **remote_relay**：`remote_backend.py:_relay_api()` 先试扁平
  `from remote_relay.remote import ...`，失败才走 `RSA_REMOTE_RELAY_ROOT` /
  同级 `remote_relay.zip` 的嵌套 fallback。relay 已平铺 vendored 在 `Agent/remote_relay/`，
  因此**走的永远是第一条**，那两条 fallback 在 DeepEvol 内不触发。
  （那个环境变量的语义有坑：必须指向 checkout 的**父**目录，而错误信息说的是
  "set RSA_REMOTE_RELAY_ROOT to the relay checkout"，具误导性。平铺是从根上消除它。）
- **`base_image: setupx-base:py310-proxy`**：rsa 仓里**没有**对应 Dockerfile，
  假定已存在于远端 daemon。归入自定义镜像预烘。
- **`grader_image: rsa-grader:py311-v1`**：由 `_ensure_grader_image()`
  （`remote_backend.py:241-275`）用内联 Dockerfile 在远端 `docker build`，
  先 `docker image inspect` 命中就跳过——**这就是预烘钩子**，镜像里提前 build
  好同名 tag 即可省下每台新机的重建时间。

## 2026-09-15 搬入 `apps/v2/agent_engine/rsa`

V1 `Agent/` 包已退役、V2 Agent 镜像只装 `apps/v2` + `apps/common`，所以三个
vendored 树平铺到 `apps/v2/agent_engine/{rsa,remote_relay,setupx}`。为此对上游
源码做了两处、且只有两处改动（都带 `DeepEvol:` 注释）：

1. `remote_backend.py::_relay_api` 先按 `apps.v2.agent_engine.remote_relay` 找
   relay，再退回上游的扁平名与 zip 路径。
2. `setupx_interop.py::setupx_root` 的默认值改为 **rsa 的兄弟目录** `../setupx`
   （上游假设 setupx 是 rsa *仓库* 的兄弟）。产品路径总是显式设
   `RSA_SETUPX_ROOT`（`experiment/run_flow.py::_setupx_env` 每进程拷一份），
   这个默认值只服务于测试与脚本。

上游测试搬到 `tests/vendor_agent_engine/rsa_upstream/`，`conftest.py` 里把
`rsa` / `remote_relay` 两个扁平名注册为这两个包的别名，上游测试文件本身不改。
