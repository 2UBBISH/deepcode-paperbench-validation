# remote_relay 迁入基线

- 来源类型：**第一方内部组件**（无公开上游，源 checkout 不是 git 仓库）
- 来源路径：`<workspace>/remote_relay/remote_relay/`（解压出的源码副本，git untracked）
- 包名 / import 名：`remote-relay` / `remote_relay`
- 上游包版本：`0.1.0`
- 许可证：`Apache-2.0`（源 checkout 无 LICENSE 文件，仅 pyproject 的 SPDX 声明）
- 迁入日期：`2026-08-22`
- 迁入范围：内层 `remote_relay/` 包目录 18 个 .py（含 `execution/`、`fs/`、
  `transport/`、`diagnostics.py`）。未迁入：`pyproject.toml`、`README.md`、
  `.venv/`、`dist/`。上游 7 个测试文件迁入到
  `Agent/tests_vendor/remote_relay_upstream/`，原样不改。上游的真机冒烟脚本
  `scripts/live_relay_smoke.py` 迁入为 `Agent/scripts/remote_relay_live_smoke.py`。

remote_relay 把一台 SSH 机器包成 Agent 可直接使用的 `Runtime`，本地与远端同一组
方法。`wait_ready(timeout=300)` 专为「刚创建、sshd 还没起、cloud-init 还在跑」的
云机器写的：认证失败立即抛不重试，其余轮询到超时。这正是实验 Agent 从
`CreateInstance` 到「可以交给 rsa」之间那段窗口需要的东西。

## 基线指纹

无 upstream commit 可记，改用内容哈希。复现命令（在源 checkout 根执行）：

    find remote_relay -name "*.py" -exec shasum -a 256 {} \; | sort | shasum -a 256

    基线值：7fe8c1f9b06171fac5b5c1401d0d6c6638677e54c2089212fb76e7936d96f301

源 checkout 的 `dist/remote_relay-0.1.0-py3-none-any.whl` **不是**本次基线，
也不得用于校验：它比源码旧，缺 `execution/jobs.py` 里 stderr 重定向顺序的修复
（源码是 `tail ... 1>&2 2>/dev/null`，wheel 是 `tail ... 2>/dev/null >&2`，后者会把
日志一起吞进 /dev/null）。**不要图省事改回装 wheel。**

## 单一真源声明

自本次迁入起，**DeepEvol1.0 是 remote_relay 的唯一真源**。工作区里的
`<workspace>/remote_relay/` 与 `remote_relay.zip` 即刻降级为历史归档，不再接受改动，
也不得再被任何构建或运行路径引用。任何修复都直接改本目录并配套测试，
不要再「改外面、同步进来」。

## 引用方式与依赖

- 平铺在 `Agent/` 下，而 `Agent/` 本就在 sys.path 上（根 `pyproject.toml` 的
  `pythonpath = [".", "Agent"]`；`deploy/Dockerfile.agent` 的
  `PYTHONPATH=/app:/app/Agent`）。所以 `import remote_relay` 开箱即用，
  **不需要新环境变量、不需要 wrapper 脚本**——这是相对 vendored OpenCode 那套
  `DEEPEVOL_OPENCODE_BIN` 间接层的改进。
- 这同时让 `rsa/remote_backend.py:_relay_api()` 命中第一条扁平 import，
  `RSA_REMOTE_RELAY_ROOT` 与 `remote_relay.zip` 两条 fallback 永不触发。
- 唯一依赖 `asyncssh>=2.14,<2.22`，已并入 `Agent/pyproject.toml` 主 dependencies。
  **上界的理由**：`asyncssh>=2.22` 要求带 ML-KEM 的 `cryptography(>=50)`，
  该版本的 macOS wheel 在本机 dlopen 时缺 OpenSSL 符号，import 即崩。
  CI 跑 ubuntu 不会暴雷，放开上界只会炸本地开发。等 cryptography 修好再放开。

## 2026-09-15 搬入 `apps/v2/agent_engine/remote_relay`

包内全部是相对导入，搬入零改动。V2 代码用
`from apps.v2.agent_engine.remote_relay import RemoteRuntime`。上游测试搬到
`tests/vendor_agent_engine/remote_relay_upstream/`，靠 conftest 的别名保持
`from remote_relay import ...` 原样可用。
