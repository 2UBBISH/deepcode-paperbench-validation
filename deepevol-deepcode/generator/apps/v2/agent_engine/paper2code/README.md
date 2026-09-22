# paper2code-kernel

从 [HKUDS/DeepCode](https://github.com/HKUDS/DeepCode) 切出来的论文复现业务层。输入一篇论文，输出一个代码仓库；中间的模型循环、MCP 连接、权限策略留成接缝，由接入方提供。

**本目录不能直接运行。** 接缝未实现时每个入口都抛 `NotImplementedError`。

## 文档

| 文档 | 读它来 |
| --- | --- |
| [CONTEXT.md](CONTEXT.md) | 统一词汇。其他文档和代码注释都按它说话 |
| [docs/INTEGRATION.md](docs/INTEGRATION.md) | 接入：十个接缝各要做什么、循环的行为约定、配置、回调、MCP 服务器表 |
| [docs/ARTIFACTS.md](docs/ARTIFACTS.md) | 输出：任务目录每个文件的格式、管线返回值、状态词汇 |
| [docs/LOGIC.md](docs/LOGIC.md) | 逻辑：十一个步骤、三个核心机制、开关、与论文的对应和缺口 |

## 目录

```
workflows/   复现管线：步骤编排、规划、审阅、实现循环、代码记忆
tools/       五个 MCP 服务器、文档转换与下载、代码索引器
prompts/     系统提示词
utils/       文件处理、配置读取、循环检测
seams/       接缝：接入方要实现的十个方法，及其数据类型
support/     从 DeepCode 拷来的无依赖真代码
```

四个业务目录放在根下，导入路径不变（`from workflows.agent_orchestration_engine import ...`），把本目录加进 `sys.path` 即可。Python 3.12。

## 来源

上游提交：`21ebc57fbcab3e3a238976771a0e06115aa30da6`（HKUDS/DeepCode main，2026-09-16），记录在 `UPSTREAM_COMMIT`。

git 历史按四步切分，每步的 diff 可以单独审阅：

1. `workflows/`、`tools/`、`prompts/`、`utils/` 原样拷贝
2. 应用本地补丁（下表）
3. 把对 `core/` 的 42 处导入改到 `seams/` 和 `support/`
4. 文档

### 相对上游的改动

| 文件 | 改动 | 原因 |
| --- | --- | --- |
| `workflows/planning_runtime.py` | `extract_yaml_candidate` 只认无缩进的代码栏 | 蓝图块标量里嵌套的代码栏会截断 YAML |
| `workflows/agents/code_implementation_agent.py` | 统计 `write_multiple_files` 写的文件 | 批量写入的运行报 0 个文件，永远到不了完成判定 |
| `workflows/code_implementation_workflow.py` | 循环检测对写工具按参数分键；验证前 `resolve_project_root` 下探一层 | 写不同文件不是循环；生成代码在子目录时测试发现不到 |
| `workflows/environment.py` | `_BraceLoggerAdapter` | 传 stdlib logger 时 loguru 风格的 `{}` 调用会抛 `TypeError` |
| `support/verification.py` | `resolve_project_root` | 同上第三行 |
| 全部 | `from core.` → `from seams.` / `from support.` | 切断对 DeepCode 的依赖 |

符号名一律未改。

## 状态

- 51 个模块可独立导入，不加载任何 DeepCode `core.*` 模块。
- 接缝里标为真代码的部分经过冒烟测试；契约部分只验证了会抛 `NotImplementedError`。
- 没有带测试。上游的 `tests/` 里有 10 个不依赖 `core` 的测试文件可以直接搬来用。
- 论文"阶段三"的闭环纠错在上游就没有实现，本目录也没有做。挂点见 [LOGIC.md](docs/LOGIC.md) 第 8 节。
