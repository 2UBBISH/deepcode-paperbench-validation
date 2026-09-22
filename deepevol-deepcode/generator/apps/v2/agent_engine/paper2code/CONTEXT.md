# Paper2Code Kernel

从 HKUDS/DeepCode 中切出的论文复现业务层：把一篇论文或技术文档变成一个可运行的代码仓库。本目录只含业务逻辑和它对外的契约，不含模型循环。

## Language

### 整体

**复现管线**：
从输入源到生成代码目录的整条流水线，由 `execute_multi_agent_research_pipeline` 驱动。
_Avoid_：工作流、pipeline、流程、Paper2Code 管线

**阶段**：
论文（arXiv 2512.07921）对系统的三段划分：蓝图生成、代码生成、验证修正。只在与论文对照时使用。
_Avoid_：Phase（指论文时）

**步骤**：
复现管线代码里按进度百分比排列的十一个执行单元，从环境准备到收尾。
_Avoid_：phase（指代码时）、stage、环节

**任务目录**：
一次复现运行的全部文件所在的目录，位于 `<工作区>/.deepcode/workflows/tasks/paper_<id>/`。
_Avoid_：paper_dir、输出目录、工作目录

**工作区**：
接入方指定的根目录，任务目录在它下面创建。
_Avoid_：workspace root、项目目录

### 输入

**输入源**：
用户提交的那个东西：本地文件路径、`file://` 路径或 `http(s)://` 地址。
_Avoid_：input、paper、论文文件

**输入种类**：
输入源按扩展名归入的六类之一：pdf、md、docx、txt、html、url。
_Avoid_：input_kind、格式、文件类型

**获取**：
把输入源复制或下载到任务目录并转成 Markdown 的那一步。本地文件是复制，不是移动。
_Avoid_：下载、导入、acquire

### 蓝图生成

**分段**：
把 Markdown 论文切成的一个带标题、类型、关键词和相关性分数的片段。
_Avoid_：chunk、切片、section

**分段索引**：
一篇论文全部分段的清单，存为 `document_index.json`，含文档类型和切分策略。
_Avoid_：document index、segments

**分段模式 / 全文模式**：
规划器的两种输入方式：前者喂排名靠前的分段，后者喂整篇 Markdown。由文档长度是否超过阈值决定。
_Avoid_：segmented / traditional

**规划器**：
把论文内容变成蓝图的那个 Agent。当前实现只有一个，不再并行运行概念分析和算法分析两个子 Agent。
_Avoid_：Code Planning Agent、planner agent、code analyzer

**蓝图**：
规划器产出的五段式 YAML 计划：文件结构、实现组件、验证方案、环境要求、实施策略。存为 `initial_plan.txt`。
_Avoid_：implementation plan、initial plan、计划、方案

**计划审阅**：
蓝图生成后、代码实现前的可选人工闸门。接入方提供回调，返回批准、修改、替换或取消。
_Avoid_：plan review、审批、确认

### 参考与索引

**参考仓库**：
从论文参考文献里找到并克隆到任务目录 `code_base/` 下的外部代码库。
_Avoid_：reference repo、下载的仓库、code base

**代码检索**：
论文里的 CodeRAG。对参考仓库逐文件摘要、与蓝图目标文件建立关系，存为 `indexes/<repo>_index.json`，实现时按需查询。
_Avoid_：CodeRAG、索引、code indexing、RAG

**关系**：
代码检索里一条"参考仓库某文件对蓝图某目标文件有用"的记录，带类型、置信度和用法建议。类型只有四种：direct_match、partial_match、reference、utility。
_Avoid_：relationship、映射、link

### 代码生成

**代码记忆**：
论文里的 CodeMem。每写完一个文件就生成一条结构化摘要（核心目的、公共接口、依赖、下一目标），追加到 `implement_code_summary.md`，然后清空对话只保留系统提示、蓝图和这条摘要。
_Avoid_：CodeMem、memory、summary、上下文压缩

**实现循环**：
代码生成步骤里跑在 AgentRunner 上的那个多轮工具调用过程。一轮是一次模型请求。
_Avoid_：implementation loop、kernel loop、生成循环

**生成代码目录**：
实现循环写文件的地方，`<任务目录>/generate_code/`。
_Avoid_：code_directory、输出代码、产物

### 验证

**验证**：
在生成代码目录里机械发现测试命令（pytest、npm test、cargo test）并各跑一次，只记录通过与否。不包含修正：失败不会回到实现循环。
_Avoid_：测试、verify、闭环、修复

**严格模式**：
`strict_outcomes=True`。没有发现测试、或任一测试失败，管线状态即为未完成。
_Avoid_：strict、verification required

**快速模式**：
`enable_indexing=False`。跳过参考挖掘、参考仓库获取和代码检索三个步骤，蓝图直接进实现循环。
_Avoid_：fast mode、无索引模式

### 边界

**接缝**：
业务层从外部需要、本目录不实现的接口。全部在 `seams/` 下，共十个方法。
_Avoid_：接口、port、adapter、依赖

**接入方**：
实现接缝并调用复现管线的那一方。
_Avoid_：调用方、用户、host

**支撑代码**：
从 DeepCode 原样拷来、不依赖其他部分的真实代码，在 `support/` 下。不是接缝。
_Avoid_：工具库、utils、helpers
