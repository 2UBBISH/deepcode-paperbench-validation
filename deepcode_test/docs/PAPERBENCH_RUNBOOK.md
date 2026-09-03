# DeepCode + PaperBench 跑通指南

> 生成于 2026-08-25。两个仓库均在 `/home/deepevol/deepevol/` 下:
> `DeepCode/`(HKUDS 源码)与 `frontier-evals/project/paperbench/`(OpenAI 官方基准)。

> **📌 2026-08-26 深夜实况**:阶段 A(dummy 白卷 + 真裁判 DeepSeek-V4-Pro + code_only)
> **已跑通**:178/178 条全部有效判分、0 失败,白卷得 0 分(预期正确)。
> 途中又修掉两个"OpenAI 中心主义"坑(见坑 5/坑 6)。实测判分成本 ≈ **¥17/篇**
> (高于预估,因每叶子携带全文论文),触发成本闸门 → B/C 烧钱环节暂停待批。
> **B 已备好一键脚本**:`bash ~/deepevol/run_stage_b.sh [--fast]`(DeepCode 复现 rice → 判分);
> **C 镜像已补齐**(python/pip/jupyter/PEP668 全解);另有无 GPU 的 **SWE-bench 备选线**
> (DeepCode 自带 `eval/swebench/` 挂架)。晨报详见 `MORNING_REPORT.md`。

---

## 一、环境现状(已替你完成 ✅)

| 项目 | 状态 |
| --- | --- |
| DeepCode 源码 + 依赖(`.venv`,`deepcode.py --version` = 2.1.0) | ✅ |
| paperbench 源码 + 4 个同仓库依赖包(`project/common/*`) | ✅ |
| paperbench Python 依赖(`uv sync`,入口 `--help` 已验证可跑) | ✅ |
| 论文数据(LFS 水合 `data/papers/**`:441 个对象 / 535MB 传输 / 落地 205MB,PDF 已验真) | ✅ |
| Docker 镜像 `pb-env`(5.38GB)+ `pb-reproducer`(1.36GB) | ✅ |
| `.env` 已填**硅基流动** key + base_url(2026-08-26,chat 接口实测连通 ✅) | ✅ |
| uv 0.12.5、git-lfs 3.7.0(装在 `~/.local/bin`) | ✅ |
| 本机:Python 3.12 / Docker 29.6 / RTX 4060 Laptop (8GB) | ✅ |

> 提示:`~/.local/bin` 若不在你的 PATH,先执行 `export PATH="$HOME/.local/bin:$PATH"`(建议加进 `~/.bashrc`)。

---

## 二、你唯一需要亲手做的事:填 API Key

**① PaperBench 主配置** —— 编辑 `frontier-evals/project/paperbench/.env`:

```
OPENAI_API_KEY=sk-...        # 必填:跑 BasicAgent 和 AI 裁判都用它
GRADER_OPENAI_API_KEY=       # 可选:想让裁判用另一个 key 才填,默认复用上面的
ANTHROPIC_API_KEY=           # 可选:agent 想用 Claude 时填
OPENROUTER_API_KEY=          # 可选
```

**①-b 实际采用:硅基流动(2026-08-26 已配置完成 ✅)**:裁判/agent 的客户端是裸的
`openai.AsyncClient()`,吃环境变量。`.env` 已写成:

```
OPENAI_API_KEY=<你的硅基流动 key,已从 llm_models.yaml 抄入>
OPENAI_BASE_URL=https://api.siliconflow.cn/v1
PB_STRUCTURED_PARSER_MODEL=deepseek-ai/DeepSeek-V4-Pro   # 见坑6
```

要点:模型名用硅基流动格式(如 `deepseek-ai/DeepSeek-V4-Pro`);裁判默认
`grade_locally=True` 在宿主机跑,直接继承环境变量 ✅;paperbench 启动时自动
`load_dotenv()`,**不需要 source**。⚠️ 实测硅基流动**没有** Responses API(404),
所以官方 BasicAgent 考生用不了(它只会这门方言)——考生走 dummy 或 DeepCode。

**② 容器内 agent 的 key** —— 编辑 `frontier-evals/project/paperbench/paperbench/solvers/agent.env`:

```
OPENAI_API_KEY=sk-...   # 个别论文(bbox 等)复现时 agent 自己要调 OpenAI
HF_TOKEN=hf_...         # 个别论文要从 HuggingFace 下 ImageNet/Llama-2,需申请权限
```

只是冒烟/轻量跑可以先留空。

**③ DeepCode 的模型连接**(想跑 DeepCode 复现论文才需要):

```bash
cd /home/deepevol/deepevol/DeepCode
.venv/bin/python deepcode.py provider set my-openai --template openai --api-key
.venv/bin/python deepcode.py provider test my-openai --model gpt-4.1
```

(`--template` 也支持 openrouter / anthropic / deepseek / ollama 等;`--api-key` 会弹出隐藏输入。)

---

## 三、跑 PaperBench:三级火箭,由免费到烧钱

所有命令都在 **`/home/deepevol/deepevol/frontier-evals/project/paperbench/`** 目录下执行。

### 🚀 L1 冒烟测试(不花一分钱,不需要任何 key)

假 agent + 假裁判 + debug 数据集(只有 `rice` 一篇),验证 Docker/数据/框架全链路通不通:

```bash
GRADER_OPENAI_API_KEY=dummy-smoke-only uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.solvers.dummy.solver:PaperBenchDummySolver \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.scaffold=dummy \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

⚠️ 六处与官方仓库的落差,都是实测踩坑后的修正:
1. 必须**显式加一行** `paperbench.solver.computer_runtime=...AlcatrazComputerRuntime`——
   README 的命令已与代码漂移,chz 参数系统按基类展开,不先指定实现类就设不了 `.env.*` 子参数;
2. 代码里有无条件断言要求 `GRADER_OPENAI_API_KEY` 存在,dummy 裁判用不到它,
   给个占位值即可(真发生 API 调用会立刻报错,不会偷偷扣费);
3. **复跑(reproduction)的运行时要单独再配一遍**(上面第 4–6 行参数)——
   README 的 dev 示例漏了它,导致复跑阶段去 Docker Hub 拉不存在的
   `pb-reproducer:latest` 而失败(`pull_from_registry` 默认为 True);
   官方只在 canonical 命令里写对了。
4. **官方 `pb-reproducer` 镜像与官方运行时自相矛盾**:公开 Dockerfile 造的镜像
   只有 `python3`,而 Alcatraz 启动容器时强制探测 `python` 和 `pip` 两个命令,
   缺了直接拒绝。修法:给镜像补一层 `python-is-python3 + python3-pip`
   (bootstrap.sh 第 5 步已自动处理,幂等)。
5. **模型上下文长度表是 OpenAI 专属硬编码**(`common/preparedness_turn_completer/utils.py`
   的 `CONTEXT_WINDOW_LENGTHS`),非 OpenAI 裁判直接 `ValueError`,且无配置注入口。
   修法:表里加了一行 `deepseek-ai/DeepSeek-V4-Pro: 1_000_000`(带 `[local compat]` 注释)。
7-11. **【订正 2026-08-26 下午】坑 11(f-string 语法错误)为我方误判**:DeepCode 官方
   `python_requires=">=3.12"`,该语法在 3.12 合法;是我们最初 `uv sync` 时被默认建成
   **3.11 venv**(uv 警告 "No requires-python found, defaulting to >=3.11" 被忽略)。
   责任在环境搭建,补丁本身双版本兼容故无害。待办:venv 重建为 3.12。
7-12. **索引阶段无幂等**:规划/参考/下载都有"产物已存在则复用",唯独
   `orchestrate_codebase_intelligence_agent` 进来必全量重建(¥3~30/次白烧)。
   已加 `[local compat]` 跳过逻辑(indexes/ 内逐仓库 `_index.json` 齐全即复用)。
7. **【DeepCode 侧】`deepcode init` 不写自己论文流水线的 MCP 服务器**:生成的
   `~/.deepcode/deepcode_config.json` 没有 `tools.mcpServers`,于是所有 agent
   在"零工具"状态运行 —— 现象是日志刷 `MCP server(s) [...] not in
   deepcode_config.json`,参考分析只吐 475 字节垃圾(还夹带 `<｜DSML｜tool_calls>`
   泄漏标记),**第 9 步写码阶段没有 `write_file`,产物必然为空**。
   修法:手工补 `tools.mcpServers` 三项,且**必须用模块方式启动**
   (`python -m tools.code_implementation_server`,直接跑脚本路径会
   `ModuleNotFoundError: No module named 'core'`)。
   验证:工具在注册表里叫 `mcp_<server>_<tool>`,经 `build_aliased_registry`
   映射回 `write_file` 等短名,`MISSING=[]` 才算通。
   另需补 `filesystem`(`npx -y @modelcontextprotocol/server-filesystem <dir>`,
   Node 18 报 EBADENGINE 警告但可用,14 工具)与 `fetch`(`uvx mcp-server-fetch`,2 工具),
   否则 Phase 6 参考分析只会回一句"请把论文内容给我"(它没有读文件的工具)。
8. **【DeepCode 侧】推理模型 + 默认 8192 token = 写码必死**:实测同一道
   "写 80 行 RND 模块"的题目——
   `deepseek-ai/DeepSeek-V4-Pro`:**225 秒**、耗 12312 输出 token(绝大部分烧在推理);
   `moonshotai/Kimi-K2.7-Code`:**29.6 秒**、2236 token、干净收尾。
   默认 `maxTokens=8192` 装不下"推理+代码",于是连续返回空响应/截断输出,
   300 秒无进展直接被 LoopDetector 熔断(现象:`Empty response on turn N`
   → `Output truncated` → `🐌 Progress stall`,只写出 3/24 个文件)。
   修法:`~/.deepcode/deepcode_config.json` 里
   `agents.defaults.maxTokens=32768`,并给写码单独指定代码专精模型:
   `agents.implementation = {"model":"moonshotai/Kimi-K2.7-Code","maxTokens":32768}`
   (规划相位仍可留给 DeepSeek——它慢但方案质量好)。
   ⚠️ **模式不对称陷阱**:`enable_indexing=True`(完整模式)只给模型
   **2 个工具**(write_file + search_code_references);`False`(fast 模式)反而给
   **11 个**(含 read_file/execute_bash)。所以当索引为空时,"完整模式"是最弱配置——
   要么把 CodeRAG 真正跑通,要么就用 fast 模式,别用"完整模式 + 空索引"。
6. **裁判的二级"格式解析"完成器写死 `gpt-4o-2024-08-06`**(`judge/simple.py`),
   每条判词都要经它转成结构化分数 → 非 OpenAI 端点上 178 条全灭。
   修法:改为读环境变量 `PB_STRUCTURED_PARSER_MODEL`(不设则保持官方默认),
   `.env` 已配置为 DeepSeek-V4-Pro;实测硅基流动支持 json_schema 结构化输出 ✅。

**当前实测进度(2026-08-26 深夜)**:上述 6 坑全部修毕。
- **阶段 A 已跑通**:dummy 白卷 + 真裁判(DeepSeek-V4-Pro)+ code_only,
  178/178 条有效判分、0 失败,白卷 0 分(预期正确);实测判分成本 ≈ ¥17/篇
  (in 5.26M + out 0.24M tokens)。
- 早前"复跑仍有一处失败"的真因也已定位修复:容器内 `pip install jupyter`
  被 Ubuntu 24.04 PEP 668 拦截 → 镜像补丁 v2 预装 jupyter + `PIP_BREAK_SYSTEM_PACKAGES=1`。
- 阶段 B(DeepCode 复现 rice → 判分)万事俱备:`bash ~/deepevol/run_stage_b.sh [--fast]`;
  阶段 C(全量复跑,吃 GPU 有风扇噪音)镜像就绪待点火。

### 🚀 L2 开发小跑(几美元级,需 OPENAI_API_KEY)

dev 数据集 + 内置 BasicAgent + 便宜模型裁判、短超时:

```bash
source .env
uv run python -m paperbench.nano.entrypoint \
    paperbench.solver=paperbench.solvers.basicagent.solver:BasicAgentSolver \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
    paperbench.paper_split=dev \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model='gpt-4.1-mini' \
    paperbench.reproduction.timeout=60 \
    runner.max_retries=0 \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

**强烈建议再加一个参数:`paperbench.judge.code_only=True`**
这是官方的 **Code-Dev 轻量变体**:跳过"GPU 容器里复跑代码"环节,只评"代码实现对不对",
裁判成本降约 85%,而且基本不吃你的 8GB 笔记本 GPU。

### 🚀 L3 正式全量(⚠️ 极贵,先想清楚)

官方"标准跑法"是 20 篇论文 × 每篇 24 小时 agent(gpt-5)+ 大模型裁判 + GPU 复跑,
API 费用轻松达到**数百~上千美元**,且 8GB 笔记本 GPU 跑复现环节非常勉强。
完整命令见 `README.md` 的 "Canonical command" 一节;想用 GPU 就在各 runtime 上加
`...env.is_nvidia_gpu_env=true`(需要 NVIDIA Container Toolkit)。
个人跑分推荐:`paper_split=all` + Code-Dev 变体 + 便宜些的模型。

---

## 四、把 DeepCode 接进来:复现论文 → 只评分

DeepCode 仓库里**没有**开源它当年跑 PaperBench 的挂架(`eval/` 里只有 SWE-bench)。
但 PaperBench 提供了 `PBDirectSubmissionSolver`:**你给成品代码,它跳过 agent 环节直接按 rubric 打分**。
组合路线:

**第 1 步** 用 DeepCode 复现某篇论文(以 `bam` 为例):

```bash
cd /home/deepevol/deepevol/DeepCode
.venv/bin/python deepcode.py        # 进入交互界面,选 Paper2Code,
                                    # 喂它 ../frontier-evals/project/paperbench/data/papers/bam/paper.pdf
```

产物在其任务目录的 `generate_code/` 里。

**第 2 步** 摆成 PaperBench 要求的目录结构:

```bash
mkdir -p ~/pb_submissions/bam/submission
cp -r <DeepCode任务目录>/generate_code/* ~/pb_submissions/bam/submission/
```

(`bam` 必须与 `data/papers/` 里的论文目录名一致;多篇就多建几个文件夹。)

**第 3 步** 只评分(真裁判 + Code-Dev 省钱模式):

```bash
cd /home/deepevol/deepevol/frontier-evals/project/paperbench && source .env
uv run python -m paperbench.nano.entrypoint \
    paperbench.paper_split=debug \
    paperbench.solver=paperbench.solvers.direct_submission.solver:PBDirectSubmissionSolver \
    paperbench.solver.submissions_dir=$HOME/pb_submissions/ \
    paperbench.solver.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.solver.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime=nanoeval_alcatraz.alcatraz_computer_interface:AlcatrazComputerRuntime \
    paperbench.reproduction.computer_runtime.env=alcatraz.clusters.local:LocalConfig \
    paperbench.reproduction.computer_runtime.env.pull_from_registry=false \
    paperbench.solver.computer_runtime.env.pull_from_registry=false \
    paperbench.judge.completer_config=preparedness_turn_completer.oai_completions_turn_completer:OpenAICompletionsTurnCompleter.Config \
    paperbench.judge.completer_config.model='gpt-4.1-mini' \
    paperbench.judge.code_only=True \
    runner.recorder=nanoeval.json_recorder:json_recorder
```

> `paper_split` 控制评哪些论文(debug/dev/all);submissions 里缺的论文按 0 分记,不会报错。

---

## 五、结果在哪看

每次运行在 `runs/<run_group_id>/` 下生成一组目录,关键文件:

- `<run_id>/grade.json` —— **这篇论文的得分**(按 rubric 树逐项)
- `<run_id>/run.log` / `group.log` —— 运行日志
- `<run_id>/submissions/<时间戳>/submission.tar.gz` —— agent 交的代码包
- `..._grader_output_0.json` —— 裁判逐条打分明细

---

## 六、备忘与坑

1. **数据默认不下载是官方故意的**(`.lfsconfig` 里 `fetchexclude`),我已用官方命令水合 `data/papers/**`;
   以后想要 judge_eval 校准数据:`PAPERBENCH_DATA_DIR=$(pwd)/data uv run python -m paperbench.judge.judge_eval.download_data`
2. 每条命令里的 `pull_from_registry=false` 不能省——否则它会去拉 OpenAI 内部镜像仓库(你没有权限)。
3. 并发默认 5,笔记本建议 `runner.concurrency=1` 或 2。
4. WSL2 下 Docker 磁盘占用会进 vhdx,镜像大约几 GB,不用了可 `docker rmi pb-env pb-reproducer`。
5. DeepCode 是浅克隆(depth=1),要完整历史:`cd DeepCode && git fetch --unshallow`。
6. frontier-evals 是稀疏检出(只有 paperbench + common),要其他项目:`git sparse-checkout add project/<名>`。
