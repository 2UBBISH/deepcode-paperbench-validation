"""实验任务的首轮：从一个仓库链接，到几档摆在用户面前的配置。

## 这一轮**不租机器**

这是整个产品最重要的一条承诺，也是前端提示语里写着的那句
「确认之后才开始租机器」。首轮只做三件不花钱的事：

    浅克隆 → 静态读代码估算力 → 在真的有货的规格里挑几档

用户看到价格再决定。把租赁塞进首轮会让「我只是想看看要多少钱」
变成一笔账单。

## 为什么产物是 markdown 而不是直接开机

选配置是**用户的决定**，不是我们的。给他看清楚三件事：
估出来是多少、每档意味着什么、各要多少钱 —— 然后等他选。

## 清理

克隆出来的目录必须删掉。它只为静态分析存在，远端 RSA 会按同一个
revision 自己再 clone 一次。留着的话，长期跑下来盘会被临时目录塞满，
而那种问题要到磁盘写满时才会暴露。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .compute_spec import (
    DEFAULT_HOST_RAM_FLOOR_GIB,
    ComputeSpec,
    build_compute_spec,
)
from .recommend import instruction_pins_cpu, Recommendation, recommend_from_repo
from .repo_facts import analyse_resources
from .repo_source import clone_for_analysis
from .zip_source import extract_zip_repo
from apps.common.repo_url import RepoSourceError, parse_repo_url

logger = logging.getLogger(__name__)


@dataclass
class ExperimentTurnResult:
    """首轮的产物。`ok=False` 时 `markdown` 里是给用户看的失败原因。"""

    ok: bool
    markdown: str
    data: dict[str, Any] = field(default_factory=dict)


def _fmt_price(cny: float) -> str:
    return f"{cny:.2f} 元/时"


def render_recommendation(rec: Recommendation, *, repo_slug: str, revision: str) -> str:
    """把推荐渲染成用户能据以决策的 markdown。

    ★ 必须出现的三件事：估计**是区间不是精确值**、每档意味着什么、
      以及各要多少钱。少了任何一件，用户就只能盲选。
    """
    spec = rec.spec
    lines: list[str] = []
    lines.append(f"## 代码读完了：`{repo_slug}`")
    if revision:
        lines.append(f"\n分析的是 `{revision[:8]}` 这个版本。远端会用同一个版本跑，"
                     "所以不会出现「我看的和它跑的不是一份代码」。\n")

    workload_cn = {"training": "训练", "finetuning": "微调",
                   "inference": "推理", "unknown": "看不出是训练还是推理"}
    lines.append(f"- **任务类型**：{workload_cn.get(spec.workload, spec.workload)}")
    if spec.params_b:
        lines.append(f"- **模型规模**：约 {spec.params_b:.2f}B 参数（{spec.params_confidence}）")
    if spec.needs_gpu and spec.vram:
        lines.append(
            f"- **显存需求**：{spec.vram.low:.1f} – {spec.vram.high:.1f} GiB"
            f"（{'置信度 ' + spec.confidence}）"
        )
    elif not spec.needs_gpu:
        # ★ CPU 仓库不该看到「显存需求」—— 那个数字对它毫无意义，
        #   而且会让用户以为我们判错了、其实需要 GPU。
        if any("指定用 CPU" in note for note in (getattr(spec, "notes", None) or [])):
            lines.append("- **按 CPU 选型**：任务说明明确指定了 CPU（代码本身能用 GPU）")
        else:
            lines.append("- **不需要 GPU**：没有检测到把张量搬到显卡上的代码")
    if spec.min_compute_capability:
        lines.append(
            f"- **架构下限**：sm_{spec.min_compute_capability} —— 这是硬门槛，"
            "低于它的卡不是慢，是跑不起来"
        )
    if spec.unknowns:
        lines.append(f"- **没读出来的**：{'、'.join(spec.unknowns[:6])}")

    if rec.blocked_reason:
        lines.append(f"\n### 没能给出配置\n\n{rec.blocked_reason}\n")
        return "\n".join(lines)

    if rec.gpu_fallback_reason:
        lines.append(
            f"\n> ⚠️ 这份代码本来想用 GPU，但{rec.gpu_fallback_reason}。"
            "下面给的是 **CPU 机器**：能配环境、能跑通小规模验证，"
            "正经训练会慢很多甚至跑不动——要不要先这样试一轮，你定。\n"
        )
    for note in getattr(spec, "notes", None) or []:
        if "指定用 CPU" not in note:
            lines.append(f"- {note}")

    lines.append("\n### 几档配置\n")
    lines.append("| 档位 | 机器 | 显存 | 价格 | 这档意味着什么 |")
    lines.append("| --- | --- | --- | --- | --- |")
    tier_by_key = {t.key: t for t in rec.plan.tiers}
    for key, opt in rec.choices:
        tier = tier_by_key.get(key)
        cards = f"{opt.accelerator_count}× " if opt.accelerator_count > 1 else ""
        lines.append(
            f"| **{tier.label if tier else key}** | {cards}{opt.accelerator_type} "
            f"({opt.instance_type}) | {opt.total_vram_gb}G | {_fmt_price(opt.hourly_price_cny)} "
            f"| {tier.blurb if tier else ''} |"
        )

    if rec.collapsed_tiers:
        lines.append(
            "\n> 有几档落在了同一台机器上 —— 目录里没有更小的规格，"
            "所以最便宜的那台已经带着更高档的余量。这不是少给了你选择"
        )
    for note in rec.plan.notes:
        lines.append(f"\n> {note}")

    lines.append(
        "\n**这些数字是估出来的，不是实测。** 真跑起来如果显存不够，"
        "我会带着实测数字回来问你要不要升级，"
        "并且尽量不让你把环境再配一遍。"
    )
    lines.append("\n选一档告诉我，我再开始租机器。**在你确认之前不会产生任何费用。**")
    return "\n".join(lines)


def _requirements_from_spec(spec: ComputeSpec) -> dict[str, Any]:
    """把静态分析的结论翻成 preflight 认的 `resource_requirements`。

    ★ 不传的话 preflight 会拿任务文本去猜 —— 而文本里通常什么线索都没有。
      「把测试跑通」既不说 GPU 也不说规模，猜出来的结果不可靠，
      却决定了目录里有没有我们要的那类机器。
    """
    return {
        "gpu_recommended": bool(spec.needs_gpu),
        "cpu_recommended": not bool(spec.needs_gpu),
        "min_cpu_cores": 2,
        # 内存下限只在**有证据**时才提 —— 参数量未知时 host_ram_gib 恒等于
        # 下限值、不携带信息，拿它去卡目录会把便宜机器全筛掉。
        **({"min_memory_gb": int(spec.host_ram_gib)}
           if spec.host_ram_gib > DEFAULT_HOST_RAM_FLOOR_GIB else {}),
        # ★ **不传 min_storage_gb。**
        #   系统盘容量是开机时我们自己设的（`settings.system_disk_size_gb`），
        #   不是实例规格的属性 —— 目录里每个规格都报同一个数。
        #   拿它去筛规格没有意义，只会全军覆没：实测 cn-hongkong 的
        #   CPU 规格全报 40G，传 60 就是 0 个候选，而真正该做的是
        #   开机时把盘开大一点。
        #   （这与 host_ram_gib 是同一类错误：把保守下限当硬门槛。）
        **({"min_vram_gb": int(spec.vram.high)} if spec.needs_gpu and spec.vram else {}),
    }


async def run_experiment_first_turn(
    context: dict[str, Any],
    *,
    fetch_cloud_options,
) -> ExperimentTurnResult:
    """实验会话首轮。**不租机器、不产生费用。**

    `fetch_cloud_options(requirements)` 是一个 async 可调用，返回 API 侧
    preflight 的 `cloud_options`。注入而不是在这里直接发 HTTP，是为了让这条
    流程能在没有 API 的情况下被测到 —— 它是整个产品的第一印象，
    不该只能靠真机验证。
    """
    options = context.get("experiment_options")
    options = options if isinstance(options, dict) else {}
    raw_url = str(options.get("repo_url") or "").strip()
    zip_path = str(options.get("repo_zip_local_path") or "").strip()

    if not raw_url and not zip_path:
        return ExperimentTurnResult(
            False,
            "### 还缺代码\n\n给我一个 Git 仓库链接，或者上传一个压缩包。",
        )

    cloned = None
    try:
        try:
            if zip_path:
                # ★ 压缩包路径**不需要 clone** —— 文件已经在本地了，
                #   解压出来直接读。首轮比 Git 路径还快一步。
                cloned = extract_zip_repo(
                    zip_path,
                    dest=Path(zip_path).parent / "extracted",
                    name=str(options.get("repo_zip_name") or "uploaded"),
                )
            else:
                source = parse_repo_url(raw_url)
                cloned = clone_for_analysis(
                    source, revision=str(options.get("revision") or "").strip()
                )
        except RepoSourceError as exc:
            return ExperimentTurnResult(False, f"### 拿不到代码\n\n{exc}")

        # ★ **先分析、再要目录**，顺序不能反。
        #   API 侧的 preflight 在没收到 `resource_requirements` 时，会用
        #   `estimate_required_resources(run.prompt, ...)` 从**任务文本**猜需求 ——
        #   而我们此刻手上已经有静态分析算出的真需求了。
        #   实测踩到：一个纯 CPU 的仓库，prompt 里那句「把测试跑通」被猜成需要 GPU，
        #   于是目录里一个 CPU 规格都没有，推荐直接落空。
        entry_first = str(options.get("entry") or "").strip()
        instruction = str(context.get("prompt") or options.get("instruction") or "")
        facts = analyse_resources(cloned.local_path, entry_first=entry_first)
        spec = build_compute_spec(facts)
        if spec.needs_gpu and instruction_pins_cpu(instruction):
            # The catalog request must match the selection below: a user who
            # asked for CPU must not have the GPU question decide the catalog.
            spec.needs_gpu = False
        try:
            cloud_options = await fetch_cloud_options(_requirements_from_spec(spec))
        except Exception as exc:
            logger.warning("实验首轮取 preflight 失败: %s", exc)
            return ExperimentTurnResult(
                False,
                "### 暂时查不到可用的机器\n\n"
                "代码已经读完了，但机器目录取不到，没法给你报价。请稍后再试。",
            )

        rec = recommend_from_repo(
            cloned.local_path,
            cloud_options=cloud_options,
            entry_first=entry_first,
            instruction=instruction,
        )
        markdown = render_recommendation(
            rec, repo_slug=cloned.slug, revision=cloned.revision
        )
        return ExperimentTurnResult(
            ok=not rec.blocked_reason,
            markdown=markdown,
            data={
                "repo": {
                    **cloned.source.to_dict(),
                    "revision": cloned.revision,
                    # 压缩包路径要靠它在「跑」那一轮找回代码。
                    **({"local_path": str(cloned.local_path)} if zip_path else {}),
                },
                "recommendation": rec.to_dict(),
                # ★ 这一条一路要传到 OOM 升级卡：仓库支不支持断点续跑决定了
                #   「换机后要不要从头训」。以前算出来就扔了，于是卡片上那句
                #   「没有检测到断点续跑」**永远**显示 —— 而它自己的注释写着
                #   「永远显示的警告等于没有警告」。
                "resumable": bool(spec.resumable),
                # 选中之后要原样交给 usage/start —— 见 preflight_catalog 的说明。
                "choices": [
                    {"tier": key, "target": opt.target, **opt.to_dict()}
                    for key, opt in rec.choices
                ],
            },
        )
    finally:
        # ★ **Git 路径**的克隆目录只为静态分析存在，远端会按同一 revision
        #   自己再 clone 一次，所以用完就删 —— 不删的话盘会被临时目录慢慢
        #   塞满，而那种问题要到写满时才暴露。
        #
        #   **压缩包路径不能删**：那份代码只存在于这里，跑的时候还要拿它
        #   打 bundle 送到机器上。清理由会话工作区的生命周期负责。
        if cloned is not None and not zip_path:
            cloned.cleanup()
