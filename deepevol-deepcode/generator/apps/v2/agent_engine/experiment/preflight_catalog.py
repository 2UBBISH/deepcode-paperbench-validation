"""把 API 侧 preflight 给的 `cloud_options` 变成本模块能选型的 `SkuOption`。

## 为什么必须用 preflight 的目录，而不是自己再查一份

`usage/start` 的 `selected_target` 是靠 `_selected_cloud_option_from_run_metadata`
在 **run metadata 里的 preflight 结果**中反查的（`remote_compute.py:868`）。
自己另查一份目录、拿一个 preflight 里不存在的 target 去开机，
catalog 反查会落空 —— 那时 `image_id` / `cloudmonitor_agent_preinstalled`
这些开机必需的字段全没有，开机要么失败，要么起一台没装监控的机器。

而且 preflight 已经做完了库存交集、DescribePrice、镜像门禁。重做一遍只会分叉。

## 两个会静默出错的字段差异

**一、`vram_gb` 的语义相反。**
preflight 的 `vram_gb` 是**总显存**（`remote_compute.py:1672` 已经乘过
`accelerator_count`），而 `SkuOption.vram_gb` 是**单卡**显存、
`total_vram_gb` 才是乘出来的。直接对接会把 2 卡机的显存算成 4 倍 ——
不报错，只是推荐结果整个错掉。这里显式除回去。

**二、preflight 里没有 `sm`。**
`DescribeInstanceTypes` 不给架构代际，所以 API 侧也没有。
而 `catalog._sm_ge` 对未知 sm **判不满足**（宁可少推荐一个能用的，
也不要推荐一个起不来的）—— 不补这一列，任何设了 `min_compute_capability`
的仓库都会得到「一个候选都没有」。所以在这里按卡型 join
`config/compute/gpu_capability.yaml`。
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

#: vGPU 切片：阿里云的 sgn/vgn 族把一张卡切成小份，卡型名形如 `NVIDIA A10*1/6`。
_VGPU_RE = re.compile(r"^(?P<base>NVIDIA [A-Z0-9]+)\*\d+/(?P<slices>\d+)$")

logger = logging.getLogger(__name__)

_DEFAULT_CAPABILITY_PATH = (
    Path(__file__).resolve().parents[4] / "config" / "compute" / "gpu_capability.yaml"
)


def load_gpu_capability(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """读卡型能力表。读不到就返回空表 —— 调用方会因此判所有卡「架构未知」，
    那是安全的方向（少推荐 > 推荐一个起不来的）。"""
    target = Path(path) if path else _DEFAULT_CAPABILITY_PATH
    try:
        import yaml

        payload = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    gpus = payload.get("gpus")
    return gpus if isinstance(gpus, dict) else {}


def capability_for(gpu_name: str, caps: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """卡型名 → 能力。vGPU 切片按**母卡**查架构。

    ★ 切片卡标 `driver_may_lag`：共享卡的驱动往往落后于独占卡
      （实测一台 A10-4Q 是 470，CUDA 上限 11.4，而 setup loop 装了 cu130 ——
      装得上，但拿不到 GPU）。架构能力不能直接照抄母卡。
    """
    name = (gpu_name or "").strip()
    if name in caps:
        return dict(caps[name])
    match = _VGPU_RE.match(name)
    if match and match.group("base") in caps:
        base = dict(caps[match.group("base")])
        base["vgpu_slices"] = int(match.group("slices"))
        base["driver_may_lag"] = True
        return base
    # 名字里带卡型但表里没有（比如 "NVIDIA A10" 写成 "A10"）时再宽松匹配一次。
    for known, cap in caps.items():
        short = known.replace("NVIDIA ", "").strip()
        if short and re.search(rf"\b{re.escape(short)}\b", name, re.IGNORECASE):
            return dict(cap)
    return {}


def _is_linux_ssh(option: dict[str, Any]) -> bool:
    """这台机器能不能跑我们的实验流程。

    判据取自 preflight 自己的字段：`os_type` 与 `execution_backend`。
    两者缺失时按**不合格**处理 —— 与 `_sm_ge` 对未知 sm 的态度一致：
    宁可少推荐一个能用的，也不要推荐一个起不来的。
    """
    os_type = str(option.get("os_type") or "").strip().lower()
    backend = str(option.get("execution_backend") or "").strip().lower()
    if os_type and os_type != "linux":
        return False
    return backend in {"ssh", "cloud_assistant"}


def catalog_from_preflight(
    cloud_options: list[dict[str, Any]],
    *,
    capability: dict[str, dict[str, Any]] | None = None,
    require_available: bool = True,
    kind: str = "gpu",
) -> list[dict[str, Any]]:
    """preflight 的 `cloud_options` → `catalog.option_from_catalog` 吃的字典。

    `require_available=True` 时只保留 `available_now` 的。那个标志同时含着
    「有货 + 有价 + 镜像装了 CloudMonitor」三件事（`remote_compute.py:1699`），
    少任何一件都开不起来 —— 摆出来只会让用户选中之后失败。

    `kind="cpu"` 收 `cloud_cpu:` 候选。CPU 那边没有 `vram_gb`/`sm`
    （preflight 里恒为 0 / 不存在），选型靠核数与内存 —— 见 `cpu_tiers`。
    """
    caps = capability if capability is not None else load_gpu_capability()
    prefix = "cloud_cpu:" if kind == "cpu" else "cloud_gpu:"
    rows: list[dict[str, Any]] = []
    # ★ 被「不是 Linux/SSH」筛掉的要记下来，别让它们**静默消失** ——
    #   否则用户只看到「没有可用机器」，而真相是「有机器，但都是 Windows」。
    #   这两句话对他意味着完全不同的下一步。
    dropped: list[str] = []
    for option in cloud_options or []:
        target = str(option.get("target") or "")
        if not target.startswith(prefix):
            continue          # 本地档、以及另一类机器，都不参与本次选型
        if require_available and not bool(option.get("available_now")):
            continue
        # ★ 只要 Linux + SSH。RSA 在远端跑 Docker 容器，靠 SSH 驱动 ——
        #   Windows 机器上这套根本起不来。
        #   踩到过：cn-hongkong 的 CPU 候选**全是 Windows 镜像**，
        #   而它们的 `available_now` 是 True（走了 manual_rdp 那条分支）。
        #   不排掉的话，用户会选中一台必然跑不起来的机器，
        #   而且要等到配环境那一步才发现。
        if not _is_linux_ssh(option):
            dropped.append(str(option.get("instance_type") or option.get("target") or ""))
            continue
        count = max(int(option.get("accelerator_count") or 1), 1)
        total_vram = int(option.get("vram_gb") or 0)
        gpu_name = str(option.get("accelerator_type") or "").strip()
        cap = capability_for(gpu_name, caps)
        rows.append({
            "instance_type": str(option.get("instance_type") or ""),
            "accelerator_type": gpu_name,
            "accelerator_count": count,
            # ★ 除回单卡。preflight 那一列是总量，见模块 docstring。
            "vram_gb": total_vram // count if count else total_vram,
            "cpu_cores": int(option.get("cpu_cores") or 0),
            "memory_gb": int(option.get("memory_gb") or 0),
            "hourly_price_cny": float(option.get("provider_hourly_price_cny") or 0),
            "sm": str(cap.get("sm") or ""),
            "gen": str(cap.get("gen") or ""),
            "bf16": cap.get("bf16"),
            "driver_may_lag": bool(cap.get("driver_may_lag")),
            # ★ 带回 target：选中之后要原样交给 usage/start，
            #   否则 API 侧的 catalog 反查落空、开机缺 image_id。
            "target": target,
        })
    if dropped and not rows:
        logger.info("preflight 有 %d 个候选不是 Linux/SSH，全部落选：%s",
                    len(dropped), dropped[:5])
    _LAST_DROPPED[:] = dropped
    return rows


#: 上一次转换里因为「不是 Linux/SSH」被丢掉的候选。
#: 模块级状态不优雅，但这条信息只用于**解释为什么没有候选**，
#: 把它塞进返回值会让所有调用方都得处理一个它们不关心的元组。
_LAST_DROPPED: list[str] = []


def last_dropped_non_linux() -> list[str]:
    return list(_LAST_DROPPED)
