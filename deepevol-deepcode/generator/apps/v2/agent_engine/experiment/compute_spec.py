"""把静态事实折成算力需求区间（ComputeSpec）。

## 为什么产出的是区间不是点

静态分析对标准 HF 微调仓库能读得很准，对自定义模型 / 3D / 视频 / GNN 可以差
**2–5 倍**。给一个点估计等于假装这个差别不存在。区间 + 置信度才是诚实的表达，
而且下游的档位生成本来就要按分位数取值（见 `tiers.py`）。

## 三个必须做对的量

1. **Adam 混合精度是 16 bytes/param，不是 8。**
   fp32 master weights(4) + momentum(4) + variance(4) = 12，再加 fp32 梯度累积(4)。
   按 8 算会把 7B 模型的优化器状态少算 56 GB。

2. **KV cache 用 `n_kv_heads` 不是 `n_heads`。**
   GQA/MQA 下两者差一个数量级（Llama-3-70B 是 64 vs 8）。用错直接高估 8 倍。

3. **allocator 要的是 reserved 不是 allocated。**
   PyTorch 缓存分配器向驱动申请的是 segment，碎片通常 5–15%。
   再加每进程每卡 0.3–1.0 GiB 的 CUDA context——这两项加起来不小。

## 精度要求其实很松

输出空间是**离散 SKU 档位**（阿里云显存按 16/24/32/48/80/96 GiB 跳），
所以要的不是"百分之几的误差"，而是"别跨档"。而且误差强不对称：
低估 → OOM，损失是环境配置时间 + 已耗机时 + 一轮失败实验；
高估 → 多花钱，线性且有界。所以系统性往上偏是正确策略，不是妥协。
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

from .oom import OomFact, required_vram_gib
from .repo_facts import ResourceFacts

GIB = 1024 ** 3

#: 每进程每卡的 CUDA context。随驱动版本变，0.3–1.0 GiB，取中间偏保守。
CUDA_CONTEXT_GIB = 0.8

#: 缓存分配器的碎片开销。向驱动申请的是 segment，不是张量实际占用。
ALLOCATOR_OVERHEAD = 0.12

#: 优化器状态的每参数字节数。**Adam 系混合精度是 16 不是 8**——
#: fp32 master(4) + m(4) + v(4) + fp32 grad(4)。
_OPTIMIZER_BYTES = {
    "adamw": 16, "adam": 16, "paged_adamw": 16, "adamw8bit": 6, "paged_adamw8bit": 6,
    "lion": 8, "adafactor": 4, "sgd": 4, "adagrad": 8, "rmsprop": 8,
}
_PRECISION_BYTES = {"fp32": 4, "tf32": 4, "fp16": 2, "bf16": 2}
_QUANT_BYTES = {"4bit": 0.5, "8bit": 1, "bitsandbytes": 1}

#: 名字里的规模后缀 → 参数量（十亿）。只在拿不到 config.json 时用，标 estimated。
_SIZE_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?)\s*([bBmM])(?![A-Za-z0-9])")


#: 主机内存的**恒定下限**。参数量未知时 `host_ram_gib` 就等于它 ——
#: 也就是说等于这个值时它**不携带任何信息**，不该被当成硬门槛用来筛机器。
#: 见 `recommend._enforceable_host_ram` 里记的那次踩坑。
DEFAULT_HOST_RAM_FLOOR_GIB = 16.0


@dataclass
class Estimate:
    """一个带区间与出处的估计。`low`/`high` 是 GiB。"""

    low: float
    high: float
    breakdown: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def point(self) -> float:
        """点估计取区间上沿偏保守的位置——低估的代价远大于高估。"""
        return self.low + (self.high - self.low) * 0.6

    def to_dict(self) -> dict[str, Any]:
        return {
            "low_gib": round(self.low, 1),
            "high_gib": round(self.high, 1),
            "point_gib": round(self.point, 1),
            "breakdown": {k: round(v, 2) for k, v in self.breakdown.items()},
            "notes": self.notes,
        }


@dataclass
class ComputeSpec:
    """一份仓库的算力需求。**所有数字都带区间，别当精确值用。**"""

    workload: str = "unknown"            # training | finetuning | inference | unknown
    params_b: float | None = None
    params_confidence: str = "unknown"   # exact | estimated | unknown
    vram: Estimate | None = None
    host_ram_gib: float = DEFAULT_HOST_RAM_FLOOR_GIB
    storage_gib: int = 60
    min_compute_capability: str = ""
    needs_gpu: bool = False
    multi_gpu: bool = False
    parallel: list[str] = field(default_factory=list)
    resumable: bool = False
    #: high | medium | low —— 直接决定档位数量与区间宽度
    confidence: str = "medium"
    #: 显存需求是否来自**实测**（一次 CUDA OOM），而不是解析式估算。
    #: 为 True 时 `vram.low` 是「已知至少要这么多」的硬下界——
    #: 任何一档都不许低于它，否则等于给用户一个已知会炸的选项。
    vram_measured: bool = False
    dominant_term: str = ""
    evidence: dict[str, str] = field(default_factory=dict)
    unknowns: list[str] = field(default_factory=list)
    #: 选型时附带给用户的说明（如「任务说明指定了 CPU」）。
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload": self.workload,
            "params_b": self.params_b,
            "params_confidence": self.params_confidence,
            "vram": self.vram.to_dict() if self.vram else None,
            "host_ram_gib": self.host_ram_gib,
            "storage_gib": self.storage_gib,
            "min_compute_capability": self.min_compute_capability,
            "needs_gpu": self.needs_gpu,
            "multi_gpu": self.multi_gpu,
            "parallel": self.parallel,
            "resumable": self.resumable,
            "confidence": self.confidence,
            "vram_measured": self.vram_measured,
            "dominant_term": self.dominant_term,
            "evidence": self.evidence,
            "unknowns": self.unknowns,
            "notes": self.notes,
        }


# -- 参数量 ------------------------------------------------------------------

def params_from_config(config: dict[str, Any]) -> float | None:
    """从 HF `config.json` 精确算参数量（十亿）。

    GQA 下 k/v 投影比 q 小 `n_heads/n_kv_heads` 倍——这一步不能省，
    否则 Llama-3-70B 这类模型会高估几个 B。
    """
    h = config.get("hidden_size") or config.get("n_embd") or config.get("d_model")
    layers = (config.get("num_hidden_layers") or config.get("n_layer")
              or config.get("num_layers"))
    if not h or not layers:
        return None
    h, layers = int(h), int(layers)
    vocab = int(config.get("vocab_size") or 32000)
    n_heads = int(config.get("num_attention_heads") or config.get("n_head") or max(h // 64, 1))
    n_kv = int(config.get("num_key_value_heads") or n_heads)
    d_head = h // max(n_heads, 1)
    ffn = int(config.get("intermediate_size") or config.get("n_inner") or 4 * h)

    # q + o 是满的，k + v 按 kv 头数缩
    attn = h * h + h * h + 2 * (h * n_kv * d_head)
    # 门控 MLP（SwiGLU）三个矩阵，普通 MLP 两个
    gated = "swiglu" in str(config.get("hidden_act", "")).lower() or config.get("intermediate_size")
    mlp = (3 if gated else 2) * h * ffn
    per_layer = attn + mlp + 2 * h  # 两个 LayerNorm
    embed = vocab * h * (1 if config.get("tie_word_embeddings") else 2)
    return (layers * per_layer + embed) / 1e9


def params_from_name(model_id: str) -> float | None:
    """从模型名里认规模后缀（`Qwen2.5-7B` → 7.0）。拿不到 config 时的退路。"""
    best: float | None = None
    for num, unit in _SIZE_RE.findall(model_id or ""):
        value = float(num) / (1000 if unit in "mM" else 1)
        # 名字里可能有版本号（Qwen2.5），取最大的那个当规模
        if best is None or value > best:
            best = value
    # 版本号误判防护：小于 0.05B 的多半不是规模
    return best if best and best >= 0.05 else None


# -- 显存 --------------------------------------------------------------------

def _activation_gib(*, layers: int, seq: int, batch: int, hidden: int, heads: int,
                    dtype_bytes: int, grad_ckpt: bool) -> float:
    """Transformer 每层激活（Korthikanti 等）：s·b·h·(34 + 5·a·s/h) bytes。

    第二项是注意力矩阵，**随 seq² 增长**——长序列下它会盖过一切，
    这也是 `seq_len` 读不出来时区间必然很宽的原因。
    """
    per_layer = seq * batch * hidden * (34 + 5 * heads * seq / max(hidden, 1))
    per_layer *= dtype_bytes / 2  # 公式按 fp16 给，按实际精度缩放
    total = per_layer * layers
    if grad_ckpt:
        # 全量重计算只留每层边界，峰值约降到 20–30%
        total *= 0.25
    return total / GIB


def estimate_vram(
    facts: ResourceFacts,
    *,
    params_b: float | None,
    config: dict[str, Any] | None = None,
    workload: str = "training",
    needs_gpu_prior: bool = False,
) -> Estimate:
    """按解析式给显存区间。缺的量用**保守假设 + 加宽区间**处理，不猜具体值。"""
    notes: list[str] = []
    breakdown: dict[str, float] = {}

    quant = facts.quantization.value if facts.quantization else None
    precision = facts.precision.value if facts.precision else None
    if precision is None:
        # 读不出精度按 fp32 估上界、bf16 估下界——现代训练多数是混合精度，
        # 但硬编成 bf16 会在老仓库上低估一半。
        notes.append("精度未读出：下界按 bf16、上界按 fp32")
    weight_bytes_low = _QUANT_BYTES.get(quant or "", _PRECISION_BYTES.get(precision or "bf16", 2))
    weight_bytes_high = _QUANT_BYTES.get(quant or "", _PRECISION_BYTES.get(precision or "fp32", 4))

    p = (params_b or 0) * 1e9
    params_low = p * weight_bytes_low / GIB
    params_high = p * weight_bytes_high / GIB
    breakdown["params"] = params_high

    grads_low = grads_high = 0.0
    optim_low = optim_high = 0.0
    if workload in {"training", "finetuning"} and p:
        grads_low, grads_high = params_low, params_high
        opt_name = (facts.optimizer.value if facts.optimizer else "adamw").lower()
        opt_bytes = _OPTIMIZER_BYTES.get(opt_name, 16)
        if quant:
            # 量化训练通常只训 LoRA 之类的适配器，优化器状态按 5% 参数计
            optim_low = optim_high = p * 0.05 * opt_bytes / GIB
            notes.append(f"检测到 {quant} 量化，优化器状态按 5% 可训参数估")
        else:
            optim_low = optim_high = p * opt_bytes / GIB
        breakdown["grads"] = grads_high
        breakdown["optimizer"] = optim_high

    # 激活
    act_low = act_high = 0.0
    cfg = config or {}
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or max(hidden // 64, 1))
    batch = facts.batch_size.value if facts.batch_size else None
    seq = facts.seq_len.value if facts.seq_len else None
    grad_ckpt = bool(facts.grad_checkpointing and facts.grad_checkpointing.value)

    if hidden and layers and batch and seq:
        act_low = _activation_gib(layers=layers, seq=seq, batch=batch, hidden=hidden,
                                  heads=heads, dtype_bytes=weight_bytes_low,
                                  grad_ckpt=grad_ckpt or True)
        act_high = _activation_gib(layers=layers, seq=seq, batch=batch, hidden=hidden,
                                   heads=heads, dtype_bytes=weight_bytes_high,
                                   grad_ckpt=grad_ckpt)
        breakdown["activations"] = act_high
    else:
        missing = [n for n, v in (("hidden/layers", hidden and layers),
                                  ("batch_size", batch), ("seq_len", seq)) if not v]
        notes.append("激活无法解析（缺 " + "、".join(missing) + "），按参数量倍数兜底")
        act_low = params_low * 0.3
        act_high = max(params_high * 2.0, 4.0)
        breakdown["activations"] = act_high

    # ★ 兜底方向必须往上，不能往下。
    #
    # 十个真实仓库的验收里，8 个判成 UNDER（会 OOM），根因就在这：参数量读不出来时
    # 上界只有 4 GiB，稳妥档落到 8G，而基准普遍是 16–40G。
    #
    # 这与设计原则正好相反——误差是**强不对称**的：低估 → OOM，损失是环境配置时间
    # + 已耗机时 + 一轮失败实验；高估 → 多花钱，线性且有界。所以"不知道"时
    # 该给的是一个**保守的先验**，不是一个乐观的小数。
    #
    # 先验取 24 GiB：单卡研究场景最常见的显存档（A10/L4/3090/4090 都是 24G），
    # 也覆盖了验收基准里的多数。这是**先验不是计算**，notes 里说清楚，
    # 且 confidence 会因此降为 low → 自动带出保险档兜住长尾。
    if needs_gpu_prior and params_b is None:
        prior_high = 24.0
        if act_high + params_high < prior_high:
            notes.append(
                f"参数量未知，上界改用 {prior_high:.0f}G 的保守先验"
                "（单卡研究场景最常见档位）——这是先验不是计算，请以保险档兜底"
            )
            act_high = prior_high - params_high
            breakdown["activations"] = act_high

    if workload == "inference" and p:
        kv = _kv_cache_gib(cfg, batch or 1, seq or 2048, weight_bytes_low)
        if kv:
            breakdown["kv_cache"] = kv
            act_low += kv
            act_high += kv

    low = (params_low + grads_low + optim_low + act_low + CUDA_CONTEXT_GIB)
    high = (params_high + grads_high + optim_high + act_high + CUDA_CONTEXT_GIB)
    breakdown["cuda_context"] = CUDA_CONTEXT_GIB
    low *= 1 + ALLOCATOR_OVERHEAD
    high *= 1 + ALLOCATOR_OVERHEAD
    notes.append(f"含 {int(ALLOCATOR_OVERHEAD*100)}% 分配器碎片与 {CUDA_CONTEXT_GIB}G CUDA context")

    return Estimate(low=max(low, 1.0), high=max(high, low, 2.0),
                    breakdown=breakdown, notes=notes)


def _kv_cache_gib(cfg: dict[str, Any], batch: int, seq: int, dtype_bytes: int) -> float:
    """KV cache = 2 · L · n_kv_heads · d_head · seq · batch · bytes。

    **是 `n_kv_heads` 不是 `n_heads`**——GQA/MQA 下差一个数量级。
    """
    layers = int(cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0)
    hidden = int(cfg.get("hidden_size") or cfg.get("n_embd") or 0)
    if not layers or not hidden:
        return 0.0
    n_heads = int(cfg.get("num_attention_heads") or cfg.get("n_head") or max(hidden // 64, 1))
    n_kv = int(cfg.get("num_key_value_heads") or n_heads)
    d_head = hidden // max(n_heads, 1)
    return 2 * layers * n_kv * d_head * seq * batch * dtype_bytes / GIB


# -- 主入口 ------------------------------------------------------------------

#: 这些目录里的模型只是示例/评测用，不代表仓库的主要负载。
_EXAMPLE_DIRS = ("examples/", "example/", "evals/", "eval/", "benchmarks/", "benchmark/",
                 "scripts/", "demo/", "notebooks/")


def _is_example_path(evidence: str) -> bool:
    path = evidence.rsplit(":", 1)[0].replace("\\", "/")
    return any(seg in path for seg in _EXAMPLE_DIRS)


def build_compute_spec(
    facts: ResourceFacts,
    *,
    model_config: dict[str, Any] | None = None,
    dataset_gib: int = 0,
) -> ComputeSpec:
    spec = ComputeSpec()
    spec.unknowns = list(facts.unknowns)

    # 工作负载
    if facts.optimizer or facts.grad_checkpointing:
        spec.workload = "finetuning" if facts.quantization else "training"
    elif facts.model_ids:
        spec.workload = "inference"

    # 参数量：config.json 是精确值，名字只是估计
    if model_config:
        exact = params_from_config(model_config)
        if exact:
            spec.params_b, spec.params_confidence = round(exact, 2), "exact"
    if spec.params_b is None:
        sizes: list[tuple[float, Any]] = []
        for fact in facts.model_ids:
            guessed = params_from_name(str(fact.value))
            if guessed:
                sizes.append((guessed, fact))

        # ★ 框架型仓库没有单一答案，挑一个是武断的。
        #
        # 验收里三个失败都是这个形状：litgpt 支持 7B–70B（README 里举了 3B 的例子）、
        # peft 有几十个示例（扫到 opt-350m）、mamba 扫到的 gpt-neox-20b 其实是
        # **评测脚本的 tokenizer**。三个提取都没读错，错的是"拿其中一个当代表"。
        #
        # 正确行为不是猜，是说清楚需要用户指定跑哪个配置——这与 RSA 自己的
        # needs_user 终态是同一个语义。这里标 ambiguous 让置信度落到 low，
        # 显存改用保守先验并带出保险档。
        distinct = {round(s, 1) for s, _ in sizes}
        only_from_examples = bool(facts.model_ids) and all(
            _is_example_path(f.evidence) for f in facts.model_ids
        )
        if len(distinct) > 1 or (sizes and only_from_examples):
            spec.params_confidence = "ambiguous"
            where = "、".join(sorted({f.evidence.rsplit(":", 1)[0] for _, f in sizes})[:3])
            spec.unknowns.append(
                f"这个仓库像是支持多种模型/配置（{where} 里出现了 "
                f"{sorted(distinct)} B 等不同规模）——请指定要跑哪个配置，"
                "否则只能按保守先验估"
            )
        elif sizes:
            spec.params_b, spec.params_confidence = sizes[0][0], "estimated"
            spec.evidence["params_b"] = sizes[0][1].evidence

    spec.needs_gpu = bool(facts.uses_gpu or facts.model_ids or facts.parallel)
    spec.parallel = [f.value for f in facts.parallel]
    spec.multi_gpu = any(p in {"ddp", "fsdp", "deepspeed", "accelerate", "torch_distributed"}
                         for p in spec.parallel)
    spec.resumable = bool(facts.resumable and facts.resumable.value)
    if facts.min_compute_capability:
        spec.min_compute_capability = str(facts.min_compute_capability.value)
        spec.evidence["min_compute_capability"] = facts.min_compute_capability.evidence

    spec.vram = estimate_vram(facts, params_b=spec.params_b, config=model_config,
                              workload=spec.workload, needs_gpu_prior=spec.needs_gpu)
    spec.dominant_term = max(spec.vram.breakdown, key=lambda k: spec.vram.breakdown[k],
                             default="")

    # 主机内存：DataLoader worker 是结构性乘数（COW 会被 refcount 写坏，实际就是复制）
    workers = facts.num_workers.value if facts.num_workers else 0
    base_ram = max(DEFAULT_HOST_RAM_FLOOR_GIB, (spec.params_b or 1) * 4)
    spec.host_ram_gib = math.ceil(base_ram * (1 + 0.25 * min(int(workers or 0), 8)))

    # 存储：数据集 + 权重 + checkpoint（按权重 3 份留）
    weights = (spec.params_b or 0) * 2
    spec.storage_gib = int(max(60, dataset_gib + weights * 4 + 20))

    spec.confidence = _confidence(spec, facts)
    return spec


def _confidence(spec: ComputeSpec, facts: ResourceFacts) -> str:
    """置信度决定档位数量与区间宽度，所以宁可判低不判高。"""
    if not facts.coherent:
        return "low"
    if spec.params_confidence in {"unknown", "ambiguous"}:
        return "low"
    span = (spec.vram.high / max(spec.vram.low, 0.1)) if spec.vram else 1
    critical_missing = sum(
        1 for name in ("batch_size", "seq_len", "precision")
        if getattr(facts, name) is None
    )
    if span > 3 or critical_missing >= 2:
        return "low"
    if span > 1.8 or critical_missing == 1 or spec.params_confidence == "estimated":
        return "medium"
    return "high"


# -- 实测覆盖 ----------------------------------------------------------------

def apply_oom_measurement(spec: ComputeSpec, oom: OomFact) -> ComputeSpec:
    """用一次真实 OOM 覆盖静态估计。**原地修改并返回同一个 spec。**

    实测值压过静态估算——这是整个 P4 的意义所在：静态分析的天花板在 5/10，
    而一次 OOM 就把「需要多少」从推测变成了观测。

    但它仍然只是**下界**：OOM 只说明「那一刻还差这么多」，不代表全程峰值
    （eval 阶段的大 batch、更长的序列都可能更高）。所以：

    - `vram.low` 设成实测需求（硬下界，档位不许低于它）
    - `vram.high` 留出余量，且 `confidence` **不因为"测过了"就升级**——
      测过的是下界，不是上界
    """
    need = required_vram_gib(oom)
    est = spec.vram or Estimate(low=need, high=need * 1.5)

    est.low = max(need, est.low)
    est.high = max(est.high, need * 1.4)
    est.breakdown["measured_floor"] = round(need, 2)
    est.notes.append(
        f"实测：在 {oom.total_capacity_gib:.1f}G 的卡上 OOM"
        f"（已占用 {oom.in_use_gib:.1f}G，还想再要 {oom.tried_to_allocate_gib:.2f}G）"
        f"→ 至少需要 {need:.1f}G。**这是下界不是精确值**，"
        "后续阶段可能有更大的峰值。"
    )
    spec.vram = est
    spec.vram_measured = True
    spec.dominant_term = "measured"
    spec.evidence["vram_measured"] = oom.excerpt or "cuda_oom"
    return spec
