"""从代码里读出算力相关的事实——不跑代码，也不问模型。

## 为什么不改 vendored 的 static_facts.py

设计文档原写「扩 RSA 的 static_facts」，落地时改成本模块，因为 vendor 纪律是
「改动集中在新增文件、尽量不改既有函数体，保住可 diff 性」
（见 `Agent/rsa/DEEPEVOL_VENDOR.md`）。这里 **import 它的 `analyse()` 复用成果**
（entrypoints/flags、requirements、config_files、readme_commands 都直接拿来用），
只对它没覆盖的资源维度自己走一趟 AST。vendored 树一个字没改。

## 一条不能破的规矩

**每个事实必须带证据行；读不出来就是 `None`，绝不填猜测值。**

这不是洁癖。显存的大头是激活，而激活 ∝ batch × seq × 分辨率——一旦这里替模型
编一个 `batch_size=32`，下游的区间、档位、选型会一路错到底，而且**错得毫无征兆**：
输出看起来一样自信。读不出来时把它标成 unknown，让区间变宽、让保险档出现，
才是对的行为。

`unknowns` 里的每一项都会在 UI 上明示，也是 `confidence` 降级的依据。
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------

#: 取值来源的优先级。数字越小越可信，只有**更可信的**才能覆盖已有值。
#: 分层是必须的：把 bench.py 的 batch 和 config/finetune_x.py 的 grad_accum
#: 拼在一起，得到的是一个**现实中不存在的配置**，估出来的数没有意义。
TIER_CLI = 0        # 命令行实际传入 / argparse 最终生效值
TIER_ENTRY = 1      # 入口脚本自身
TIER_CONFIG = 2     # 入口引用的配置文件
TIER_ELSEWHERE = 3  # 仓库里其它地方 —— 只在没有更好来源时才用


@dataclass(frozen=True)
class Fact:
    """一个读出来的值 + 它的出处。`evidence` 形如 `train.py:42`。"""

    value: Any
    evidence: str
    tier: int = TIER_ELSEWHERE

    @property
    def source_file(self) -> str:
        return self.evidence.rsplit(":", 1)[0]

    def __repr__(self) -> str:  # 让日志里一眼能看到出处
        return f"Fact({self.value!r} @{self.evidence})"


@dataclass
class ResourceFacts:
    """算力相关的静态事实。字段为 None 表示**没读出来**，不是"没有"。"""

    model_ids: list[Fact] = field(default_factory=list)
    precision: Fact | None = None
    quantization: Fact | None = None
    optimizer: Fact | None = None
    grad_checkpointing: Fact | None = None
    batch_size: Fact | None = None
    grad_accum: Fact | None = None
    seq_len: Fact | None = None
    image_size: Fact | None = None
    num_workers: Fact | None = None
    pin_memory: Fact | None = None
    parallel: list[Fact] = field(default_factory=list)
    resumable: Fact | None = None
    #: 由 flash-attn / bf16 / fp8 等用法反推的架构下限，如 "8.0"
    min_compute_capability: Fact | None = None
    torch_cuda: Fact | None = None
    uses_gpu: Fact | None = None
    #: 读不出来但对估算重要的字段，驱动区间变宽与保险档
    unknowns: list[str] = field(default_factory=list)
    #: 关键超参是否来自**同一份配置**。为 False 说明它们被从不同文件拼起来了，
    #: 那是一个现实中可能不存在的组合，估算必须按低置信处理。
    coherent: bool = True
    #: 关键超参各自的来源文件，用于让上面那条可审计
    hyperparam_sources: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, value in self.__dict__.items():
            if isinstance(value, Fact):
                out[name] = {"value": value.value, "evidence": value.evidence}
            elif isinstance(value, list) and value and isinstance(value[0], Fact):
                out[name] = [{"value": f.value, "evidence": f.evidence} for f in value]
            else:
                out[name] = value
        return out


# -- 识别表 ------------------------------------------------------------------

_PRECISION_CALLS = {
    "half": "fp16",
    "bfloat16": "bf16",
    "float16": "fp16",
}
_DTYPE_TOKENS = {
    "torch.bfloat16": "bf16",
    "torch.float16": "fp16",
    "torch.half": "fp16",
    "torch.float32": "fp32",
    "bfloat16": "bf16",
    "float16": "fp16",
}
_OPTIMIZER_HINTS = (
    "AdamW", "Adam", "SGD", "Adafactor", "Adagrad", "RMSprop", "Lion",
    "AdamW8bit", "PagedAdamW", "PagedAdamW8bit",
)
_PARALLEL_IMPORTS = {
    "deepspeed": "deepspeed",
    "accelerate": "accelerate",
    "fairscale": "fairscale",
}
_SEQ_KEYS = ("max_seq_len", "max_seq_length", "max_length", "block_size",
             "seq_len", "sequence_length", "n_ctx", "context_length")
_BATCH_KEYS = ("batch_size", "per_device_train_batch_size", "train_batch_size",
               "micro_batch_size")
_ACCUM_KEYS = ("gradient_accumulation_steps", "grad_accum", "accumulate_grad_batches")
_IMAGE_KEYS = ("image_size", "img_size", "resolution", "crop_size", "input_size")


#: 这些目录里的值不是真实负载：测试夹具刻意用小模型，示例往往是最小可跑配置。
_FIXTURE_DIRS = ("tests/", "test/", "testing/")
_FIXTURE_PREFIXES = ("test_", "conftest")


def _is_fixture_path(rel: str) -> bool:
    norm = rel.replace("\\", "/")
    if any(seg in norm for seg in _FIXTURE_DIRS):
        return True
    return pathlib_name(norm).startswith(_FIXTURE_PREFIXES)


def pathlib_name(rel: str) -> str:
    return rel.rsplit("/", 1)[-1]


def _first_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        m = re.fullmatch(r"\s*(\d{1,7})\s*", value)
        if m:
            return int(m.group(1))
    return None


class _ResourceVisitor(ast.NodeVisitor):
    """只看与算力有关的调用与赋值。刻意不做数据流分析——

    能确定就记，不能确定就不记。半确定的值比没有值更危险：它会以同样的自信
    进入估算，而没人知道它是猜的。
    """

    def __init__(self, rel: str, facts: ResourceFacts, tier: int = TIER_ELSEWHERE) -> None:
        self.rel = rel
        self.facts = facts
        self.tier = tier

    def _ev(self, node: ast.AST) -> str:
        return f"{self.rel}:{getattr(node, 'lineno', 0)}"

    # -- import ------------------------------------------------------------

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._note_module(alias.name, node)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._note_module(node.module, node)
        self.generic_visit(node)

    def _note_module(self, module: str, node: ast.AST) -> None:
        root = module.split(".")[0]
        if root in _PARALLEL_IMPORTS:
            self._add_parallel(_PARALLEL_IMPORTS[root], node)
        if module.startswith("torch.distributed"):
            self._add_parallel("torch_distributed", node)
        if root in {"flash_attn", "flash_attn_2"}:
            # flash-attn 2 要 Ampere 起步。这是硬门槛：在 T4 上不是慢，是起不来。
            self._set("min_compute_capability", "8.0", node)
        if root == "bitsandbytes":
            self._set("quantization", "bitsandbytes", node)

    # -- call --------------------------------------------------------------

    def visit_Call(self, node: ast.Call) -> None:
        name = _dotted(node.func)
        short = name.rsplit(".", 1)[-1]
        kwargs = {k.arg: k.value for k in node.keywords if k.arg}

        if short == "from_pretrained":
            ident = _const_str(node.args[0]) if node.args else None
            # 测试夹具刻意用小模型，不是真实负载。litgpt 支持 7B–70B，
            # 却在 tests/ 里写着 stablelm-zephyr-3b —— 拿它当准会低估一个数量级。
            if ident and not _is_fixture_path(self.rel):
                self.facts.model_ids.append(Fact(ident, self._ev(node), self.tier))
            self._scan_dtype(kwargs, node)
            for flag, label in (("load_in_4bit", "4bit"), ("load_in_8bit", "8bit")):
                if _const_bool(kwargs.get(flag)) is True:
                    self._set("quantization", label, node)

        if short in _PRECISION_CALLS:
            self._set("precision", _PRECISION_CALLS[short], node)
        self._scan_dtype(kwargs, node)

        if short in _OPTIMIZER_HINTS or name.startswith("torch.optim."):
            opt = short if short in _OPTIMIZER_HINTS else name.rsplit(".", 1)[-1]
            self._set("optimizer", opt, node)

        if short in {"gradient_checkpointing_enable", "enable_input_require_grads"}:
            self._set("grad_checkpointing", True, node)

        if short == "DataLoader":
            for key, target in (("num_workers", "num_workers"),
                                ("pin_memory", "pin_memory"),
                                ("batch_size", "batch_size")):
                node_val = kwargs.get(key)
                if node_val is None:
                    continue
                if key == "pin_memory":
                    val = _const_bool(node_val)
                else:
                    val = _first_int(_const_any(node_val))
                if val is not None:
                    self._set(target, val, node)

        if short in {"DistributedDataParallel", "DDP"}:
            self._add_parallel("ddp", node)
        if short in {"FullyShardedDataParallel", "FSDP"}:
            self._add_parallel("fsdp", node)
        if short in {"autocast", "amp_autocast"}:
            self._scan_dtype(kwargs, node)

        if short in {"load_state_dict", "resume_from_checkpoint", "load_checkpoint"}:
            self._set("resumable", True, node)

        if name.startswith("torch.cuda") or short == "cuda":
            self._set("uses_gpu", True, node)

        # ★ 上面那条只认**调用名**里的 cuda（`.cuda()` / `torch.cuda.*()`），
        #   而现代 PyTorch 更常见的写法是把 cuda 作为**参数**传进去：
        #       torch.zeros(..., device="cuda")
        #       tensor.to("cuda") / tensor.to(device="cuda:0")
        #       torch.device("cuda")
        #   这些一个都匹配不上，于是一个明明要 GPU 的仓库被判成「不需要 GPU」，
        #   推荐出 CPU 机器，等真跑起来才炸在 "no CUDA device" 上 ——
        #   而那时机器已经开了、环境已经配了、钱已经花了。
        #   所以位置参数和关键字参数里的 cuda 字面量同样算数。
        for candidate in list(node.args) + [k.value for k in node.keywords]:
            literal = _const_str(candidate)
            if literal and (literal == "cuda" or literal.startswith("cuda:")):
                self._set("uses_gpu", True, node)
                break

        self.generic_visit(node)

    def _scan_dtype(self, kwargs: dict[str, ast.AST], node: ast.AST) -> None:
        for key in ("torch_dtype", "dtype", "compute_dtype", "bnb_4bit_compute_dtype"):
            token = _dotted(kwargs[key]) if key in kwargs else ""
            if token in _DTYPE_TOKENS:
                self._set("precision", _DTYPE_TOKENS[token], node)
                if _DTYPE_TOKENS[token] == "bf16":
                    # bf16 是 Ampere(sm_80) 起才有的原生支持
                    self._set("min_compute_capability", "8.0", node)

    # -- assign ------------------------------------------------------------

    def visit_Assign(self, node: ast.Assign) -> None:
        value = _const_any(node.value)
        for target in node.targets:
            key = _dotted(target).rsplit(".", 1)[-1].lower()
            self._maybe_numeric(key, value, node)
        self.generic_visit(node)

    def _maybe_numeric(self, key: str, value: Any, node: ast.AST) -> None:
        num = _first_int(value)
        if num is None:
            return
        for keys, target in ((_BATCH_KEYS, "batch_size"), (_ACCUM_KEYS, "grad_accum"),
                             (_SEQ_KEYS, "seq_len"), (_IMAGE_KEYS, "image_size")):
            if key in keys:
                self._set(target, num, node)
                return

    # -- 写入（先到先得：入口脚本先被扫，覆盖顺序由调用方保证） ----------------

    def _set(self, field_name: str, value: Any, node: ast.AST) -> None:
        """只有**更可信来源**才能覆盖已有值；同层则先到先得。"""
        existing = getattr(self.facts, field_name)
        if existing is not None and existing.tier <= self.tier:
            return
        setattr(self.facts, field_name, Fact(value, self._ev(node), self.tier))

    def _add_parallel(self, label: str, node: ast.AST) -> None:
        if any(f.value == label for f in self.facts.parallel):
            return
        self.facts.parallel.append(Fact(label, self._ev(node), self.tier))


# -- AST 小工具（不依赖 vendored 内部实现，免得上游改了就断） ------------------

def _dotted(node: ast.AST | None) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}".lstrip(".")
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _const_str(node: ast.AST | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _const_bool(node: ast.AST | None) -> bool | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, bool) else None


def _const_any(node: ast.AST | None) -> Any:
    return node.value if isinstance(node, ast.Constant) else None


# -- 文件级事实（非 AST） -----------------------------------------------------

_TORCH_CUDA_RE = re.compile(r"torch[=<>~!\s]*([\d.]+)?\+cu(\d{2,3})")
_FLASH_ATTN_RE = re.compile(r"flash[-_]attn\s*[=<>~]*\s*(\d+)?")


def _scan_requirements(root: Path, facts: ResourceFacts) -> None:
    for name in ("requirements.txt", "pyproject.toml", "setup.py", "environment.yml"):
        path = root / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            m = _TORCH_CUDA_RE.search(line)
            if m and facts.torch_cuda is None:
                cu = m.group(2)
                facts.torch_cuda = Fact(f"cu{cu}", f"{name}:{idx}")
            m2 = _FLASH_ATTN_RE.search(line)
            if m2 and facts.min_compute_capability is None:
                facts.min_compute_capability = Fact("8.0", f"{name}:{idx}")


#: HF 模型 id 的形状：`org/name`，且 **name 里必须带规模后缀**（`7b` / `13B` / `70m`）。
#:
#: 初版只要求 name 里"有数字"，扫出来的是一堆垃圾：qlora 的 `19/2023`（日期）、
#: dinov2 的 `ViT-S/14`（架构记号）、yolov5 的 `example.com/media.mp4`（URL 片段）。
#: 这比读不出来更糟——它会以同样的自信喂进参数量估算。
#: 规模后缀是这里唯一可靠的信号：没有它就宁可留空。
_MODEL_ID_RE = re.compile(
    r"\b([A-Za-z][\w.-]{1,40}/[\w.-]*\d+\s?[bBmM](?![A-Za-z0-9])[\w.-]*)"
)
_MODEL_ID_STOPWORDS = ("github.com", "http", "pip ", "pypi.org", "raw.githubusercontent",
                       "arxiv.org", "://")


def _scan_readme_for_models(root: Path, facts: ResourceFacts) -> None:
    """从 README / 源码里的 assert 提示里捞模型 id。

    很多仓库**把模型名做成运行时参数**：alpaca-lora 的 `base_model: str = ""`
    必须由用户 `--base_model='huggyllama/llama-7b'` 传入，源码里
    `from_pretrained(base_model)` 传的是变量，字面量扫描一无所获。
    但 README 的示例命令与 `assert` 的报错文案里写着真实的模型名——
    那是作者告诉使用者该用什么，作为参数量的来源足够。

    只在字面量一无所获时才用，且证据行指向 README，**不冒充源码事实**。
    """
    if facts.model_ids:
        return
    for name in ("README.md", "README.rst", "readme.md"):
        path = root / name
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for idx, line in enumerate(text.splitlines(), start=1):
            if any(stop in line for stop in _MODEL_ID_STOPWORDS):
                continue
            for candidate in _MODEL_ID_RE.findall(line):
                if "/" not in candidate or candidate.count("/") > 1:
                    continue
                facts.model_ids.append(Fact(candidate, f"{name}:{idx}", TIER_CONFIG))
                if len(facts.model_ids) >= 4:
                    return
        if facts.model_ids:
            return


def _scan_model_config(root: Path, facts: ResourceFacts) -> dict[str, Any]:
    """`config.json` 里的结构参数——有它就能把参数量算准，而不是估。"""
    for candidate in ("config.json", "model_config.json"):
        path = root / candidate
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("hidden_size"):
            return payload
    return {}


_UNKNOWN_IMPACT = {
    "batch_size": "显存的斜率项 ∝ batch，缺它区间会很宽",
    "seq_len": "Transformer 激活 ∝ seq²，缺它无法定上界",
    "precision": "按 fp32 保守估会高一倍",
    "model_ids": "拿不到参数量，只能靠结构猜",
}


def _entry_from_static_facts(root: Path) -> tuple[str, dict[str, str], str]:
    """借 vendored static_facts 找入口脚本 + 它的 argparse 默认值。

    这一步不能省：像 makemore 那样纯 CLI 驱动的仓库，超参**只存在于 argparse
    默认值里**，源码里根本没有 `batch_size = 32` 这样的赋值语句。
    static_facts 的 `EntryPoint.options`（dest → default 的 repr）正是为此存在的。
    """
    try:
        from apps.v2.agent_engine.rsa.static_facts import analyse as _analyse
    except Exception:
        return "", {}, ""
    try:
        facts = _analyse(root)
    except Exception:
        return "", {}, ""
    entries = sorted(facts.entrypoints, key=lambda e: (-e.score, e.file))

    # README 里作者自己写的命令优先于打分。
    # 打分靠的是 `if __name__ == "__main__"` 守卫，而**训练脚本经常没有它**——
    # nanoGPT 的 train.py 就是模块级直接跑，于是打分把 data/openwebtext/prepare.py
    # 选成了入口。README 的 `python train.py config/xxx.py` 才是真话。
    readme_entry, readme_config = _entry_from_readme(facts.readme_commands or [], root)
    if readme_entry:
        for e in entries:
            if e.file == readme_entry:
                return readme_entry, dict(e.options or {}), readme_config
        return readme_entry, {}, readme_config

    if not entries:
        return "", {}, ""
    best = entries[0]
    return best.file, dict(best.options or {}), ""


_PY_CMD_RE = re.compile(r"\bpython3?\s+(?:-m\s+\S+|(\S+\.py))((?:\s+\S+)*)")


def _entry_from_readme(commands: list[str], root: Path) -> tuple[str, str]:
    """从 README 命令里认出 `python train.py [config.py]`。

    返回 (入口相对路径, 配置文件相对路径)。只认仓库里真实存在的文件——
    README 里的示例路径经常是占位符，认错了比认不出更糟。
    """
    best: tuple[str, str] = ("", "")
    for cmd in commands:
        m = _PY_CMD_RE.search(cmd)
        if not m or not m.group(1):
            continue
        script = m.group(1)
        if not (root / script).is_file():
            continue
        config = ""
        for token in (m.group(2) or "").split():
            if token.startswith("-"):
                continue
            if token.endswith((".py", ".json", ".yaml", ".yml")) and (root / token).is_file():
                config = token
                break
        # 训练脚本优先于其它脚本（prepare/sample/bench 都不是我们要估的对象）
        score = (2 if "train" in script.lower() else 1, 1 if config else 0)
        prev = (2 if "train" in best[0].lower() else 1, 1 if best[1] else 0) if best[0] else (0, 0)
        if score > prev:
            best = (script, config)
    return best


def _apply_cli_defaults(facts: ResourceFacts, entry_file: str, options: dict[str, str]) -> None:
    """argparse 默认值按 TIER_CLI 写入——它是**最终生效值**，压过源码里的同名变量。"""
    for dest, raw in options.items():
        key = dest.lower()
        num = _first_int(raw.strip("'\"") if isinstance(raw, str) else raw)
        if num is None:
            continue
        for keys, target in ((_BATCH_KEYS, "batch_size"), (_ACCUM_KEYS, "grad_accum"),
                             (_SEQ_KEYS, "seq_len"), (_IMAGE_KEYS, "image_size")):
            if key in keys:
                existing = getattr(facts, target)
                if existing is None or existing.tier > TIER_CLI:
                    setattr(facts, target, Fact(num, f"{entry_file}:argparse", TIER_CLI))
                break


_CRITICAL_HYPERPARAMS = ("batch_size", "grad_accum", "seq_len", "image_size")


def analyse_resources(root: str | Path, *, entry_first: str = "") -> ResourceFacts:
    """扫一个仓库，返回算力相关事实。

    `entry_first` 给入口脚本的相对路径；留空则用 static_facts 打分最高的那个。
    入口脚本按 TIER_ENTRY 记，其余文件按 TIER_ELSEWHERE——**只有更可信的来源
    才能覆盖已有值**，避免把不同配置的超参拼成一个不存在的组合。
    """
    root_path = Path(root)
    facts = ResourceFacts()

    detected_entry, options, config_rel = _entry_from_static_facts(root_path)
    entry_rel = entry_first or detected_entry

    files = sorted(p for p in root_path.rglob("*.py")
                   if not any(part.startswith((".", "__pycache__")) for part in p.parts))
    entry_path = (root_path / entry_rel) if entry_rel else None
    if entry_path is not None:
        files.sort(key=lambda p: (p != entry_path, str(p)))

    for path in files[:400]:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8", errors="ignore"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        rel = str(path.relative_to(root_path))
        tier = TIER_ENTRY if (entry_path is not None and path == entry_path) else TIER_ELSEWHERE
        _ResourceVisitor(rel, facts, tier).visit(tree)

    # README 命令里点名的配置文件按 TIER_CONFIG 扫——它比仓库里随便一个 .py 可信，
    # 但比入口脚本本身弱。
    if config_rel and config_rel.endswith(".py"):
        cfg_path = root_path / config_rel
        try:
            cfg_tree = ast.parse(cfg_path.read_text(encoding="utf-8", errors="ignore"))
            _ResourceVisitor(config_rel, facts, TIER_CONFIG).visit(cfg_tree)
        except (OSError, SyntaxError):
            pass

    if entry_rel and options:
        _apply_cli_defaults(facts, entry_rel, options)

    _scan_requirements(root_path, facts)
    _scan_readme_for_models(root_path, facts)

    # 关键超参是否同源。不同源意味着这个组合现实中可能根本不存在，
    # 下游必须按低置信处理（区间加宽 + 触发保险档）。
    # 一致性按**来源层级**判，不按文件名。
    #
    # 「入口 + 它 README 里点名的配置文件」是**设计好的组合**：配置覆盖入口默认值
    # 正是 nanoGPT 这类仓库的正常模式，按文件名判会把它误报成拼凑。
    # 真正危险的是 TIER_ELSEWHERE ——那是从一个与本次运行无关的文件里捡来的值
    # （比如 bench.py 的 batch），它和其它值放在一起就是个不存在的配置。
    sources: dict[str, str] = {}
    stray: dict[str, str] = {}
    for name in _CRITICAL_HYPERPARAMS:
        got = getattr(facts, name)
        if got is None:
            continue
        sources[name] = got.source_file
        if got.tier >= TIER_ELSEWHERE:
            stray[name] = got.source_file
    facts.hyperparam_sources = sources
    facts.coherent = not stray
    if stray:
        facts.unknowns.append(
            "这些关键超参来自与入口无关的文件（"
            + "、".join(f"{k}←{v}" for k, v in stray.items())
            + "），拼在一起可能不是一个真实存在的配置组合"
        )

    for name, why in _UNKNOWN_IMPACT.items():
        got = getattr(facts, name, None)
        if not got:
            facts.unknowns.append(f"{name}: {why}")

    return facts
