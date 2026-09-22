"""把用户给的东西变成一个**能读的本地目录**，并且在租机器之前就拒掉坏输入。

## 为什么校验必须发生在租机器之前

租一台 GPU 机是几秒钟就开始计费的动作。「URL 打错了」「仓库是私有的」
「仓库有 40GB」这几件事，全都可以在花钱之前发现。
在租完之后才发现，用户付的是「发现自己打错字」的钱。

## 归一成 URL，不归一成本地文件

RSA 的 `create_container(repo_url, revision)` 在**远端** clone。
`put_file` 虽然能绕过去，但会丢掉 revision 语义，而
`create_checkpoint` / `rollback_to_checkpoint` 都建立在
「容器内有 git 工作树」的假设上。所以本地这份克隆只用于**静态分析**，
远端仍然按 URL clone。

## zip 上传（未做）

设计里 zip 要先解压、初始化裸仓、再给一个带一次性凭据的内部 repo_url。
那需要一个对远端租赁机器可达的 Git 服务（W2），现在还没有。
所以本模块只处理 URL，遇到 zip 明确报错，而不是悄悄降级成 put_file ——
后者会让 checkpoint / rollback 在跑到一半时才失败。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

# ★ 地址校验与域名白名单在 `apps/common/repo_url.py`，**API 与 Agent 共用一份**。
#   那是安全边界（任意 URL ⇒ clone 会去连任意主机，SSRF 面），
#   复制成两份迟早有一份先被放宽而没人注意到。
from apps.common.repo_url import (  # noqa: F401 —— 转出给本包的调用方
    ALLOWED_HOSTS,
    RepoSource,
    RepoSourceError,
    parse_repo_url,
)

#: 克隆体积上限。超了就拒，而不是租完机器才发现盘不够。
MAX_REPO_MB = 2048

CLONE_TIMEOUT_SECONDS = 300.0


@dataclass
class ClonedRepo:
    """一次本地克隆的结果。

    ★ 本地路径**不放进共用的 `RepoSource`**：那个类型 API 侧也在用，
      而 API 从不克隆。把只有一侧有意义的字段塞进共用类型，
      读的人会以为另一侧也能拿到。
    """

    source: RepoSource
    local_path: Path
    revision: str

    @property
    def slug(self) -> str:
        return self.source.slug

    def cleanup(self) -> None:
        shutil.rmtree(self.local_path, ignore_errors=True)


def clone_for_analysis(
    source: RepoSource,
    *,
    revision: str = "",
    dest: Path | None = None,
    max_mb: int = MAX_REPO_MB,
    timeout: float = CLONE_TIMEOUT_SECONDS,
) -> ClonedRepo:
    """浅克隆到本地，只为静态分析用。

    ★ `--depth 1` + `--filter=blob:none`：我们只读代码和配置，不需要历史。
      一个带大文件历史的仓库完整 clone 可能是几个 GB，而分析用不到那些字节。

    ★ 克隆完记录 commit sha 作 fingerprint。远端 RSA 会按同一个 revision
      再 clone 一次 —— 不记的话，两次 clone 之间仓库被 push 了，
      我们分析的和远端跑的就不是同一份代码，而这种错极难查。
    """
    target = Path(dest) if dest else Path(tempfile.mkdtemp(prefix="deepevol-repo-"))
    target.mkdir(parents=True, exist_ok=True)
    cmd = [
        "git", "clone", "--depth", "1", "--filter=blob:none",
        "--quiet", "--no-tags",
    ]
    # ★ `--branch` 只认分支/标签。用户贴的常常是 commit sha（复现某篇论文的
    #   固定版本），那时要先浅克隆默认分支，再单独 fetch 那个 commit —— 否则
    #   git 报 "Remote branch <sha> not found"，用户看到的却是「找不到仓库」。
    pinned_commit = bool(revision) and _looks_like_commit(revision)
    if revision and not pinned_commit:
        cmd += ["--branch", revision]
    cmd += [source.url, str(target)]

    steps: list[list[str]] = [cmd]
    if pinned_commit:
        steps.append(["git", "-C", str(target), "fetch", "--depth", "1", "--quiet",
                      "origin", revision])
        steps.append(["git", "-C", str(target), "checkout", "--quiet", "FETCH_HEAD"])

    for step in steps:
        try:
            proc = subprocess.run(
                step, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired as exc:
            shutil.rmtree(target, ignore_errors=True)
            raise RepoSourceError(f"克隆超时（{timeout:.0f} 秒），仓库可能太大或网络不通") from exc

        if proc.returncode != 0:
            shutil.rmtree(target, ignore_errors=True)
            if step is not cmd:
                raise RepoSourceError(f"仓库里没有 commit {revision}，请检查版本号是否写对")
            raise RepoSourceError(_clone_error_message(proc.stderr or proc.stdout, source))

    size_mb = _dir_size_mb(target)
    if size_mb > max_mb:
        shutil.rmtree(target, ignore_errors=True)
        raise RepoSourceError(
            f"仓库有 {size_mb:.0f}MB，超过 {max_mb}MB 上限。"
            "请先精简，或把大文件挪到数据集下载步骤里"
        )

    head = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    resolved = (head.stdout or "").strip() or revision
    source.revision = resolved
    return ClonedRepo(source=source, local_path=target, revision=resolved)


_COMMIT_SHA = re.compile(r"^[0-9a-fA-F]{7,40}$")


def _looks_like_commit(revision: str) -> bool:
    """7-40 位十六进制当作 commit sha。分支名也可能长这样（如 `deadbeef`），
    但那种分支极少见，而贴 sha 复现固定版本是常态。"""
    return _COMMIT_SHA.match(revision.strip()) is not None


def _clone_error_message(stderr: str, source: RepoSource) -> str:
    """把 git 的英文报错换成用户能据以行动的中文。

    「Authentication failed」对用户的含义是「这个仓库是私有的」——
    照抄原文他不知道该怎么办。
    """
    text = (stderr or "").strip()
    low = text.lower()
    if "could not read username" in low or "authentication failed" in low or "403" in low:
        return f"{source.slug} 看起来是私有仓库。我们只能访问公开仓库"
    # 「Remote branch X not found」也含 "not found"，得先于「仓库不存在」判断，
    # 否则一个写错的分支名会被翻译成「找不到仓库」。
    if "remote branch" in low and "not found" in low:
        return "指定的分支或标签不存在"
    if "not found" in low or "repository not found" in low or "404" in low:
        return f"找不到 {source.slug}，请检查地址是否写错"
    if "could not resolve host" in low or "unable to access" in low:
        return "连不上代码托管服务，请稍后再试"
    return f"克隆失败：{text[:300] or '未知原因'}"


def _dir_size_mb(root: Path) -> float:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.is_symlink():
                total += path.stat().st_size
        except OSError:
            continue
    return total / (1024 * 1024)
