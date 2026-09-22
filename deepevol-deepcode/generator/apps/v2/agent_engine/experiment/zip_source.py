"""用户传上来的压缩包 → 一个能读、能分析、能送去远端跑的本地仓库。

## 三道必须在租机器之前拦住的关

**一、zip-slip。** 压缩包里的路径可以是 `../../etc/cron.d/x` 或一条指向
`/` 的软链，解压时就写到了包外。判据照 `rsa/remote_backend.py:578`
的 tarfile 版本抄一份等价的（那是同类防护的现成参考），
但 zipfile 的软链要看 `external_attr` 的高 16 位，与 tarfile 的 API 不同。

**二、zip bomb。** 只量压缩包大小挡不住 —— 一个几 MB 的包可以解出几十 GB。
必须**边解边量**，超了立刻停手，而不是解完再看（解完时盘已经满了）。

**三、传错文件。** 没有任何 `.py` 的包多半是用户拖错了东西。
早说比租完机器再说便宜得多。

## 为什么要 git init

远端 RSA 的 `create_checkpoint` / `rollback_to_checkpoint` 都建立在
「容器内有 git 工作树」的假设上（设计文档 §6）。而且有了 commit 才有
fingerprint —— 与 GitHub 路径同一套语义，日志里能对上是哪一版代码。
"""

from __future__ import annotations

import shutil
import stat
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

from apps.common.repo_url import RepoSource, RepoSourceError

from .repo_source import MAX_REPO_MB, ClonedRepo

#: 解压后的体积上限。与 GitHub 克隆同一个数 —— 用户不该因为换了上传方式
#: 就遇到不同的限制。
MAX_EXTRACTED_MB = MAX_REPO_MB

#: 单个文件的上限。一个 1.5GB 的 checkpoint 混在代码里传上来，
#: 既没用（远端会重新训）又会把盘吃掉。
MAX_MEMBER_MB = 512

#: 条目数上限。几十万个空文件也是一种 bomb，而且它绕过体积检查。
MAX_MEMBERS = 50_000


def _reject(message: str) -> None:
    raise RepoSourceError(message)


def _validate_member(info: zipfile.ZipInfo) -> None:
    """解压前逐条判。**拒绝比修复安全** —— 我们不知道用户想要什么，
    只知道这条路径不能写。"""
    name = info.filename
    if not name:
        _reject("压缩包里有匿名条目，看起来已经损坏")
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or name.startswith("/") or (len(name) > 1 and name[1] == ":"):
        _reject(f"压缩包里有绝对路径（{name[:80]}），不安全")
    if ".." in path.parts:
        _reject(f"压缩包里有跳出目录的路径（{name[:80]}），不安全")
    # ★ zipfile 的软链藏在 external_attr 的高 16 位（Unix mode）里，
    #   不像 tarfile 有 issym()。一条指向 / 的软链能让后续写入跑到包外。
    #
    #   ★★ 只在**类型位真的存在**时才判类型。很多 zip 写入器
    #      （包括 Python 自己的 `writestr`）只存权限位、不存文件类型，
    #      那时 S_IFMT 是 0 —— 把这种当「特殊文件」拒掉，
    #      会把绝大多数正常压缩包全部误杀。
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if file_type == stat.S_IFLNK:
        _reject(f"压缩包里有符号链接（{name[:80]}），不安全")
    if file_type and file_type not in (stat.S_IFREG, stat.S_IFDIR):
        _reject(f"压缩包里有特殊文件（{name[:80]}），不安全")
    if info.file_size > MAX_MEMBER_MB * 1024 * 1024:
        _reject(
            f"压缩包里有 {info.file_size / 1024 / 1024:.0f}MB 的单个文件"
            f"（{Path(name).name}），超过 {MAX_MEMBER_MB}MB。"
            "模型权重/数据集请让代码去下载，不要打进包里"
        )


def _strip_single_root(root: Path) -> Path:
    """`unzip` 出来常常是 `项目名-main/` 套一层。剥掉它，
    否则静态分析会在一个只含一个目录的根上找不到入口脚本。"""
    entries = [p for p in root.iterdir() if p.name != "__MACOSX"]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return root


def extract_zip_repo(
    zip_path: str | Path,
    *,
    dest: str | Path,
    name: str = "uploaded",
    max_mb: int = MAX_EXTRACTED_MB,
) -> ClonedRepo:
    """解压 + 校验 + `git init`，返回和 GitHub 路径同形状的 `ClonedRepo`。

    失败一律抛 `RepoSourceError`，消息**会原样给用户看**，所以要说人话。
    """
    archive_path = Path(zip_path)
    target = Path(dest)
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True, exist_ok=True)

    try:
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            if not members:
                _reject("压缩包是空的")
            if len(members) > MAX_MEMBERS:
                _reject(f"压缩包里有 {len(members)} 个条目，超过 {MAX_MEMBERS} 上限")

            budget = max_mb * 1024 * 1024
            written = 0
            for info in members:
                _validate_member(info)
                # ★ 边解边量。解完再看的话，盘在「看」之前就已经满了。
                written += info.file_size
                if written > budget:
                    _reject(
                        f"解压后超过 {max_mb}MB 上限。"
                        "请先精简，或把大文件挪到数据集下载步骤里"
                    )
                archive.extract(info, target)
    except RepoSourceError:
        shutil.rmtree(target, ignore_errors=True)
        raise
    except zipfile.BadZipFile as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise RepoSourceError("这个文件不是有效的 zip 压缩包") from exc
    except OSError as exc:
        shutil.rmtree(target, ignore_errors=True)
        raise RepoSourceError(f"解压失败：{exc}") from exc

    root = _strip_single_root(target)
    if not any(root.rglob("*.py")):
        shutil.rmtree(target, ignore_errors=True)
        _reject(
            "压缩包里没有任何 .py 文件 —— 是不是传错了？"
            "我们需要的是能跑的代码仓库"
        )

    revision = _git_init_commit(root)
    source = RepoSource(
        url=f"zip://{name}", host="upload", owner="upload", repo=name,
        revision=revision,
    )
    return ClonedRepo(source=source, local_path=root, revision=revision)


def _git_init_commit(root: Path) -> str:
    """把解压出来的目录变成一个有一次提交的 git 仓库。

    远端 RSA 的 checkpoint / rollback 都假设容器内有 git 工作树；
    而 commit sha 同时充当 fingerprint，与 GitHub 路径保持同一套语义。
    """
    env_args = [
        "-c", "user.email=experiment-agent@deepevol.local",
        "-c", "user.name=DeepEvol Experiment Agent",
        "-c", "commit.gpgsign=false",
    ]
    steps = [
        ["git", "init", "--quiet", "--initial-branch=main"],
        ["git", *env_args, "add", "-A"],
        ["git", *env_args, "commit", "--quiet", "-m", "uploaded archive"],
    ]
    for step in steps:
        result = subprocess.run(
            step, cwd=root, capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()[:300]
            raise RepoSourceError(f"给上传的代码建 git 仓库失败：{detail}")
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root,
        capture_output=True, text=True, check=False,
    )
    return (head.stdout or "").strip()
