"""把上传的代码变成一个**只有那台机器自己能访问**的 git URL。

## 为什么必须是 URL

`rsa/pipeline.py:130-134` 显式拒绝远端路径下的 `local_path`：
「provide a repository URL so cloning happens on the remote server」。
而 `rsa/remote_backend.py:315` 的 `create_container` 是在远端机器的
**容器内** `git clone`。所以 URL 必须是那台机器能访问的。

## 为什么是 localhost 而不是一个公网 Git 服务

设计文档原本打算起一个对外可达的内部 Git 服务，配一次性凭据。
那意味着：一个新的部署面、TLS、鉴权、凭据轮换，
以及**用户的代码挂在一个公网 URL 上**——凭据一旦泄漏就是源码泄漏。

而容器用 `--network host`（`remote_backend.py:320`）与宿主共享网络命名空间，
所以容器里的 `127.0.0.1` 就是机器本身的回环。在机器上起一个
只监听 127.0.0.1 的 git daemon，容器就能 clone，而**外面谁也连不上**。
没有凭据可泄，随机器释放自动消失。

★ 这条路完全依赖 `--network host` 这个 vendored 行为。
  `test_experiment_git_daemon.py` 里有一条断言直接读 RSA 源码钉住它 ——
  它一旦消失，症状是「clone 失败」，看不出真因。

## 端口

用 git 协议的默认端口 9418。只监听回环，所以不存在和别人抢端口的问题
（这台机器是我们独占租来的）。
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

#: 机器上放裸仓的地方。放 /opt 而不是 /tmp —— /tmp 可能被 tmpfs 或
#: 定时清理盯上，而这个仓要活到实验跑完。
REMOTE_GIT_ROOT = "/opt/deepevol/git"
REMOTE_REPO_NAME = "repo.git"
GIT_DAEMON_PORT = 9418

_UPLOAD_TIMEOUT = 600.0
_SETUP_TIMEOUT = 300.0


class GitDaemonError(RuntimeError):
    """把代码送上机器这一步失败了。消息面向排查，不直接给用户看。"""


def local_bundle(repo_dir: str | Path, dest: str | Path | None = None) -> Path:
    """把本地仓库打成一个 git bundle。

    ★ 用 bundle 而不是 tar 整个 `.git`：bundle 是单文件、自带完整性校验，
      而且 `git clone <bundle>` 原生支持 —— 机器那侧不需要懂我们的打包约定。
    """
    source = Path(repo_dir)
    target = Path(dest) if dest else Path(
        tempfile.mkdtemp(prefix="deepevol-bundle-")) / "repo.bundle"
    target.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "bundle", "create", str(target), "--all"],
        cwd=source, capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[:400]
        raise GitDaemonError(f"git bundle 失败：{detail}")
    return target


async def serve_repo_on_machine(runtime, repo_dir: str | Path, *, port: int = GIT_DAEMON_PORT) -> str:
    """把仓库送上机器并起一个只监听回环的 git daemon，返回 clone URL。

    `runtime` 是租期给的 remote_relay Runtime（有 `upload` 与 `exec`）。
    """
    bundle = local_bundle(repo_dir)
    remote_bundle = f"{REMOTE_GIT_ROOT}/repo.bundle"
    repo_path = f"{REMOTE_GIT_ROOT}/{REMOTE_REPO_NAME}"

    await _exec(runtime, f"mkdir -p {shlex.quote(REMOTE_GIT_ROOT)}", timeout=60)

    # ★ 机器上要有 git。基础镜像里不一定装了（bootstrap.sh 原来只装 docker），
    #   所以这里幂等补一次 —— 老镜像还在跑，不能假设它已经有了。
    await _exec(
        runtime,
        "command -v git >/dev/null 2>&1 || "
        "(apt-get -qq update >/dev/null 2>&1 && apt-get -qq install -y git >/dev/null 2>&1)",
        timeout=_SETUP_TIMEOUT,
    )

    try:
        await runtime.upload(str(bundle), remote_bundle)
    except Exception as exc:
        raise GitDaemonError(f"把代码传到机器上失败：{exc}") from exc

    # bundle → 裸仓。`--bare` 让 git daemon 能直接导出它。
    await _exec(
        runtime,
        f"rm -rf {shlex.quote(repo_path)} && "
        f"git clone --bare --quiet {shlex.quote(remote_bundle)} {shlex.quote(repo_path)} && "
        # git daemon 默认只导出带这个标记的仓；我们同时用 --export-all，
        # 两个都放着是因为 daemon 的默认策略在不同发行版上被改过。
        f"touch {shlex.quote(repo_path)}/git-daemon-export-ok",
        timeout=_SETUP_TIMEOUT,
    )

    # ★ `--listen=127.0.0.1`：**外面谁也连不上**，这是整个方案的安全前提。
    #   容器靠 --network host 与宿主共享回环，所以它连得上。
    await _exec(
        runtime,
        # ★ `git-daemo[n]` 的方括号是**必须的**，不是风格问题。
        #   `pkill -f` 拿整条命令行匹配，而跑这条命令的 shell 自己的命令行里
        #   就含着这个模式串 —— 写成 'git-daemon.*PORT' 的话，
        #   pkill 会把**执行它的那个 shell** 一起杀掉，
        #   表现为 exit=-1、输出为空，`|| true` 也救不回来（进程已经没了）。
        #   加个方括号后，命令行里的字面量是 `git-daemo[n]`，不再匹配该正则，
        #   而真正的 git-daemon 进程照常被匹中。
        f"(pkill -f 'git-daemo[n].*{port}' || true); "
        f"git daemon --base-path={shlex.quote(REMOTE_GIT_ROOT)} --export-all "
        f"--listen=127.0.0.1 --port={port} --detach --reuseaddr "
        f"--enable=upload-pack",
        timeout=60,
    )
    url = f"git://127.0.0.1:{port}/{REMOTE_REPO_NAME}"

    # 起没起来当场验，别等到 RSA clone 失败时才发现 ——
    # 那时的报错是「clone failed」，看不出是 daemon 没起来。
    probe = await _exec(
        runtime, f"git ls-remote {shlex.quote(url)} >/dev/null && echo DAEMON_OK",
        timeout=120, allow_failure=True,
    )
    if "DAEMON_OK" not in probe:
        raise GitDaemonError(
            f"git daemon 起来了但连不上（{url}）。机器上的输出：{probe[:400]}"
        )
    logger.info("git daemon 就绪 %s", url)
    return url


async def _exec(runtime, command: str, *, timeout: float, allow_failure: bool = False) -> str:
    result = await runtime.exec(command, timeout=timeout)
    output = ((getattr(result, "stdout", "") or "") +
              (getattr(result, "stderr", "") or "")).strip()
    status = getattr(result, "exit_status", 0)
    if status not in (0, None) and not allow_failure:
        # ★ 一定要带上**是哪条命令**。只报 exit code 和输出的话，
        #   遇到 exit=-1 且输出为空（exec 通道当场失败）就完全无从下手 ——
        #   这里有五条命令，不说清是哪条等于没报错。
        raise GitDaemonError(
            f"机器上执行失败（exit={status}）：命令 `{command[:160]}` "
            f"输出：{output[:400] or '(空)'}")
    return output
