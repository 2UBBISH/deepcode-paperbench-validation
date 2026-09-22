"""代码仓库地址的校验。**API 与 Agent 共用这一份。**

## 为什么放在 apps/common 而不是各留一份

这里的域名白名单是**安全边界**：允许任意 URL 意味着 `git clone` 会去连
用户指定的任意主机，在服务端执行的场景下那是一个 SSRF 面。
安全相关的判断复制成两份，迟早会有一份先被放宽，而另一份不会有人注意到。

## 为什么 API 侧也要有

实验会话在**建立之前**就该拒掉打错的地址。放到第一轮再报，
用户已经有了一个跑不起来的会话，还得自己去删。
而这件事完全不需要联网 —— 纯格式与白名单检查。

克隆逻辑不在这里：只有 Agent 需要，且它依赖 subprocess 与本地盘。
见 `Agent/DeepEvol/experiment/repo_source.py`。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: 只认这几个域名。见模块 docstring 里的 SSRF 说明。
ALLOWED_HOSTS = ("github.com", "gitlab.com", "gitee.com")

_URL_RE = re.compile(
    r"^https://(?P<host>[a-z0-9.\-]+)/(?P<owner>[\w.\-]+)/(?P<repo>[\w.\-]+?)(?:\.git)?/?$",
    re.IGNORECASE,
)


class RepoSourceError(ValueError):
    """用户给的仓库有问题。**消息会原样给用户看**，所以要说人话。"""


@dataclass
class RepoSource:
    url: str
    host: str
    owner: str
    repo: str
    revision: str = ""

    @property
    def slug(self) -> str:
        return f"{self.owner}/{self.repo}"

    def to_dict(self) -> dict[str, str]:
        return {
            "url": self.url, "host": self.host, "owner": self.owner,
            "repo": self.repo, "revision": self.revision, "slug": self.slug,
        }


def parse_repo_url(raw: str) -> RepoSource:
    """校验并拆解仓库 URL。**不联网** —— 纯粹的格式与白名单检查。

    分成两步（先 parse 后 clone）是为了让前端能在用户还在输入时就给出反馈，
    而不必等一次可能几十秒的 clone。
    """
    text = (raw or "").strip()
    if not text:
        raise RepoSourceError("请填写代码仓库地址")
    if text.endswith(".zip") or text.startswith("file://"):
        raise RepoSourceError(
            "暂时只支持 Git 仓库链接。压缩包上传需要一个远端机器能访问的内部 Git 服务，"
            "还没做"
        )
    if text.startswith("git@") or text.startswith("ssh://"):
        raise RepoSourceError("请用 https 链接，SSH 形式需要密钥、我们不持有你的密钥")
    if not text.startswith("https://"):
        raise RepoSourceError("请用 https:// 开头的完整仓库地址")

    match = _URL_RE.match(text)
    if not match:
        raise RepoSourceError(
            "看不懂这个地址。请给形如 https://github.com/用户名/仓库名 的链接"
        )
    host = match.group("host").lower()
    if host not in ALLOWED_HOSTS:
        raise RepoSourceError(
            f"暂时只支持 {'、'.join(ALLOWED_HOSTS)}；收到的是 {host}"
        )
    return RepoSource(
        url=f"https://{host}/{match.group('owner')}/{match.group('repo')}.git",
        host=host, owner=match.group("owner"), repo=match.group("repo"),
    )
