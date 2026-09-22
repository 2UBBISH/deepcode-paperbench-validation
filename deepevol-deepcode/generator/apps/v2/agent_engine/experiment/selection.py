"""把用户那句「就第二个吧」变成一台具体的机器。

## 为什么不交给 LLM

这一步的下游是**花钱**。选错一档，用户付的是另一台机器的价钱，
而且他不会立刻发现 —— 报告里写着经济档，实际开的是加速档。
确定性匹配能穷举、能测、能解释；LLM 在这里的额外能力（理解绕弯的说法）
换不来它带来的风险。

真正说不清的时候，正确的动作是**回去问**，不是猜。

## 认哪些说法

用户会用四种方式指代一档：档位名（「经济档」「便宜的」）、
序号（「第二个」「2」）、机器型号（「A10」「T4 那个」）、
或者干脆只说「好」（只有一档时才成立）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

#: 档位 key → 用户可能说出口的词。写全一点：漏一个词的代价是
#: 用户明明说清楚了却被反问一遍。
_TIER_WORDS: dict[str, tuple[str, ...]] = {
    "economy": ("经济", "便宜", "省钱", "最低", "低配", "economy", "cheap"),
    "standard": ("稳妥", "标准", "默认", "推荐", "standard", "default"),
    "insurance": ("保险", "保底", "宽裕", "余量", "insurance"),
    "speed": ("加速", "快", "多卡", "速度", "speed", "fast"),
}

_ORDINALS: dict[str, int] = {
    "第一": 1, "第二": 2, "第三": 3, "第四": 4,
    "一": 1, "二": 2, "三": 3, "四": 4,
    "1": 1, "2": 2, "3": 3, "4": 4,
    "first": 1, "second": 2, "third": 3, "fourth": 4,
}

#: 只说「好/可以/就它」——只有**唯一一档**时才能当作确认。
#: 有多档时这句话没有指向，猜哪个都可能猜错。
_BARE_YES = ("好", "可以", "行", "就它", "就这个", "确认", "开始",
             "ok", "yes", "go", "sure")

_CANCEL = ("取消", "算了", "不用了", "先不", "停", "cancel", "stop", "no")


@dataclass
class Selection:
    """`target` 非空表示选定了；否则看 `cancelled` 与 `question`。"""

    target: str = ""
    tier: str = ""
    cancelled: bool = False
    #: 说不清时要问用户的话。**不猜**——猜错的代价是用户付了另一台机器的钱。
    question: str = ""

    @property
    def resolved(self) -> bool:
        return bool(self.target)


def _norm(text: str) -> str:
    return re.sub(r"\s+", "", (text or "")).lower()


def _mentions(text: str, words: tuple[str, ...]) -> bool:
    """文本里有没有出现这些词。

    ★ **ASCII 词必须按词边界匹配。** 踩过：取消词表里的 `no` 是
      「eco*no*my」的子串，于是用户说 economy 被判成了取消 ——
      而取消的代价是整轮白做。中文没有词边界，按子串匹配是对的。
    """
    for word in words:
        if word.isascii():
            if re.search(rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])", text):
                return True
        elif word in text:
            return True
    return False


def _mentions_card(text: str, accelerator_type: str) -> bool:
    """文本里有没有提到这张卡。全称与短名都认（用户说「A10」不说「NVIDIA A10」）。"""
    full = _norm(accelerator_type)
    if not full:
        return False
    if full in text:
        return True
    short = _norm(accelerator_type.replace("NVIDIA", ""))
    # 短名要求**独立成词**：否则 "a10" 会命中 "a100"，
    # 而那是两张差着一个数量级的卡。
    return bool(short) and re.search(rf"(?<![a-z0-9]){re.escape(short)}(?![a-z0-9])", text) is not None


def _ordinal_in(text: str) -> int | None:
    """从文本里读序号。裸数字必须独立成词，中文序数词不受此限。"""
    for word, index in _ORDINALS.items():
        if word.isdigit():
            if re.search(rf"(?<![a-z0-9]){word}(?![a-z0-9])", text):
                return index
        elif word in text:
            return index
    return None


def parse_selection(reply: str, choices: list[dict[str, Any]]) -> Selection:
    """把用户的回复对到 `choices` 里的一项。

    `choices` 是首轮产出的那个列表，每项至少有 `tier` 与 `target`。
    """
    if not choices:
        return Selection(question="现在没有可选的配置，请先让我重新读一遍代码")

    text = _norm(reply)
    if not text:
        return Selection(question="想用哪一档？告诉我档位名或序号就行")

    if _mentions(text, _CANCEL):
        return Selection(cancelled=True)

    # 1) 档位名。最直接，也最不容易歧义。
    for choice in choices:
        tier = str(choice.get("tier") or "")
        if _mentions(text, _TIER_WORDS.get(tier, ())):
            return Selection(target=str(choice.get("target") or ""), tier=tier)

    # 2) 机器型号。用户常常直接说「就 A10」而不是「NVIDIA A10」，
    #    所以要同时认全称和短名。
    #    ★ 只在**唯一命中**时接受：目录里可能有两台都带 A10 的规格，
    #      那时「A10」并没有指向哪一台。
    hits = [c for c in choices if _mentions_card(text, str(c.get("accelerator_type") or ""))]
    if len(hits) == 1:
        return Selection(target=str(hits[0].get("target") or ""),
                         tier=str(hits[0].get("tier") or ""))
    if len(hits) > 1:
        names = "、".join(str(c.get("instance_type") or "") for c in hits)
        return Selection(question=f"有几档都是这张卡（{names}），你要哪一个？")

    # 3) 序号。
    #    ★ 裸数字必须**独立成词**才算序号。踩过：「就 A10」里的 `1`
    #      被读成「第一个」，于是用户说 A10、系统开了 T4 ——
    #      这类错不会报错，只会让他付另一台机器的钱。
    index = _ordinal_in(text)
    if index is not None and 1 <= index <= len(choices):
        choice = choices[index - 1]
        return Selection(target=str(choice.get("target") or ""),
                         tier=str(choice.get("tier") or ""))

    # 4) 光说「好」。只有一档时它才有指向。
    if _mentions(text, _BARE_YES):
        if len(choices) == 1:
            return Selection(target=str(choices[0].get("target") or ""),
                             tier=str(choices[0].get("tier") or ""))
        return Selection(question="有好几档，你要哪一个？说档位名或序号都行")

    return Selection(
        question="没太看懂你要哪一档。可以说档位名（比如「经济档」）或者序号（比如「第二个」）"
    )
