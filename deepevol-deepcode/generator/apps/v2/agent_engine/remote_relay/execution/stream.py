"""流式输出的解码与治理。

三件事，缺一个远端输出就会以某种方式糊掉：

1. **UTF-8 边界**：SSH chunk 会从任意字节位切断，半个汉字必须缓存到下一 chunk 再解码，
   否则中文日志和 tqdm 进度条必然乱码。用增量解码器。
2. **哨兵剥离**：粘性 cwd 靠在命令尾部打一个 `__RELAY_STATE__<cwd>` 标记回传，它是中转层的
   内部协议，一个字节都不能透给 Agent——包括流式路径。
3. **输出上限**：一次 `pip install -v` 能吐几十 MB，全塞进 Agent 上下文就废了。超限只留
   head+tail 并显式标注截断，完整内容留在远端。
"""

from __future__ import annotations

import codecs

from ..types import DEFAULT_MAX_OUTPUT_BYTES


STATE_MARKER = b"\n__RELAY_STATE__"


class SentinelSplitter:
    """按字节流切出"正文"和尾部哨兵载荷。

    正文里出现半个 marker 前缀时会先扣住不发（最多扣 len(marker) 字节），等下一 chunk 来了
    再判断——所以 Agent 永远看不到 `__RELAY_STA` 这种残片。扣住的字节不计入 offset，
    断线后从 offset 续读会把它们重新读一遍，不丢。
    """

    def __init__(self, marker: bytes = STATE_MARKER, *, start_offset: int = 0) -> None:
        self._marker = marker
        self._held = b""
        self._tail = b""
        self._found = False
        self.offset = start_offset
        """已经吐出去的正文字节数（含起始偏移）——续读游标，扣住未发的字节不算在内。"""
        self.consumed = start_offset
        """已经吃进来的字节数（含起始偏移）——读取游标。

        必须和 offset 分开：从文件按偏移轮询读取时得用 consumed，否则被扣住的那几个字节会被
        反复读进来，循环永不收敛。
        """

    @property
    def found(self) -> bool:
        return self._found

    def _partial_suffix_len(self, data: bytes) -> int:
        """data 的后缀里，最长的、同时是 marker 前缀的那一段有多长。"""
        limit = min(len(data), len(self._marker) - 1)
        for size in range(limit, 0, -1):
            if self._marker.startswith(data[-size:]):
                return size
        return 0

    def feed(self, chunk: bytes) -> bytes:
        """喂入原始字节，返回可以安全外发的正文字节。"""
        self.consumed += len(chunk)
        if self._found:
            self._tail += chunk
            return b""
        buffer = self._held + chunk
        self._held = b""
        index = buffer.find(self._marker)
        if index >= 0:
            self._found = True
            body = buffer[:index]
            self._tail = buffer[index + len(self._marker) :]
            self.offset += len(body)
            return body
        hold = self._partial_suffix_len(buffer)
        if hold:
            self._held = buffer[-hold:]
            buffer = buffer[:-hold]
        self.offset += len(buffer)
        return buffer

    def finish(self) -> bytes:
        """流结束：把扣住的字节放出来（它们最终没能凑成 marker）。"""
        if self._found:
            return b""
        held, self._held = self._held, b""
        self.offset += len(held)
        return held

    @property
    def payload(self) -> str:
        """哨兵后面那段载荷（首行）。没命中哨兵则为空串。"""
        if not self._found:
            return ""
        return self._tail.split(b"\n", 1)[0].decode("utf-8", errors="replace")


class StreamDecoder:
    """字节流 → 文本增量，负责 UTF-8 边界 + 哨兵剥离。"""

    def __init__(
        self,
        *,
        start_offset: int = 0,
        strip_state: bool = True,
        state_marker: bytes = STATE_MARKER,
    ) -> None:
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self._splitter = (
            SentinelSplitter(marker=state_marker, start_offset=start_offset)
            if strip_state
            else None
        )
        self._offset = start_offset
        self._consumed = start_offset

    def _buffered_utf8_bytes(self) -> int:
        buffered, _ = self._decoder.getstate()
        return len(buffered)

    @property
    def offset(self) -> int:
        """续读游标：只越过已经解成完整字符的字节。"""
        raw_offset = self._splitter.offset if self._splitter is not None else self._offset
        # 增量解码器可能扣着半个 UTF-8 字符。断线后必须把这些字节重新读一遍，否则会把一个
        # 汉字的前半截永远跳过去。
        return raw_offset - self._buffered_utf8_bytes()

    @property
    def consumed(self) -> int:
        """读取游标：已吃进来的字节末尾。从文件轮询读取时用它，否则会重复读扣住的字节。"""
        return self._splitter.consumed if self._splitter is not None else self._consumed

    @property
    def state_payload(self) -> str:
        return self._splitter.payload if self._splitter is not None else ""

    def feed(self, chunk: bytes) -> str:
        if self._splitter is not None:
            body = self._splitter.feed(chunk)
        else:
            body = chunk
            self._offset += len(chunk)
            self._consumed += len(chunk)
        if not body:
            return ""
        return self._decoder.decode(body, False)

    def finish(self) -> str:
        body = self._splitter.finish() if self._splitter is not None else b""
        return self._decoder.decode(body, True)


class CappedText:
    """带上限的文本累加器：超限只留 head + tail，中间用一行标注替换。"""

    def __init__(self, max_bytes: int = DEFAULT_MAX_OUTPUT_BYTES) -> None:
        self.max_bytes = max(1024, int(max_bytes))
        self._head: list[str] = []
        self._head_size = 0
        self._head_closed = False
        self._tail: list[str] = []
        self._tail_size = 0
        self._dropped = 0
        self._half = self.max_bytes // 2

    @property
    def truncated(self) -> bool:
        return self._dropped > 0

    def append(self, text: str) -> None:
        if not text:
            return
        size = _text_size(text)
        if not self._head_closed and self._head_size < self._half:
            head, text = _split_prefix(text, self._half - self._head_size)
            if head:
                self._head.append(head)
                self._head_size += _text_size(head)
            if not text:
                return
            self._head_closed = True
            size = _text_size(text)
        self._tail.append(text)
        self._tail_size += size
        while self._tail_size > self._half and self._tail:
            excess = self._tail_size - self._half
            first = self._tail[0]
            first_size = _text_size(first)
            if first_size <= excess:
                self._tail.pop(0)
                self._tail_size -= first_size
                self._dropped += first_size
                continue
            kept, dropped_size = _drop_prefix(first, excess)
            self._tail[0] = kept
            self._tail_size -= dropped_size
            self._dropped += dropped_size
            break

    def value(self) -> str:
        head = "".join(self._head)
        if not self._dropped:
            return head + "".join(self._tail)
        notice = (
            f"\n\n[remote-relay] …输出过长，中间省略约 {self._dropped} 字节"
            f"（上限 {self.max_bytes} 字节）。完整日志在远端，可用 download 取回。…\n\n"
        )
        return head + notice + "".join(self._tail)


def _text_size(text: str) -> int:
    return len(text.encode("utf-8", errors="replace"))


def _split_prefix(text: str, budget: int) -> tuple[str, str]:
    """按 UTF-8 字节预算切前缀，永远不劈开字符。"""
    if budget <= 0:
        return "", text
    if _text_size(text) <= budget:
        return text, ""
    low, high = 0, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _text_size(text[:middle]) <= budget:
            low = middle
        else:
            high = middle - 1
    return text[:low], text[low:]


def _drop_prefix(text: str, minimum_bytes: int) -> tuple[str, int]:
    """从头至少丢 minimum_bytes，并返回剩余文本与实际丢弃字节数。"""
    if minimum_bytes <= 0:
        return text, 0
    low, high = 0, len(text)
    while low < high:
        middle = (low + high) // 2
        if _text_size(text[:middle]) < minimum_bytes:
            low = middle + 1
        else:
            high = middle
    dropped = _text_size(text[:low])
    return text[low:], dropped
