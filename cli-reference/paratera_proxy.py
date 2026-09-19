#!/usr/bin/env python3
"""Logging pass-through for the bare-run arm: Codex (OpenAI chat or responses wire) -> https://llmapi.paratera.com.

The request body is forwarded UNCHANGED by default (thinking stays at the provider's default, which is on - the
bam three-way of 2026-09-15). With PROXY_THINKING=disabled the proxy sets `thinking: {"type": "disabled"}` on every
JSON body (the only form Paratera honours; `enable_thinking:false` is ignored) and logs `injected` - the knob of the
V4-Flash thinking-off caliber the DeepCode and DeepEvol arms run with since 2026-09-17; nothing else in the body is
touched. One JSON line per request goes to the log: path, model, stream, the thinking fields as sent, and the usage
block of the response (prompt/completion/reasoning tokens), so the run's model and thinking state are on record
(`reasoning_tokens` must be 0 on every line of a thinking-off run). On the Anthropic Messages wire (Claude Code,
`/v1/messages`) usage carries no reasoning counter; thinking appears as content blocks, so each such line also
records `thinking_blocks` (documents: blocks of type thinking / redacted_thinking; streams: content_block_start
events of that type) — the same must-be-0 audit. The key is never logged.

Usage:  [PROXY_THINKING=disabled] [PROXY_UPSTREAM_KEY_ENV=PARATERA_API_KEY] python3 paratera_proxy.py [port] [logfile]
        (default 8787, ./proxy_requests.log). With PROXY_UPSTREAM_KEY_ENV the proxy authenticates upstream itself
        (Authorization + x-api-key from that variable, logged as auth=proxy:<var>); the CLI may then hold any placeholder.
Codex:  ~/.codex/config.toml  model_providers.<name>.base_url = "http://127.0.0.1:8787" (keep the wire_api and the
        rest of the provider block as they are; the path is forwarded untouched)."""
import http.server, json, os, sys, time, urllib.request, urllib.error

THINKING = os.environ.get("PROXY_THINKING", "")  # "" = pass through; "disabled" = force thinking off
#: name of the environment variable holding the upstream key; when set, the proxy replaces the client's
#: Authorization / x-api-key headers with it, so the CLI's own auth config (cc-switch profiles, config.toml tokens,
#: ANTHROPIC_AUTH_TOKEN) can hold a placeholder and the real key lives only in the env file sourced for the proxy
FORCE_MODEL = os.environ.get("PROXY_FORCE_MODEL", "")  # rewrite every request's model to this id (desktop apps choose their own); logs model_in
USER_AGENT = os.environ.get("PROXY_USER_AGENT", "")  # e.g. "paperbench-bare/1"; replaces the client's User-Agent upstream
UPSTREAM_KEY_ENV = os.environ.get("PROXY_UPSTREAM_KEY_ENV", "")
UPSTREAM_KEY = os.environ.get(UPSTREAM_KEY_ENV, "") if UPSTREAM_KEY_ENV else ""
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
LOG = sys.argv[2] if len(sys.argv) > 2 else "proxy_requests.log"
UPSTREAM = os.environ.get("PROXY_UPSTREAM", "https://llmapi.paratera.com").rstrip("/")  # e.g. https://api.deepseek.com


def usage_of(chunk):
    """The usage block of a completion document, a Responses-API document/event (`response.usage`), an Anthropic
    Messages document or its `message_start` / `message_delta` events, or an SSE chunk, flattened: reasoning tokens
    live under completion_tokens_details (chat) or output_tokens_details (responses) on OpenAI-compatible routes;
    the Anthropic wire has no such counter (see thinking_blocks_of)."""
    u = chunk.get("usage") or (chunk.get("response") or {}).get("usage") or (chunk.get("message") or {}).get("usage") or {}
    if not u:
        return None
    details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {"prompt_tokens": u.get("prompt_tokens", u.get("input_tokens")),
            "completion_tokens": u.get("completion_tokens", u.get("output_tokens")),
            "reasoning_tokens": u.get("reasoning_tokens", details.get("reasoning_tokens"))}


def merge_usage(acc, u):
    """Anthropic streams split the usage over message_start (input) and message_delta (output); keep the non-None parts."""
    if u is None:
        return acc
    acc = dict(acc or {})
    for k, v in u.items():
        if v is not None:
            acc[k] = v
    return acc


THINKING_BLOCK_TYPES = ("thinking", "redacted_thinking")


def reasoning_items_of(chunk):
    """Responses wire (Codex): reasoning shows up as output items of type `reasoning` (documents) or as
    `response.reasoning*` events (streams: reasoning_summary_text.delta …) besides usage.reasoning_tokens; the count
    is the second, usage-independent thinking-off evidence on that wire."""
    t = str(chunk.get("type") or "")
    if t.startswith("response.reasoning"):
        return 1
    if t == "response.output_item.added" and (chunk.get("item") or {}).get("type") == "reasoning":
        return 1
    output = chunk.get("output")
    if isinstance(output, list):
        return sum(1 for o in output if isinstance(o, dict) and o.get("type") == "reasoning")
    return 0


def thinking_blocks_of(chunk):
    """Anthropic Messages: thinking shows up as content blocks, not in usage — a `content` array with blocks of
    type thinking / redacted_thinking (documents) or a `content_block_start` event whose block has that type (streams).
    This count is the thinking-off evidence on that wire (reasoning_tokens is on the OpenAI wires)."""
    if chunk.get("type") == "content_block_start":
        return 1 if (chunk.get("content_block") or {}).get("type") in THINKING_BLOCK_TYPES else 0
    content = chunk.get("content")
    if isinstance(content, list):
        return sum(1 for b in content if isinstance(b, dict) and b.get("type") in THINKING_BLOCK_TYPES)
    return 0


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep stderr quiet; the JSON log is the record
        pass

    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "path": self.path}
        try:
            d = json.loads(body)
            rec.update({"model": d.get("model"), "stream": d.get("stream"), "max_tokens": d.get("max_tokens"),
                        "thinking_fields": {k: d[k] for k in ("thinking", "enable_thinking", "reasoning_effort", "reasoning", "reasoning_effort_level") if k in d}})
            if os.environ.get("PROXY_LOG_KEYS") == "1":  # debugging a client's request shape: keys only, never content
                rec["body_keys"] = sorted(d.keys())
                rec["include"] = d.get("include")
                rec["stream"] = d.get("stream")
            dump_dir = os.environ.get("PROXY_DUMP_DIR")  # debugging only: full request bodies (prompts, no key) to files
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                stem = os.path.join(dump_dir, f"{time.strftime('%H%M%S')}_{os.getpid()}_{id(self)}")
                with open(stem + ".json", "wb") as fh:
                    fh.write(body)
                with open(stem + ".headers.json", "w") as fh:  # request headers, auth values redacted
                    json.dump({k: ("…" if k.lower() in ("authorization", "x-api-key") else v) for k, v in self.headers.items()}, fh, indent=1)
            if FORCE_MODEL and d.get("model") != FORCE_MODEL:
                rec["model_in"] = d.get("model")
                d["model"] = FORCE_MODEL
                rec["model"] = FORCE_MODEL
                body = json.dumps(d).encode("utf-8")
            if THINKING == "disabled":
                if self.path.split("?", 1)[0].rstrip("/").endswith("/responses"):
                    # Responses API: the documented off switch is reasoning.effort=none (api-docs.deepseek.com, thinking_mode)
                    d["reasoning"] = {**(d.get("reasoning") or {}), "effort": "none"}
                    rec["injected"] = {"reasoning": d["reasoning"]}
                else:
                    d["thinking"] = {"type": "disabled"}  # chat completions and Anthropic Messages
                    rec["injected"] = {"thinking": d["thinking"]}
                body = json.dumps(d).encode("utf-8")
        except Exception:
            rec["note"] = "non-json body passed through"
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() in ("host", "content-length", "accept-encoding"):
                continue
            if UPSTREAM_KEY and k.lower() in ("authorization", "x-api-key"):
                continue  # replaced below
            if USER_AGENT and (k.lower() == "user-agent" or k.lower().startswith("x-codex-")):
                continue  # replaced / dropped below: DeepSeek keys its Codex profile on either
            req.add_header(k, v)
        if USER_AGENT:
            # DeepSeek serves a Codex-specific profile keyed on Codex's User-Agent (codex_exec/…) OR its x-codex-turn-metadata
            # header: reasoning.effort=none is then ignored and thinking stays on (measured 2026-09-19 with the identical
            # body: plain curl → 0 reasoning tokens; + UA codex_exec/0.155.1 → 14; + x-codex-turn-metadata alone → 14; the
            # other x-codex-* / session-id / thread-id / originator headers alone → 0). A neutral UA and no x-codex-*
            # headers restore the documented behaviour.
            req.add_header("user-agent", USER_AGENT)
            rec["user_agent"] = USER_AGENT
        if UPSTREAM_KEY:
            req.add_header("authorization", f"Bearer {UPSTREAM_KEY}")
            req.add_header("x-api-key", UPSTREAM_KEY)
            rec["auth"] = f"proxy:{UPSTREAM_KEY_ENV}"
        req.add_header("content-length", str(len(body)))
        usage = None
        route = self.path.split("?", 1)[0].rstrip("/")  # Claude Code appends ?beta=true
        anthropic = route.endswith("/messages")  # Claude Code's wire: /v1/messages
        responses = route.endswith("/responses")  # Codex's wire: /v1/responses
        thinking_blocks = 0 if anthropic else None
        reasoning_items = 0 if responses else None
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "content-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                streaming = (r.headers.get("content-type") or "").startswith("text/event-stream")
                collected = b""
                while True:
                    chunk = r.read(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
                    collected += chunk
                try:
                    if streaming:
                        for line in collected.decode("utf-8", "replace").splitlines():
                            if line.startswith("data: ") and line[6:].strip() not in ("", "[DONE]"):
                                ev = json.loads(line[6:])
                                usage = merge_usage(usage, usage_of(ev))
                                if anthropic:
                                    thinking_blocks += thinking_blocks_of(ev)
                                if responses:
                                    reasoning_items += reasoning_items_of(ev)
                    else:
                        doc = json.loads(collected)
                        usage = usage_of(doc)
                        if anthropic:
                            thinking_blocks = thinking_blocks_of(doc)
                        if responses:
                            reasoning_items = reasoning_items_of(doc)
                except Exception as e:
                    rec["usage_parse_error"] = str(e)
                rec["status"] = r.status
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            self.end_headers()
            self.wfile.write(e.read())
            rec["status"] = e.code
        except Exception as e:
            self.send_response(502)
            self.end_headers()
            rec["status"] = 502
            rec["error"] = str(e)
        rec["usage"] = usage
        if anthropic:
            rec["thinking_blocks"] = thinking_blocks  # must be 0 on every line of a thinking-off run
        if responses:
            rec["reasoning_items"] = reasoning_items  # same audit on Codex's wire (reasoning output items / events)
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print(f"paratera pass-through on http://127.0.0.1:{PORT} -> {UPSTREAM}; log: {LOG}; thinking: {THINKING or 'as sent (provider default = on)'}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
