#!/usr/bin/env python3
"""Logging pass-through for the bare-run arm: Codex (OpenAI chat or responses wire) -> https://llmapi.paratera.com.

The request body is forwarded UNCHANGED (thinking stays at the provider's default, which is on - the same setting
the DeepCode and DeepEvol arms run with). One JSON line per request goes to the log: path, model, stream, and the
usage block of the response (prompt/completion/reasoning tokens), so the run's model and thinking state are on
record. The key is never logged.

Usage:  python3 paratera_proxy.py [port] [logfile]      (default 8787, ./proxy_requests.log)
Codex:  ~/.codex/config.toml  model_providers.<name>.base_url = "http://127.0.0.1:8787" (keep the wire_api and the
        rest of the provider block as they are; the path is forwarded untouched)."""
import http.server, json, sys, time, urllib.request, urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
LOG = sys.argv[2] if len(sys.argv) > 2 else "proxy_requests.log"
UPSTREAM = "https://llmapi.paratera.com"


def usage_of(chunk):
    """The usage block of a completion document, a Responses-API document/event (`response.usage`), or an SSE
    chunk, flattened: reasoning tokens live under completion_tokens_details (chat) or output_tokens_details
    (responses) on OpenAI-compatible routes."""
    u = chunk.get("usage") or (chunk.get("response") or {}).get("usage") or {}
    if not u:
        return None
    details = u.get("completion_tokens_details") or u.get("output_tokens_details") or {}
    return {"prompt_tokens": u.get("prompt_tokens", u.get("input_tokens")),
            "completion_tokens": u.get("completion_tokens", u.get("output_tokens")),
            "reasoning_tokens": u.get("reasoning_tokens", details.get("reasoning_tokens"))}


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
        except Exception:
            rec["note"] = "non-json body passed through"
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "accept-encoding"):
                req.add_header(k, v)
        req.add_header("content-length", str(len(body)))
        usage = None
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
                                u = usage_of(json.loads(line[6:]))
                                if u:
                                    usage = u
                    else:
                        usage = usage_of(json.loads(collected))
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
        with open(LOG, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    print(f"paratera pass-through on http://127.0.0.1:{PORT} -> {UPSTREAM}; log: {LOG}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
