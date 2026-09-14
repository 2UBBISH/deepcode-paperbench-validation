#!/usr/bin/env python3
"""Local proxy for the bare-run arm: Claude Code -> https://llmapi.paratera.com (Anthropic Messages format),
with thinking FORCED OFF on every request (the same setting DeepCode and the DeepEvol line ran with), and a
one-line log per request (path, model, thinking, max_tokens, stream) so the run's settings are on record.

Usage:  python3 paratera_proxy.py [port] [logfile]      (default 8787, ./proxy_requests.log)
Then point Claude Code at it: ANTHROPIC_BASE_URL=http://127.0.0.1:8787 (key stays in ANTHROPIC_AUTH_TOKEN).
The key is never logged."""
import http.server, json, sys, urllib.request, urllib.error

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8787
LOG = sys.argv[2] if len(sys.argv) > 2 else "proxy_requests.log"
UPSTREAM = "https://llmapi.paratera.com"


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        body = self.rfile.read(int(self.headers.get("content-length", 0)))
        try:
            d = json.loads(body)
            d["thinking"] = {"type": "disabled"}  # the whole point of this proxy
            body = json.dumps(d).encode()
            rec = {"path": self.path, "model": d.get("model"), "thinking": d["thinking"], "max_tokens": d.get("max_tokens"), "stream": d.get("stream")}
        except Exception:
            rec = {"path": self.path, "note": "non-json body passed through"}
        with open(LOG, "a") as f:
            f.write(json.dumps(rec) + "\n")
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method="POST")
        for k, v in self.headers.items():
            if k.lower() not in ("host", "content-length", "accept-encoding"):
                req.add_header(k, v)
        req.add_header("content-length", str(len(body)))
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                self.send_response(r.status)
                for k, v in r.headers.items():
                    if k.lower() not in ("transfer-encoding", "content-length", "content-encoding", "connection"):
                        self.send_header(k, v)
                self.end_headers()
                while True:
                    chunk = r.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk); self.wfile.flush()
        except urllib.error.HTTPError as e:
            data = e.read()
            self.send_response(e.code); self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(data))); self.end_headers(); self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print(f"paratera proxy on http://127.0.0.1:{PORT} -> {UPSTREAM}, thinking forced off, log {LOG}")
    http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
