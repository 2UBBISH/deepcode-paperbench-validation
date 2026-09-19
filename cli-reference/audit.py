#!/usr/bin/env python3
"""Caliber audit of one bare-arm run from its proxy log:  python3 audit.py <workspace> <codex|claude> [model]
Prints one summary line and CALIBER_OK / CALIBER_BROKEN: … ; exit 1 when broken."""
import json, sys

ws, arm = sys.argv[1], sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else "deepseek-flash"
rows = [json.loads(l) for l in open(f"{ws}/proxy_requests.log") if l.strip()]
bad = []
if not rows:
    bad.append("no requests went through the proxy")
models = {r.get("model") for r in rows}
if rows and models != {model}: bad.append(f"model(s) {sorted(str(m) for m in models)} != {model}")
if not all(r.get("injected") for r in rows): bad.append("a request without the thinking injection")
if not all((r.get("auth") or "").startswith("proxy:") for r in rows): bad.append("a request not authenticated by the proxy")
if arm == "codex" and not all(r.get("user_agent") for r in rows): bad.append("a Codex request without the User-Agent replacement (DeepSeek would force thinking on)")
key = "thinking_blocks" if arm == "claude" else "reasoning_items"
over = [r for r in rows if ((r.get("usage") or {}).get("reasoning_tokens") or 0) > 0 or (r.get(key) or 0) > 0]
if over: bad.append(f"{len(over)} responses with reasoning ({key} / reasoning_tokens > 0)")
errs = [r for r in rows if r.get("status") != 200]
tok = (sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in rows), sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in rows))
model_in = sorted({str(r.get("model_in")) for r in rows if r.get("model_in")})
print(f"requests {len(rows)}  model {sorted(str(m) for m in models)}{' (client asked for ' + ', '.join(model_in) + ')' if model_in else ''}  non-200 {len(errs)}  prompt/completion tokens {tok[0]}/{tok[1]}")
print("CALIBER_OK" if not bad else "CALIBER_BROKEN: " + "; ".join(bad))
sys.exit(1 if bad else 0)
