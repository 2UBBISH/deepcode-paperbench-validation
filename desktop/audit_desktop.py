#!/usr/bin/env python3
"""Caliber audit of a desktop-app run from the app's OWN session logs (no proxy in the loop):

    python3 audit_desktop.py <codex|claude> <workspace> <start-epoch> [model] [--copy DIR]

Codex app / CLI write ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl (session_meta.cwd, turn_context.model,
event_msg token_count → total_token_usage incl. reasoning_output_tokens); Claude desktop's Code tab / claude CLI write
~/.claude/projects/<cwd with '/' and '.' → '-'>/<session>.jsonl (assistant messages: message.model, usage incl.
output_tokens_details.thinking_tokens, content blocks). Sessions are matched by cwd == workspace and mtime ≥ start.
Caliber of the 0919 batch: model == deepseek-flash on every turn, thinking ON (reasoning / thinking tokens > 0 overall).
Prints a summary line and CALIBER_OK / CALIBER_BROKEN: …; --copy DIR copies the matched session files there."""
import glob, json, os, re, shutil, sys, time

EXEC_PATTERN = re.compile(r"(^|[\s;&|(])(python[0-9.]*|pytest|pip[0-9]?|uv|conda|node|npm|bash|sh|zsh|make|docker|\./[\w./-]+)(\s|$)")

arm, ws = sys.argv[1], os.path.realpath(sys.argv[2])
start = float(sys.argv[3])
model = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "deepseek-flash"
copy_dir = sys.argv[sys.argv.index("--copy") + 1] if "--copy" in sys.argv else None
bad, files, models, turns, out_tokens, think_tokens = [], [], set(), 0, 0, 0
executed: list[str] = []  # commands that actually ran an interpreter / build tool (the batch forbids execution)
blocked = 0

def same_dir(a, b):
    try: return os.path.realpath(a) == b
    except Exception: return False

if arm == "codex":
    for f in sorted(glob.glob(os.path.expanduser("~/.codex/sessions/*/*/*/rollout-*.jsonl"))):
        if os.path.getmtime(f) < start: continue
        meta_cwd, mod, last, cmds, ran, rejected = None, set(), None, {}, [], 0
        for line in open(f, encoding="utf-8", errors="replace"):
            try: e = json.loads(line)
            except Exception: continue
            p = e.get("payload") or {}
            if e.get("type") == "session_meta": meta_cwd = p.get("cwd")
            elif e.get("type") == "turn_context": mod.add(p.get("model"))
            elif p.get("type") == "token_count" and (p.get("info") or {}).get("total_token_usage"): last = p["info"]["total_token_usage"]
            elif e.get("type") == "response_item" and p.get("type") == "function_call" and p.get("name") == "exec_command":
                try: cmds[p.get("call_id")] = json.loads(p.get("arguments") or "{}").get("cmd") or ""
                except Exception: pass
            elif e.get("type") == "response_item" and p.get("type") == "function_call_output" and p.get("call_id") in cmds:
                cmd = cmds.pop(p["call_id"]); out = str(p.get("output") or "")[:300]
                if "Rejected" in out or "blocked by policy" in out: rejected += 1
                elif EXEC_PATTERN.search(cmd.replace("/bin/zsh -lc", "").replace("/bin/bash -lc", "").strip(" '\"")): ran.append(cmd)
        if not (meta_cwd and same_dir(meta_cwd, ws)): continue
        executed += ran; blocked += rejected
        files.append(f); models |= mod; turns += 1
        if last: out_tokens += int(last.get("output_tokens") or 0); think_tokens += int(last.get("reasoning_output_tokens") or 0)
else:
    key = ws.replace("/", "-").replace(".", "-")
    for f in sorted(glob.glob(os.path.expanduser(f"~/.claude/projects/{key}/*.jsonl"))):
        if os.path.getmtime(f) < start: continue
        n = 0
        for line in open(f, encoding="utf-8", errors="replace"):
            try: e = json.loads(line)
            except Exception: continue
            if e.get("type") != "assistant": continue
            m = e.get("message") or {}; u = m.get("usage") or {}
            models.add(m.get("model")); n += 1
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Bash":
                    executed.append(str((c.get("input") or {}).get("command") or "")[:120])
            out_tokens += int(u.get("output_tokens") or 0)
            think_tokens += int((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
            think_tokens += sum(len(c.get("thinking") or "") // 4 for c in m.get("content", []) if isinstance(c, dict) and c.get("type") == "thinking" and not (u.get("output_tokens_details") or {}).get("thinking_tokens"))
        if n: files.append(f); turns += n

if not files: bad.append(f"no {arm} session with cwd {ws} modified since {time.strftime('%H:%M:%S', time.localtime(start))}")
if files and models != {model}: bad.append(f"model(s) {sorted(str(m) for m in models)} != {model}")
if files and think_tokens == 0: bad.append("no reasoning/thinking tokens at all — thinking appears OFF (caliber is ON)")
# Codex: a forbidden command is attempted, rejected by the rules file and answered "Rejected: …" — that is fine; what breaks
# the caliber is a command that ran. The rejected ones are still listed so a sudden spike is visible.
if arm == "codex":
    if blocked: print(f"  ({blocked} execution attempt(s) rejected by the rules file — fine)")
    if executed: bad.append(f"{len(executed)} interpreter/build command(s) RAN — execution is forbidden in this batch: " + "; ".join(c[:60] for c in executed[:3]))
elif executed: bad.append(f"{len(executed)} Bash tool call(s) — execution is forbidden in this batch: " + "; ".join(executed[:3]))
print(f"{arm}: {len(files)} session file(s), {turns} turn(s), models {sorted(str(m) for m in models)}, output tokens {out_tokens}, thinking tokens {think_tokens}")
for f in files: print("  " + f)
if copy_dir and files:
    os.makedirs(copy_dir, exist_ok=True)
    for f in files: shutil.copy2(f, copy_dir)
print("CALIBER_OK" if not bad else "CALIBER_BROKEN: " + "; ".join(bad))
sys.exit(1 if bad else 0)
