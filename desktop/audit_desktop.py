#!/usr/bin/env python3
"""Caliber audit of a desktop-app run from the app's OWN session logs (no proxy in the loop):

    python3 audit_desktop.py <codex|claude> <workspace> <start-epoch> [model] [--copy DIR]

Codex app / CLI write ~/.codex/sessions/YYYY/MM/DD/rollout-*.jsonl (session_meta.cwd, turn_context.model,
event_msg token_count → total_token_usage incl. reasoning_output_tokens); Claude desktop's Code tab / claude CLI write
~/.claude/projects/<cwd with '/' and '.' → '-'>/<session>.jsonl (assistant messages: message.model, usage incl.
output_tokens_details.thinking_tokens, content blocks). Sessions are matched by cwd == workspace and mtime ≥ start.
Caliber of the 0919 batch: model == deepseek-flash on every turn, thinking ON (reasoning / thinking tokens > 0 overall),
execution rule (09-20 evening): commands are allowed; only long CPU / GPU training or evaluation runs are not (the prompt
says they run remotely later) — nothing is blocked. Every interpreter / installer command that RAN is listed with its wall
time; one command over MAX_SINGLE_S or a total over MAX_TOTAL_S = it ran a long experiment → BROKEN; training-looking
commands are flagged for the owner; anything that ran at all → REVIEW.
Prints a summary and CALIBER_OK / CALIBER_REVIEW / CALIBER_BROKEN: …; --copy DIR copies the matched session files there."""
import datetime, glob, json, os, re, shutil, sys, time

EXEC_PATTERN = re.compile(r"(^|[\s;&|(])(python[0-9.]*|pytest|pip[0-9]?|uv|conda|node|npm|bash|sh|zsh|make|docker|wget|curl|\./[\w./-]+)(\s|$)")
EXPERIMENT_PATTERN = re.compile(r"(train|eval|benchmark|experiment|run_all|sweep|epochs?|\.ckpt\b)", re.I)  # hints only, never fatal
MAX_SINGLE_S, MAX_TOTAL_S = 600, 3600   # only LONG training / evaluation is out: 10 min for one command / 60 min of execution in total
INSTALL_PATTERN = re.compile(r"(^|\s)(pip[0-9]?|uv|conda)\s+(install|sync|add)")  # installs are exempt from the single-command clock

def _ts(v):
    try: return datetime.datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception: return None

arm, ws = sys.argv[1], os.path.realpath(sys.argv[2])
start = float(sys.argv[3])
model = sys.argv[4] if len(sys.argv) > 4 and not sys.argv[4].startswith("--") else "deepseek-flash"
copy_dir = sys.argv[sys.argv.index("--copy") + 1] if "--copy" in sys.argv else None
bad, files, models, turns, out_tokens, think_tokens = [], [], set(), 0, 0, 0
executed: list[tuple[str, float]] = []  # (command, wall seconds) for every interpreter / installer / shell command that actually ran
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
                try: cmds[p.get("call_id")] = (json.loads(p.get("arguments") or "{}").get("cmd") or "", _ts(e.get("timestamp")))
                except Exception: pass
            elif e.get("type") == "response_item" and p.get("type") == "function_call_output" and p.get("call_id") in cmds:
                cmd, t0 = cmds.pop(p["call_id"]); out = str(p.get("output") or "")[:400]
                if "Rejected" in out or "blocked by policy" in out or "not approved" in out.lower(): rejected += 1
                elif EXEC_PATTERN.search(cmd.replace("/bin/zsh -lc", "").replace("/bin/bash -lc", "").strip(" '\"")):
                    m = re.search(r"Wall time: ([0-9.]+) seconds", out); t1 = _ts(e.get("timestamp"))
                    ran.append((cmd, float(m.group(1)) if m else ((t1 - t0) if t0 and t1 else 0.0)))
        if not (meta_cwd and same_dir(meta_cwd, ws)): continue
        executed += ran; blocked += rejected
        files.append(f); models |= mod; turns += 1
        if last: out_tokens += int(last.get("output_tokens") or 0); think_tokens += int(last.get("reasoning_output_tokens") or 0)
else:
    key = ws.replace("/", "-").replace(".", "-")
    for f in sorted(glob.glob(os.path.expanduser(f"~/.claude/projects/{key}/*.jsonl"))):
        if os.path.getmtime(f) < start: continue
        n = 0; pending = {}
        for line in open(f, encoding="utf-8", errors="replace"):
            try: e = json.loads(line)
            except Exception: continue
            if e.get("type") == "user":
                for c in ((e.get("message") or {}).get("content") or []) if isinstance((e.get("message") or {}).get("content"), list) else []:
                    if isinstance(c, dict) and c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                        cmd, t0 = pending.pop(c["tool_use_id"]); body = c.get("content"); body = body if isinstance(body, str) else json.dumps(body)[:400]
                        if "doesn't want to proceed" in body or "was rejected" in body or "permission" in body.lower()[:200]: blocked += 1
                        else:
                            t1 = _ts(e.get("timestamp")); executed.append((cmd, (t1 - t0) if t0 and t1 else 0.0))
                continue
            if e.get("type") != "assistant": continue
            m = e.get("message") or {}; u = m.get("usage") or {}
            models.add(m.get("model")); n += 1
            for c in m.get("content", []):
                if isinstance(c, dict) and c.get("type") == "tool_use" and c.get("name") == "Bash":
                    pending[c.get("id")] = (str((c.get("input") or {}).get("command") or "")[:160], _ts(e.get("timestamp")))
            out_tokens += int(u.get("output_tokens") or 0)
            think_tokens += int((u.get("output_tokens_details") or {}).get("thinking_tokens") or 0)
            think_tokens += sum(len(c.get("thinking") or "") // 4 for c in m.get("content", []) if isinstance(c, dict) and c.get("type") == "thinking" and not (u.get("output_tokens_details") or {}).get("thinking_tokens"))
        if n: files.append(f); turns += n

if not files: bad.append(f"no {arm} session with cwd {ws} modified since {time.strftime('%H:%M:%S', time.localtime(start))}")
if files and models != {model}: bad.append(f"model(s) {sorted(str(m) for m in models)} != {model}")
if files and think_tokens == 0: bad.append("no reasoning/thinking tokens at all — thinking appears OFF (caliber is ON)")
# Execution (09-20 evening rule): commands allowed, long training / evaluation not; nothing is gated. What ran is listed
# with wall time; over the clock → BROKEN (it ran a long experiment); training-looking commands flagged; anything ran → REVIEW.
review = []
if blocked: print(f"  ({blocked} command(s) rejected / not run — fine)")
total_s = sum(t for _, t in executed)
looks = 0
for cmd, t in executed:
    head = (cmd.strip().splitlines() or [""])[0][:110]; tag = []
    if EXPERIMENT_PATTERN.search(cmd): tag.append("training-looking"); looks += 1
    if t > MAX_SINGLE_S and not INSTALL_PATTERN.search(cmd): tag.append(f"over {MAX_SINGLE_S}s"); bad.append(f"over {MAX_SINGLE_S}s: {head[:60]} ({t:.0f}s)")
    print(f"  ran {t:7.1f}s  {head}" + (f"   ⚠️ {', '.join(tag)}" if tag else ""))
if total_s > MAX_TOTAL_S: bad.append(f"execution total {total_s/60:.0f} min > {MAX_TOTAL_S//60} min")
if executed and not bad: review.append(f"{len(executed)} command(s) ran ({total_s:.0f}s total{f', {looks} training-looking by keyword' if looks else ''}) — owner to glance at the list; none exceeded the long-experiment line")
print(f"{arm}: {len(files)} session file(s), {turns} turn(s), models {sorted(str(m) for m in models)}, output tokens {out_tokens}, thinking tokens {think_tokens}")
for f in files: print("  " + f)
if copy_dir and files:
    os.makedirs(copy_dir, exist_ok=True)
    for f in files: shutil.copy2(f, copy_dir)
print("CALIBER_BROKEN: " + "; ".join(bad) if bad else ("CALIBER_REVIEW: " + "; ".join(review) if review else "CALIBER_OK"))
sys.exit(1 if bad else 0)
