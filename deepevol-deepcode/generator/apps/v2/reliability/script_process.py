"""Bounded POSIX script execution; no request threads remain after return."""

import math
import os
import selectors
import signal
import subprocess
import time


class ScriptLimitExceeded(RuntimeError):
    pass


def run_script(args, *, cwd, env, timeout, maximum_output_bytes=4 * 1024 * 1024):
    if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Script timeout must be positive and finite")
    if os.name != "posix":
        raise ScriptLimitExceeded("Bounded script process groups require POSIX")
    deadline = time.monotonic() + timeout
    process = subprocess.Popen(args, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    outputs = {"stdout": bytearray(), "stderr": bytearray()}
    total = 0
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ, "stdout")
            selector.register(process.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ScriptLimitExceeded("Script exceeded its time limit")
                for key, _ in selector.select(timeout=remaining):
                    chunk = os.read(key.fileobj.fileno(), 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > maximum_output_bytes:
                        raise ScriptLimitExceeded("Script exceeded its output limit")
                    outputs[key.data].extend(chunk)
        try:
            code = process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            raise ScriptLimitExceeded("Script exceeded its time limit") from exc
        return subprocess.CompletedProcess(args, code, outputs["stdout"].decode("utf-8", errors="replace"), outputs["stderr"].decode("utf-8", errors="replace"))
    finally:
        # Kill the group even when its leader has exited but descendants still
        # hold pipes or continue work. Only this invocation's session is used.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        finally:
            process.wait()
            process.stdout.close()
            process.stderr.close()
