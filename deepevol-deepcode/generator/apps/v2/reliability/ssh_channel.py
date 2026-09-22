"""Bounded SSH channel draining without serial stdout/stderr deadlocks."""
from contextlib import contextmanager
import math
import time
from threading import Event, Timer


class SSHOutputLimitExceeded(RuntimeError):
    pass


def drain_channel(channel, *, maximum: int, timeout_seconds: float) -> tuple[bytes, bytes]:
    if maximum < 1 or not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("invalid SSH read budget")
    deadline = time.monotonic() + timeout_seconds
    output, error = bytearray(), bytearray()
    while True:
        if time.monotonic() >= deadline:
            raise TimeoutError("SSH execution deadline exceeded")
        progressed = False
        # One bounded read per stream per iteration keeps both streams moving
        # and checks the deadline even when stdout is continuously ready.
        for ready, read, target in (
            (channel.recv_ready, channel.recv, output),
            (channel.recv_stderr_ready, channel.recv_stderr, error),
        ):
            if ready():
                block = read(min(65536, maximum + 1 - len(output) - len(error)))
                target.extend(block)
                progressed = progressed or bool(block)
                if len(output) + len(error) > maximum:
                    raise SSHOutputLimitExceeded("SSH output limit exceeded")
        if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
            return bytes(output), bytes(error)
        if not progressed:
            time.sleep(min(0.01, max(0, deadline - time.monotonic())))


def execute_command(client, command: str, *, maximum: int, timeout_seconds: float):
    """Close local I/O at deadline; never claim that the remote effect stopped."""
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("invalid SSH execution deadline")
    deadline = time.monotonic() + timeout_seconds
    channel = client.get_transport().open_session(timeout=timeout_seconds)
    expired = Event()

    def cancel():
        expired.set()
        client.close()

    timer = Timer(max(0, deadline - time.monotonic()), cancel)
    timer.daemon = True
    timer.start()
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("SSH execution deadline exceeded")
        channel.settimeout(remaining)
        channel.set_combine_stderr(False)
        channel.exec_command(command)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("SSH execution deadline exceeded")
        output, error = drain_channel(channel, maximum=maximum, timeout_seconds=remaining)
        exit_code = channel.recv_exit_status()
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("SSH execution deadline exceeded")
        return output, error, exit_code
    except Exception:
        if expired.is_set():
            raise TimeoutError("SSH execution deadline exceeded") from None
        raise
    finally:
        timer.cancel()
        channel.close()


@contextmanager
def connection_deadline(clients, *, timeout_seconds):
    """Bound a multi-connection operation; remote effects remain uncertain."""
    deadline = time.monotonic() + timeout_seconds
    expired = Event()

    def cancel():
        expired.set()
        for client in clients:
            try:
                client.close()
            except Exception:
                pass

    timer = Timer(timeout_seconds, cancel)
    timer.daemon = True
    timer.start()
    try:
        yield
        if expired.is_set() or time.monotonic() >= deadline:
            raise TimeoutError("SSH transfer deadline exceeded")
    except Exception:
        if expired.is_set():
            raise TimeoutError("SSH transfer deadline exceeded") from None
        raise
    finally:
        timer.cancel()
