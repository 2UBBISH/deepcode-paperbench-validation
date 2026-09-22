"""执行层：快路径命令、耐久作业、流式解码、粘性 shell 状态。"""

from .fast import run_fast
from .jobs import DEFAULT_JOBS_ROOT, JobManager, new_job_id
from .shellstate import DEFAULT_BOOTSTRAP, DEFAULT_REMOTE_ENV, ShellState
from .stream import CappedText, SentinelSplitter, StreamDecoder

__all__ = [
    "CappedText",
    "DEFAULT_BOOTSTRAP",
    "DEFAULT_JOBS_ROOT",
    "DEFAULT_REMOTE_ENV",
    "JobManager",
    "SentinelSplitter",
    "ShellState",
    "StreamDecoder",
    "new_job_id",
    "run_fast",
]
