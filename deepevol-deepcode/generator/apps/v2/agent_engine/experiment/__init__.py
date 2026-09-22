"""实验 Agent：代码 → 配置推荐 → 远端租赁 → 配环境跑实验 → 自动释放。

设计见 `docs/experiment-agent-design.md`。本包是 DeepEvol 自己的代码；
vendored 的 rsa / remote_relay / setupx 在 `Agent/` 顶层，本包只调用它们，不改它们。
"""

from .catalog import SkuOption, TierMatch, match_plan, match_tier, option_from_catalog
from .cpu_tiers import build_cpu_tiers, cpu_tier_requirements, filter_cpu_options
from .compute_spec import (
    apply_oom_measurement,
    ComputeSpec,
    Estimate,
    build_compute_spec,
    estimate_vram,
    params_from_config,
    params_from_name,
)
from .lease import (
    DEFAULT_READY_TIMEOUT,
    ExperimentLease,
    LeaseBackend,
    LeaseError,
    LeaseEvent,
    LeaseHandle,
    LeaseState,
)
from .release_policy import (
    HOLD_STATUSES,
    STILL_RUNNING,
    TERMINAL_RELEASE_STATUSES,
    Disposition,
    ReleaseDecision,
    decide,
)
from .oom import OomFact, detect_oom, parse_cuda_oom, required_vram_gib
from .preflight_catalog import (
    capability_for,
    catalog_from_preflight,
    load_gpu_capability,
)
from .flow import (
    ExperimentTurnResult,
    render_recommendation,
    run_experiment_first_turn,
)
from .recommend import Recommendation, recommend_from_repo
from .repo_source import (
    ALLOWED_HOSTS,
    MAX_REPO_MB,
    ClonedRepo,
    RepoSource,
    RepoSourceError,
    clone_for_analysis,
    parse_repo_url,
)
from .run_flow import (
    DEFAULT_HARD_CAP_SECONDS,
    ExperimentRunResult,
    build_rsa_config,
    run_experiment_on_machine,
    with_resume_hint,
)
from .git_daemon import (
    GIT_DAEMON_PORT,
    GitDaemonError,
    local_bundle,
    serve_repo_on_machine,
)
from .zip_source import (
    MAX_EXTRACTED_MB,
    MAX_MEMBER_MB,
    extract_zip_repo,
)
from .selection import Selection, parse_selection
from .repo_facts import Fact, ResourceFacts, analyse_resources
from .rsa_usage import RsaUsage, collect_rsa_usage
from .setupx_env import (
    LlmTarget,
    clear_stale_env_local,
    render_backend_env,
    resolve_llm_target,
    write_setupx_backend_envs,
)

from .upgrade_plan import (
    IMAGE_WORTH_IT_SECONDS,
    MEASURED_CREATE_IMAGE_SECONDS,
    UpgradeCandidate,
    UpgradeProposal,
    UpgradeStrategy,
    build_upgrade_proposal,
    choose_upgrade_strategy,
)
from .tiers import Tier, TierPlan, build_tiers, round_up_to_step

__all__ = [
    "DEFAULT_READY_TIMEOUT",
    "HOLD_STATUSES",
    "STILL_RUNNING",
    "TERMINAL_RELEASE_STATUSES",
    "ClonedRepo",
    "ComputeSpec",
    "Disposition",
    "Estimate",
    "GIT_DAEMON_PORT",
    "GitDaemonError",
    "DEFAULT_HARD_CAP_SECONDS",
    "ExperimentRunResult",
    "ExperimentTurnResult",
    "ExperimentLease",
    "Fact",
    "LeaseBackend",
    "LeaseError",
    "LeaseEvent",
    "LeaseHandle",
    "LeaseState",
    "LlmTarget",
    "OomFact",
    "ALLOWED_HOSTS",
    "MAX_EXTRACTED_MB",
    "MAX_MEMBER_MB",
    "MAX_REPO_MB",
    "Recommendation",
    "ReleaseDecision",
    "RepoSource",
    "RepoSourceError",
    "ResourceFacts",
    "RsaUsage",
    "Selection",
    "SkuOption",
    "Tier",
    "TierMatch",
    "TierPlan",
    "IMAGE_WORTH_IT_SECONDS",
    "MEASURED_CREATE_IMAGE_SECONDS",
    "UpgradeCandidate",
    "UpgradeProposal",
    "UpgradeStrategy",
    "analyse_resources",
    "apply_oom_measurement",
    "build_compute_spec",
    "build_cpu_tiers",
    "cpu_tier_requirements",
    "filter_cpu_options",
    "build_rsa_config",
    "build_tiers",
    "build_upgrade_proposal",
    "capability_for",
    "catalog_from_preflight",
    "choose_upgrade_strategy",
    "clear_stale_env_local",
    "clone_for_analysis",
    "collect_rsa_usage",
    "decide",
    "detect_oom",
    "estimate_vram",
    "extract_zip_repo",
    "load_gpu_capability",
    "local_bundle",
    "match_plan",
    "match_tier",
    "option_from_catalog",
    "params_from_config",
    "recommend_from_repo",
    "params_from_name",
    "parse_selection",
    "parse_repo_url",
    "parse_cuda_oom",
    "render_backend_env",
    "render_recommendation",
    "run_experiment_first_turn",
    "run_experiment_on_machine",
    "with_resume_hint",
    "serve_repo_on_machine",
    "required_vram_gib",
    "resolve_llm_target",
    "round_up_to_step",
    "write_setupx_backend_envs",
]
