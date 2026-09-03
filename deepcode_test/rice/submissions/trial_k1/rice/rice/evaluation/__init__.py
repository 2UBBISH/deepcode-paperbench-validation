"""RICE evaluation package.

This module aggregates the evaluation utilities used throughout the RICE
reproduction: standardized policy rollouts, explanation fidelity benchmarks,
training-efficiency benchmarks, and visualization helpers.
"""

try:
    from rice.evaluation.evaluate_policy import (
        compare_policies,
        evaluate_policy,
        evaluate_policy_from_domain,
        log_evaluation,
    )
except Exception:  # pragma: no cover - optional dependency guard
    evaluate_policy = None  # type: ignore
    evaluate_policy_from_domain = None  # type: ignore
    compare_policies = None  # type: ignore
    log_evaluation = None  # type: ignore

try:
    from rice.evaluation.fidelity import (
        compare_fidelity,
        compute_airs_fidelity,
        compute_fidelity_score,
        compute_ig_fidelity,
        compute_rice_fidelity,
        compute_random_fidelity,
        compute_statemask_fidelity,
        fidelity_from_domain,
        log_fidelity_table,
        rank_airs,
        rank_integrated_gradients,
        rank_rice,
        rank_random,
        rank_statemask,
    )
except Exception:  # pragma: no cover - optional dependency guard
    compare_fidelity = None  # type: ignore
    compute_airs_fidelity = None  # type: ignore
    compute_fidelity_score = None  # type: ignore
    compute_ig_fidelity = None  # type: ignore
    compute_rice_fidelity = None  # type: ignore
    compute_random_fidelity = None  # type: ignore
    compute_statemask_fidelity = None  # type: ignore
    fidelity_from_domain = None  # type: ignore
    log_fidelity_table = None  # type: ignore
    rank_airs = None  # type: ignore
    rank_integrated_gradients = None  # type: ignore
    rank_rice = None  # type: ignore
    rank_random = None  # type: ignore
    rank_statemask = None  # type: ignore

try:
    from rice.evaluation.efficiency import (
        benchmark_rice_mask,
        benchmark_statemask_mask,
        compare_efficiency,
        efficiency_from_domain,
        log_efficiency_table,
    )
except Exception:  # pragma: no cover - optional dependency guard
    benchmark_rice_mask = None  # type: ignore
    benchmark_statemask_mask = None  # type: ignore
    compare_efficiency = None  # type: ignore
    efficiency_from_domain = None  # type: ignore
    log_efficiency_table = None  # type: ignore

try:
    from rice.evaluation.visualize import (
        plot_critical_state_distribution,
        plot_multiple_trajectory_heatmaps,
        plot_observation_importance_overlay,
        plot_trajectory_importance_heatmap,
        visualize_malware_case_study,
        visualize_metadrive_case_study,
        visualize_mujoco_critical_steps,
    )
except Exception:  # pragma: no cover - optional dependency guard
    plot_critical_state_distribution = None  # type: ignore
    plot_multiple_trajectory_heatmaps = None  # type: ignore
    plot_observation_importance_overlay = None  # type: ignore
    plot_trajectory_importance_heatmap = None  # type: ignore
    visualize_malware_case_study = None  # type: ignore
    visualize_metadrive_case_study = None  # type: ignore
    visualize_mujoco_critical_steps = None  # type: ignore

__all__ = [
    # evaluate_policy
    "evaluate_policy",
    "evaluate_policy_from_domain",
    "compare_policies",
    "log_evaluation",
    # fidelity
    "compare_fidelity",
    "compute_airs_fidelity",
    "compute_fidelity_score",
    "compute_ig_fidelity",
    "compute_rice_fidelity",
    "compute_random_fidelity",
    "compute_statemask_fidelity",
    "fidelity_from_domain",
    "log_fidelity_table",
    "rank_airs",
    "rank_integrated_gradients",
    "rank_rice",
    "rank_random",
    "rank_statemask",
    # efficiency
    "benchmark_rice_mask",
    "benchmark_statemask_mask",
    "compare_efficiency",
    "efficiency_from_domain",
    "log_efficiency_table",
    # visualize
    "plot_trajectory_importance_heatmap",
    "plot_multiple_trajectory_heatmaps",
    "plot_observation_importance_overlay",
    "plot_critical_state_distribution",
    "visualize_metadrive_case_study",
    "visualize_malware_case_study",
    "visualize_mujoco_critical_steps",
]
