"""
Logger and checkpointing utilities for the FRE project.

Provides:
- Logger: Base logger for training metrics, checkpointing, and metric tracking.
- WandbLogger: Weights & Biases logger (extends Logger) for experiment tracking.
"""

import os
import json
import logging
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch


class Logger:
    """
    Base logger for training metrics and checkpointing.

    Tracks scalar metrics over time, maintains running averages,
    and supports saving/loading checkpoints.

    Attributes:
        log_dir: Directory for logs and checkpoints.
        metrics_history: Dict mapping metric names to lists of (step, value) tuples.
        running_metrics: Dict mapping metric names to deque for running averages.
        window_size: Size of running average window.
        step: Current global step counter.
        start_time: Time when logger was created.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        window_size: int = 100,
        verbose: bool = True,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the logger.

        Args:
            log_dir: Directory to save logs and checkpoints. If None, no file logging.
            window_size: Number of recent values to keep for running average.
            verbose: Whether to print metrics to console.
            log_level: Logging level for file/console output.
        """
        self.log_dir = log_dir
        self.window_size = window_size
        self.verbose = verbose

        # Create log directory if specified
        if self.log_dir is not None:
            os.makedirs(self.log_dir, exist_ok=True)

        # Set up Python logging
        self._logger = logging.getLogger(f"fre_logger_{id(self)}")
        self._logger.setLevel(log_level)
        self._logger.handlers.clear()

        # Console handler
        if verbose:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_formatter = logging.Formatter(
                "[%(asctime)s] %(message)s", datefmt="%H:%M:%S"
            )
            console_handler.setFormatter(console_formatter)
            self._logger.addHandler(console_handler)

        # File handler
        if self.log_dir is not None:
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, "training.log")
            )
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            self._logger.addHandler(file_handler)

        # Metric storage
        self.metrics_history: Dict[str, List[tuple]] = defaultdict(list)
        self.running_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        self.step = 0
        self.start_time = time.time()

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar metric.

        Args:
            name: Metric name.
            value: Scalar value.
            step: Step number. If None, uses internal step counter.
        """
        if step is None:
            step = self.step

        self.metrics_history[name].append((step, value))
        self.running_metrics[name].append(value)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dict mapping metric names to scalar values.
            step: Step number. If None, uses internal step counter.
        """
        for name, value in metrics.items():
            self.log_metric(name, value, step)

    def get_metric(self, name: str, running: bool = True) -> float:
        """
        Get the current value of a metric.

        Args:
            name: Metric name.
            running: If True, returns running average; else returns last value.

        Returns:
            Current metric value, or 0.0 if no data.
        """
        if running and len(self.running_metrics[name]) > 0:
            return np.mean(self.running_metrics[name])
        elif len(self.metrics_history[name]) > 0:
            return self.metrics_history[name][-1][1]
        return 0.0

    def get_all_metrics(self, running: bool = True) -> Dict[str, float]:
        """
        Get all current metric values.

        Args:
            running: If True, returns running averages.

        Returns:
            Dict mapping metric names to current values.
        """
        return {name: self.get_metric(name, running) for name in self.metrics_history}

    def log_info(self, message: str):
        """Log an informational message."""
        self._logger.info(message)

    def log_warning(self, message: str):
        """Log a warning message."""
        self._logger.warning(message)

    def log_error(self, message: str):
        """Log an error message."""
        self._logger.error(message)

    def increment_step(self):
        """Increment the internal step counter."""
        self.step += 1

    def set_step(self, step: int):
        """Set the internal step counter."""
        self.step = step

    def get_elapsed_time(self) -> float:
        """Get elapsed time since logger creation in seconds."""
        return time.time() - self.start_time

    def get_elapsed_time_str(self) -> str:
        """Get elapsed time as a human-readable string."""
        elapsed = self.get_elapsed_time()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        return f"{seconds}s"

    def print_summary(self, step: Optional[int] = None):
        """
        Print a summary of all current metrics.

        Args:
            step: Step number for the summary line.
        """
        if step is None:
            step = self.step

        metrics = self.get_all_metrics(running=True)
        metric_str = " | ".join(
            f"{name}: {value:.4f}" for name, value in sorted(metrics.items())
        )
        elapsed = self.get_elapsed_time_str()
        self._logger.info(f"Step {step} [{elapsed}] | {metric_str}")

    def save_metrics(self, filepath: Optional[str] = None):
        """
        Save metrics history to a JSON file.

        Args:
            filepath: Path to save. If None, saves to log_dir/metrics.json.
        """
        if filepath is None:
            if self.log_dir is None:
                self.log_warning("No log_dir specified; cannot save metrics.")
                return
            filepath = os.path.join(self.log_dir, "metrics.json")

        # Convert to serializable format
        serializable = {}
        for name, history in self.metrics_history.items():
            serializable[name] = [(int(s), float(v)) for s, v in history]

        with open(filepath, "w") as f:
            json.dump(serializable, f, indent=2)

    def load_metrics(self, filepath: str):
        """
        Load metrics history from a JSON file.

        Args:
            filepath: Path to the metrics JSON file.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        for name, history in data.items():
            for step, value in history:
                self.log_metric(name, value, step)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filename: str = "checkpoint.pt",
        is_best: bool = False,
    ):
        """
        Save a training checkpoint.

        Args:
            checkpoint: Dict containing model state_dicts, optimizer states, etc.
            filename: Base filename for the checkpoint.
            is_best: If True, also saves as 'best_checkpoint.pt'.
        """
        if self.log_dir is None:
            self.log_warning("No log_dir specified; cannot save checkpoint.")
            return

        filepath = os.path.join(self.log_dir, filename)
        torch.save(checkpoint, filepath)
        self._logger.info(f"Checkpoint saved to {filepath}")

        if is_best:
            best_path = os.path.join(self.log_dir, "best_checkpoint.pt")
            torch.save(checkpoint, best_path)
            self._logger.info(f"Best checkpoint saved to {best_path}")

    def load_checkpoint(self, filename: str = "checkpoint.pt") -> Optional[Dict[str, Any]]:
        """
        Load a training checkpoint.

        Args:
            filename: Checkpoint filename.

        Returns:
            Checkpoint dict, or None if not found.
        """
        if self.log_dir is None:
            self.log_warning("No log_dir specified; cannot load checkpoint.")
            return None

        filepath = os.path.join(self.log_dir, filename)
        if not os.path.exists(filepath):
            self.log_warning(f"Checkpoint not found: {filepath}")
            return None

        checkpoint = torch.load(filepath, map_location="cpu")
        self._logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint

    def close(self):
        """Close the logger and flush any pending writes."""
        for handler in self._logger.handlers:
            handler.close()
            self._logger.removeHandler(handler)


class WandbLogger(Logger):
    """
    Weights & Biases logger extending the base Logger.

    Automatically logs metrics to W&B when wandb is available.
    Falls back to base Logger behavior if wandb is not installed.

    Attributes:
        use_wandb: Whether W&B logging is active.
        project: W&B project name.
        config: Configuration dict logged to W&B.
    """

    def __init__(
        self,
        log_dir: Optional[str] = None,
        window_size: int = 100,
        verbose: bool = True,
        log_level: int = logging.INFO,
        use_wandb: bool = True,
        project: str = "fre",
        entity: Optional[str] = None,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        notes: Optional[str] = None,
        group: Optional[str] = None,
        job_type: Optional[str] = None,
        mode: str = "online",
    ):
        """
        Initialize the W&B logger.

        Args:
            log_dir: Directory for local logs and checkpoints.
            window_size: Running average window size.
            verbose: Whether to print to console.
            log_level: Logging level.
            use_wandb: Whether to attempt W&B initialization.
            project: W&B project name.
            entity: W&B entity (username or team).
            name: W&B run name.
            config: Configuration dict to log to W&B.
            tags: List of tags for the W&B run.
            notes: Notes for the W&B run.
            group: W&B group name.
            job_type: W&B job type.
            mode: W&B mode ('online', 'offline', 'disabled').
        """
        super().__init__(
            log_dir=log_dir,
            window_size=window_size,
            verbose=verbose,
            log_level=log_level,
        )

        self.use_wandb = False
        self.wandb_run = None

        if use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=project,
                    entity=entity,
                    name=name,
                    config=config or {},
                    tags=tags,
                    notes=notes,
                    group=group,
                    job_type=job_type,
                    dir=log_dir,
                    mode=mode,
                    reinit=True,
                )
                self.use_wandb = True
                self._logger.info(f"W&B initialized: {wandb.run.name}")
            except ImportError:
                self._logger.warning(
                    "wandb not installed; falling back to base Logger."
                )
            except Exception as e:
                self._logger.warning(f"Failed to initialize wandb: {e}")

    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """
        Log a scalar metric to both local storage and W&B.

        Args:
            name: Metric name.
            value: Scalar value.
            step: Step number.
        """
        super().log_metric(name, value, step)

        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.log({name: value}, step=step if step is not None else self.step)

    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log multiple metrics to both local storage and W&B.

        Args:
            metrics: Dict mapping metric names to values.
            step: Step number.
        """
        super().log_metrics(metrics, step)

        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.log(metrics, step=step if step is not None else self.step)

    def log_config(self, config: Dict[str, Any]):
        """
        Update the W&B config.

        Args:
            config: Configuration dict.
        """
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.config.update(config)

    def log_artifact(self, filepath: str, artifact_type: str = "model"):
        """
        Log a file as a W&B artifact.

        Args:
            filepath: Path to the file.
            artifact_type: Type of artifact.
        """
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            artifact = wandb.Artifact(
                name=os.path.basename(filepath),
                type=artifact_type,
            )
            artifact.add_file(filepath)
            wandb.log_artifact(artifact)

    def watch_model(self, model: torch.nn.Module, log_freq: int = 100):
        """
        Watch a PyTorch model's gradients and parameters in W&B.

        Args:
            model: PyTorch model to watch.
            log_freq: Frequency of logging gradients.
        """
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.watch(model, log="all", log_freq=log_freq)

    def finish(self):
        """Finish the W&B run and close the logger."""
        if self.use_wandb and self.wandb_run is not None:
            import wandb

            wandb.finish()
            self.use_wandb = False
            self.wandb_run = None

        self.close()


class MetricTracker:
    """
    Lightweight metric tracker for accumulating and averaging values.

    Useful for tracking metrics within a single training epoch or evaluation run.

    Example:
        tracker = MetricTracker()
        tracker.add("loss", 0.5)
        tracker.add("loss", 0.3)
        print(tracker.mean("loss"))  # 0.4
    """

    def __init__(self):
        self._values: Dict[str, List[float]] = defaultdict(list)

    def add(self, name: str, value: float):
        """Add a value for a metric."""
        self._values[name].append(value)

    def add_batch(self, metrics: Dict[str, float]):
        """Add multiple metric values at once."""
        for name, value in metrics.items():
            self.add(name, value)

    def mean(self, name: str) -> float:
        """Get the mean of a metric."""
        values = self._values.get(name, [])
        if len(values) == 0:
            return 0.0
        return np.mean(values)

    def std(self, name: str) -> float:
        """Get the standard deviation of a metric."""
        values = self._values.get(name, [])
        if len(values) == 0:
            return 0.0
        return np.std(values)

    def sum(self, name: str) -> float:
        """Get the sum of a metric."""
        return np.sum(self._values.get(name, []))

    def last(self, name: str) -> float:
        """Get the last value of a metric."""
        values = self._values.get(name, [])
        if len(values) == 0:
            return 0.0
        return values[-1]

    def all_means(self) -> Dict[str, float]:
        """Get means of all tracked metrics."""
        return {name: self.mean(name) for name in self._values}

    def all_stds(self) -> Dict[str, float]:
        """Get standard deviations of all tracked metrics."""
        return {name: self.std(name) for name in self._values}

    def reset(self):
        """Reset all tracked metrics."""
        self._values.clear()

    def __len__(self) -> int:
        """Number of unique metric names tracked."""
        return len(self._values)

    def __contains__(self, name: str) -> bool:
        return name in self._values