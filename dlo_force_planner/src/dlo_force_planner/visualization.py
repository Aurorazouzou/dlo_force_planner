"""Plotting and animation utilities for the DLO force-planning demo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .config import DemoConfig


def _set_equal_dlo_axes(ax, config: DemoConfig) -> None:
    """Apply consistent axis limits so all plots are easy to compare."""

    margin = 0.15 * config.rod_length
    y_margin = 0.20 * config.rod_length
    ax.set_xlim(-margin, config.length + margin)
    ax.set_ylim(-y_margin, config.target_height + y_margin)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x")
    ax.set_ylabel("y")


def plot_shape_result(
    initial_shape: np.ndarray,
    target_shape: np.ndarray,
    final_shape: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save a figure comparing initial, target, and optimized final shapes."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(initial_shape[:, 0], initial_shape[:, 1], "o--", label="initial")
    ax.plot(target_shape[:, 0], target_shape[:, 1], "k-", linewidth=2.0, label="target")
    ax.plot(final_shape[:, 0], final_shape[:, 1], "ro-", label="optimized")
    _set_equal_dlo_axes(ax, config)
    ax.set_title("DLO final shape")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trajectory_snapshots(
    snapshots: np.ndarray,
    target_shape: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save several DLO shapes from the rollout as a single trajectory plot."""

    fig, ax = plt.subplots(figsize=(7, 4))
    snapshot_ids = np.linspace(0, len(snapshots) - 1, 6, dtype=int)
    colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(snapshot_ids)))

    for color, snapshot_id in zip(colors, snapshot_ids):
        shape = snapshots[snapshot_id]
        ax.plot(shape[:, 0], shape[:, 1], "o-", color=color, label=f"step {snapshot_id}")

    ax.plot(target_shape[:, 0], target_shape[:, 1], "k--", linewidth=2.0, label="target")
    _set_equal_dlo_axes(ax, config)
    ax.set_title("Trajectory snapshots")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_trajectory_error(
    snapshots: np.ndarray,
    target_shape: np.ndarray,
    output_path: Path,
) -> None:
    """Save per-step shape error to the final target shape."""

    errors = []
    for snapshot in snapshots:
        delta = snapshot - target_shape
        errors.append(float(np.mean(np.sum(delta * delta, axis=1))))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(np.arange(len(errors)), errors, "o-", color="tab:red")
    ax.set_xlabel("planning step")
    ax.set_ylabel("shape error to target")
    ax.set_title("Trajectory error to target")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_force_history(
    force_sequence: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save x/y force values over the planning horizon."""

    steps = np.arange(force_sequence.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for force_index in range(force_sequence.shape[1]):
        label = f"force {force_index + 1}"
        axes[0].plot(steps, force_sequence[:, force_index, 0], "o-", label=label)
        axes[1].plot(steps, force_sequence[:, force_index, 1], "o-", label=label)

    axes[0].set_ylabel("Fx")
    axes[1].set_ylabel("Fy")
    axes[1].set_xlabel("planning step")
    for ax in axes:
        ax.axhline(config.force_max, color="0.25", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axhline(-config.force_max, color="0.25", linestyle="--", linewidth=1.0, alpha=0.6)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Optimized force history")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_length_error(
    final_shape: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save per-segment length error for the final DLO shape."""

    segment_lengths = np.linalg.norm(np.diff(final_shape, axis=0), axis=1)
    rest_length = config.rod_length / (config.n_nodes - 1)
    length_error = segment_lengths - rest_length
    segment_ids = np.arange(len(length_error))

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(segment_ids, length_error, color="tab:blue", alpha=0.8)
    ax.axhline(0.0, color="0.2", linewidth=1.0)
    ax.set_xlabel("segment index")
    ax.set_ylabel("length change")
    ax.set_title("Segment length error")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_force_locations(
    initial_shape: np.ndarray,
    force_locations: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save a plot showing the optimized continuous force locations."""

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(initial_shape[:, 0], initial_shape[:, 1], "o-", color="0.35", label="initial DLO")

    for force_index, location in enumerate(force_locations):
        left = int(np.floor(location))
        left = int(np.clip(left, 0, config.n_nodes - 1))
        right = min(left + 1, config.n_nodes - 1)
        alpha = float(location - left)
        point = (1.0 - alpha) * initial_shape[left] + alpha * initial_shape[right]
        ax.scatter(point[0], point[1], s=90, label=f"force {force_index + 1}")
        ax.annotate(
            f"{location:.2f}",
            xy=(point[0], point[1]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
        )

    _set_equal_dlo_axes(ax, config)
    ax.set_title("Optimized force locations")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_convergence(convergence: np.ndarray, output_path: Path) -> None:
    """Save GA best-cost history."""

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(convergence) + 1), convergence, "b-")
    ax.set_xlabel("generation")
    ax.set_ylabel("best cost")
    ax.set_title("GA convergence")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_motion_gif(
    snapshots: np.ndarray,
    target_shape: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save a GIF animation of the optimized rollout."""

    fig, ax = plt.subplots(figsize=(7, 4))
    _set_equal_dlo_axes(ax, config)
    ax.set_title("DLO motion")
    ax.plot(target_shape[:, 0], target_shape[:, 1], "k--", linewidth=2.0, label="target")
    (line,) = ax.plot([], [], "ro-", label="DLO")
    ax.legend()

    def update(frame_index):
        shape = snapshots[frame_index]
        line.set_data(shape[:, 0], shape[:, 1])
        return (line,)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(snapshots),
        interval=180,
        blit=True,
    )
    animation.save(output_path, writer=PillowWriter(fps=6))
    plt.close(fig)
