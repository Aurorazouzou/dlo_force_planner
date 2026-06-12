"""Plotting and animation utilities for the DLO force-planning demo."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .config import DemoConfig


def _set_equal_dlo_axes(ax, config: DemoConfig) -> None:
    """Apply consistent axis limits so all plots are easy to compare."""

    margin = 0.12
    ax.set_xlim(-margin, config.length + margin)
    ax.set_ylim(-0.18, max(0.45, config.target_height + 0.15))
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


def plot_force_history(
    force_sequence: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save x/y force values over the planning horizon."""

    steps = np.arange(force_sequence.shape[0])
    fig, axes = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    for local_index, node_index in enumerate(config.force_nodes):
        axes[0].plot(steps, force_sequence[:, local_index, 0], "o-", label=f"node {node_index}")
        axes[1].plot(steps, force_sequence[:, local_index, 1], "o-", label=f"node {node_index}")

    axes[0].set_ylabel("Fx")
    axes[1].set_ylabel("Fy")
    axes[1].set_xlabel("planning step")
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("Optimized force history")
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
