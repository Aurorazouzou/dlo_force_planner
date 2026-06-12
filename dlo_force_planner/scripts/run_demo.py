"""Run the minimal DLO force-planning demo.

Execute from the project root:

    python scripts/run_demo.py
"""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dlo_force_planner.analysis import save_force_sequence_csv, save_metrics_csv
from dlo_force_planner.config import DemoConfig
from dlo_force_planner.planner import optimize_force_sequence
from dlo_force_planner.simulator import SimpleDLOSimulator
from dlo_force_planner.target_shapes import make_upward_circular_arc
from dlo_force_planner.visualization import (
    plot_convergence,
    plot_force_history,
    plot_shape_result,
    plot_trajectory_snapshots,
    save_motion_gif,
)


def make_output_dirs(config: DemoConfig) -> None:
    """Create all folders required by the requested output files."""

    config.figure_dir.mkdir(parents=True, exist_ok=True)
    config.animation_dir.mkdir(parents=True, exist_ok=True)
    config.data_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Build the problem, run GA optimization, and export all results."""

    config = DemoConfig(output_dir=PROJECT_ROOT / "outputs")
    make_output_dirs(config)

    simulator = SimpleDLOSimulator(config)
    target_shape = make_upward_circular_arc(
        n_nodes=config.n_nodes,
        length=config.length,
        height=config.target_height,
    )

    print("Running GA optimization...")
    print(f"  horizon: {config.horizon}")
    print(f"  force nodes: {config.force_nodes}")
    print(f"  generations: {config.n_generations}")

    result = optimize_force_sequence(simulator, target_shape, config)

    plot_shape_result(
        simulator.initial_positions,
        target_shape,
        result.final_shape,
        config,
        config.figure_dir / "shape_result.png",
    )
    plot_trajectory_snapshots(
        result.snapshots,
        target_shape,
        config,
        config.figure_dir / "trajectory_snapshots.png",
    )
    plot_force_history(
        result.best_force_sequence,
        config,
        config.figure_dir / "force_history.png",
    )
    plot_convergence(
        result.convergence,
        config.figure_dir / "convergence.png",
    )
    save_motion_gif(
        result.snapshots,
        target_shape,
        config,
        config.animation_dir / "dlo_motion.gif",
    )

    save_force_sequence_csv(
        result.best_force_sequence,
        config,
        config.data_dir / "best_force_sequence.csv",
    )
    save_metrics_csv(
        result.cost_components,
        config.data_dir / "metrics.csv",
    )

    print("Done. Generated files:")
    print(f"  {config.figure_dir / 'shape_result.png'}")
    print(f"  {config.figure_dir / 'trajectory_snapshots.png'}")
    print(f"  {config.figure_dir / 'force_history.png'}")
    print(f"  {config.figure_dir / 'convergence.png'}")
    print(f"  {config.animation_dir / 'dlo_motion.gif'}")
    print(f"  {config.data_dir / 'best_force_sequence.csv'}")
    print(f"  {config.data_dir / 'metrics.csv'}")
    print(f"Best cost: {result.best_cost:.6f}")


if __name__ == "__main__":
    main()
