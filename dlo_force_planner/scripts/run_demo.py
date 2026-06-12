"""Run the minimal DLO force-planning demo.

Execute from the project root:

    python scripts/run_demo.py
"""

from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dlo_force_planner.analysis import (
    save_force_locations_csv,
    save_force_sequence_csv,
    save_metrics_csv,
)
from dlo_force_planner.config import DemoConfig
from dlo_force_planner.costs import total_cost
from dlo_force_planner.planner import optimize_force_sequence
from dlo_force_planner.simulator import SimpleDLOSimulator
from dlo_force_planner.target_shapes import arc_length, generate_inextensible_arc_target
from dlo_force_planner.visualization import (
    plot_convergence,
    plot_force_history,
    plot_force_locations,
    plot_length_error,
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

    backend = "simple"
    replay_backend = "pyelastica"

    config = DemoConfig(output_dir=PROJECT_ROOT / "outputs")
    make_output_dirs(config)

    if backend != "simple":
        raise ValueError("This demo currently optimizes only with backend='simple'.")

    simulator = SimpleDLOSimulator(config)
    target_shape = generate_inextensible_arc_target(
        n_nodes=config.n_nodes,
        rod_length=config.rod_length,
        arc_height=config.target_height,
        center_x=config.length / 2.0,
    )
    initial_length = arc_length(simulator.initial_positions)
    target_length = arc_length(target_shape)
    if abs(target_length - initial_length) > 1e-3:
        print(
            "Warning: target length does not match initial length. "
            f"initial_length={initial_length:.6f}, target_length={target_length:.6f}"
        )

    print("Running GA optimization...")
    print(f"  backend: {backend}")
    print(f"  replay_backend: {replay_backend}")
    print(f"  horizon: {config.horizon}")
    print(f"  n_forces: {config.n_forces}")
    print(f"  generations: {config.n_generations}")

    result = optimize_force_sequence(simulator, target_shape, config)
    final_length_error = abs(result.cost_components["final_length_error"])
    if final_length_error > 1e-3:
        print(
            "Warning: final DLO length differs from rod_length by more than 1e-3. "
            f"final_length_error={final_length_error:.6f}"
        )

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
    plot_force_locations(
        simulator.initial_positions,
        result.best_force_locations,
        config,
        config.figure_dir / "force_locations.png",
    )
    plot_convergence(
        result.convergence,
        config.figure_dir / "convergence.png",
    )
    plot_length_error(
        result.final_shape,
        config,
        config.figure_dir / "length_error.png",
    )
    save_motion_gif(
        result.snapshots,
        target_shape,
        config,
        config.animation_dir / "dlo_motion.gif",
    )

    save_force_sequence_csv(
        result.best_force_sequence,
        config.data_dir / "best_force_sequence.csv",
    )
    save_force_locations_csv(
        result.best_force_locations,
        config.data_dir / "optimized_force_locations.csv",
    )
    save_metrics_csv(
        result.cost_components,
        config.data_dir / "metrics.csv",
    )

    replay_success = False
    if replay_backend == "pyelastica":
        print("Running PyElastica replay...")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                replay_input_path = Path(temp_dir) / "pyelastica_replay_input.npz"
                replay_output_path = Path(temp_dir) / "pyelastica_replay_trajectory.npy"
                np.savez(
                    replay_input_path,
                    force_locations=result.best_force_locations,
                    force_sequence=result.best_force_sequence,
                )

                replay_process = subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "run_pyelastica_replay.py"),
                        str(replay_input_path),
                        str(replay_output_path),
                    ],
                    cwd=PROJECT_ROOT,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if replay_process.returncode == 0:
                    replay_snapshots = np.load(replay_output_path)

            if replay_process.returncode != 0:
                print("Skipping PyElastica replay because the optional backend failed.")
                if replay_process.stdout.strip():
                    print(replay_process.stdout.strip())
                if replay_process.stderr.strip():
                    print(replay_process.stderr.strip())
                raise RuntimeError("PyElastica replay subprocess failed")

            _, replay_metrics = total_cost(
                replay_snapshots[-1],
                target_shape,
                result.best_force_locations,
                result.best_force_sequence,
                config,
            )
            for index, location in enumerate(result.best_force_locations, start=1):
                replay_metrics[f"force_location_{index}"] = float(location)

            plot_shape_result(
                simulator.initial_positions,
                target_shape,
                replay_snapshots[-1],
                config,
                config.figure_dir / "pyelastica_replay_shape.png",
            )
            plot_trajectory_snapshots(
                replay_snapshots,
                target_shape,
                config,
                config.figure_dir / "pyelastica_replay_snapshots.png",
            )
            save_metrics_csv(
                replay_metrics,
                config.data_dir / "pyelastica_replay_metrics.csv",
            )
            replay_success = True
        except subprocess.TimeoutExpired:
            print(
                "Skipping PyElastica replay: importing or running PyElastica "
                "took longer than 15 seconds. Install/check with: pip install pyelastica"
            )
        except Exception as exc:
            if str(exc) != "PyElastica replay subprocess failed":
                print("Skipping PyElastica replay because the optional backend failed.")
                print(f"  Error: {exc}")

    print("Done. Generated files:")
    print(f"  {config.figure_dir / 'shape_result.png'}")
    print(f"  {config.figure_dir / 'trajectory_snapshots.png'}")
    print(f"  {config.figure_dir / 'force_history.png'}")
    print(f"  {config.figure_dir / 'force_locations.png'}")
    print(f"  {config.figure_dir / 'convergence.png'}")
    print(f"  {config.figure_dir / 'length_error.png'}")
    print(f"  {config.animation_dir / 'dlo_motion.gif'}")
    print(f"  {config.data_dir / 'best_force_sequence.csv'}")
    print(f"  {config.data_dir / 'optimized_force_locations.csv'}")
    print(f"  {config.data_dir / 'metrics.csv'}")
    if replay_success:
        print(f"  {config.figure_dir / 'pyelastica_replay_shape.png'}")
        print(f"  {config.figure_dir / 'pyelastica_replay_snapshots.png'}")
        print(f"  {config.data_dir / 'pyelastica_replay_metrics.csv'}")
    print(f"Best cost: {result.best_cost:.6f}")


if __name__ == "__main__":
    main()
