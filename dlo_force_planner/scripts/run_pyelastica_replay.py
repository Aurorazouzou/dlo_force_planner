"""Run PyElastica replay in a separate process.

The main demo uses this helper so a slow or incompatible optional PyElastica
install cannot block the simple optimizer.
"""

from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dlo_force_planner.config import DemoConfig
from dlo_force_planner.simulator import PyElasticaReplaySimulator


def main() -> None:
    """Load replay inputs, run PyElastica, and save trajectory as ``.npy``."""

    if len(sys.argv) != 3:
        raise SystemExit(
            "Usage: python scripts/run_pyelastica_replay.py input.npz output.npy"
        )

    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    data = np.load(input_path)

    config = DemoConfig(output_dir=PROJECT_ROOT / "outputs")
    replay_simulator = PyElasticaReplaySimulator(config)
    trajectory = replay_simulator.rollout(
        data["force_locations"],
        data["force_sequence"],
    )
    np.save(output_path, trajectory)


if __name__ == "__main__":
    main()
