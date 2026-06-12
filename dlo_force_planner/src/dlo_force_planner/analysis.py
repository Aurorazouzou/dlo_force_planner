"""CSV output helpers for the demo."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DemoConfig


def save_force_sequence_csv(
    force_sequence: np.ndarray,
    config: DemoConfig,
    output_path: Path,
) -> None:
    """Save the optimized force sequence in a long, beginner-readable table."""

    rows = []
    for step in range(force_sequence.shape[0]):
        for local_index, node_index in enumerate(config.force_nodes):
            fx, fy = force_sequence[step, local_index]
            rows.append(
                {
                    "step": step,
                    "node": node_index,
                    "Fx": float(fx),
                    "Fy": float(fy),
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_metrics_csv(metrics: dict[str, float], output_path: Path) -> None:
    """Save scalar cost components to a two-column CSV file."""

    rows = [{"metric": key, "value": float(value)} for key, value in metrics.items()]
    pd.DataFrame(rows).to_csv(output_path, index=False)
