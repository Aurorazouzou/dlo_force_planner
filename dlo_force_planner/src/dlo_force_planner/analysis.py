"""CSV output helpers for the demo."""

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DemoConfig


def save_force_sequence_csv(
    force_sequence: np.ndarray,
    output_path: Path,
) -> None:
    """Save the optimized force sequence in a long, beginner-readable table."""

    rows = []
    for step in range(force_sequence.shape[0]):
        for force_index in range(force_sequence.shape[1]):
            fx, fy = force_sequence[step, force_index]
            rows.append(
                {
                    "step": step,
                    "force_id": force_index + 1,
                    "Fx": float(fx),
                    "Fy": float(fy),
                }
            )

    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_force_locations_csv(force_locations: np.ndarray, output_path: Path) -> None:
    """Save optimized continuous force locations to a small CSV table."""

    rows = [
        {"force_id": index + 1, "location": float(location)}
        for index, location in enumerate(force_locations)
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_metrics_csv(metrics: dict[str, float], output_path: Path) -> None:
    """Save scalar cost components to a two-column CSV file."""

    rows = [{"metric": key, "value": float(value)} for key, value in metrics.items()]
    pd.DataFrame(rows).to_csv(output_path, index=False)
