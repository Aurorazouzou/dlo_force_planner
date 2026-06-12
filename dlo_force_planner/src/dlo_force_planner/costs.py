"""Cost terms used by the genetic algorithm."""

import numpy as np

from .config import DemoConfig


def discrete_curvature(shape: np.ndarray) -> np.ndarray:
    """Approximate curvature with the norm of the second finite difference."""

    second_difference = shape[:-2] - 2.0 * shape[1:-1] + shape[2:]
    return np.linalg.norm(second_difference, axis=1)


def shape_error(final_shape: np.ndarray, target_shape: np.ndarray) -> float:
    """Mean squared distance between final and target node positions."""

    distances = final_shape - target_shape
    return float(np.mean(np.sum(distances * distances, axis=1)))


def force_magnitude_penalty(force_sequence: np.ndarray) -> float:
    """Mean squared force magnitude across all steps and control nodes."""

    return float(np.mean(np.sum(force_sequence * force_sequence, axis=2)))


def force_smoothness_penalty(force_sequence: np.ndarray) -> float:
    """Penalize sudden force changes between neighboring time steps."""

    if force_sequence.shape[0] <= 1:
        return 0.0
    force_delta = np.diff(force_sequence, axis=0)
    return float(np.mean(np.sum(force_delta * force_delta, axis=2)))


def curvature_penalty(final_shape: np.ndarray) -> float:
    """Penalize very sharp bends in the final DLO shape."""

    curvature = discrete_curvature(final_shape)
    return float(np.mean(curvature * curvature))


def total_cost(
    final_shape: np.ndarray,
    target_shape: np.ndarray,
    force_sequence: np.ndarray,
    config: DemoConfig,
) -> tuple[float, dict[str, float]]:
    """Return weighted total cost plus a dictionary of readable components."""

    components = {
        "shape_error": shape_error(final_shape, target_shape),
        "force_magnitude": force_magnitude_penalty(force_sequence),
        "force_smoothness": force_smoothness_penalty(force_sequence),
        "curvature": curvature_penalty(final_shape),
    }

    weighted_total = (
        config.shape_weight * components["shape_error"]
        + config.force_weight * components["force_magnitude"]
        + config.smooth_weight * components["force_smoothness"]
        + config.curvature_weight * components["curvature"]
    )
    components["total_cost"] = float(weighted_total)
    return float(weighted_total), components
