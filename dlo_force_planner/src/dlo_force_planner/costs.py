"""Cost terms used by the genetic algorithm."""

import numpy as np

from .config import DemoConfig
from .target_shapes import arc_length


def discrete_curvature(shape: np.ndarray) -> np.ndarray:
    """Approximate curvature with the norm of the second finite difference."""

    second_difference = shape[:-2] - 2.0 * shape[1:-1] + shape[2:]
    return np.linalg.norm(second_difference, axis=1)


def segment_length_change(shape: np.ndarray, config: DemoConfig) -> np.ndarray:
    """Return absolute segment-length error relative to the initial spacing."""

    segment_lengths = np.linalg.norm(np.diff(shape, axis=0), axis=1)
    rest_length = config.rod_length / (config.n_nodes - 1)
    return np.abs(segment_lengths - rest_length)


def shape_error(final_shape: np.ndarray, target_shape: np.ndarray) -> float:
    """Mean squared distance between final and target node positions."""

    distances = final_shape - target_shape
    return float(np.mean(np.sum(distances * distances, axis=1)))


def aligned_shape_error(final_shape: np.ndarray, target_shape: np.ndarray) -> float:
    """Shape error after removing translation by centroid alignment."""

    final_centered = final_shape - np.mean(final_shape, axis=0)
    target_centered = target_shape - np.mean(target_shape, axis=0)
    return shape_error(final_centered, target_centered)


def configured_shape_error(
    final_shape: np.ndarray,
    target_shape: np.ndarray,
    config: DemoConfig,
) -> float:
    """Use absolute error for fixed endpoints and aligned error for free DLO."""

    if config.fix_endpoints:
        return shape_error(final_shape, target_shape)
    return aligned_shape_error(final_shape, target_shape)


def trajectory_shape_error(
    snapshots: np.ndarray,
    initial_shape: np.ndarray,
    target_shape: np.ndarray,
) -> float:
    """Penalize deviation from a linear interpolation in configuration space."""

    horizon = snapshots.shape[0] - 1
    if horizon <= 0:
        return 0.0

    errors = []
    for step, snapshot in enumerate(snapshots):
        alpha = step / horizon
        intermediate_target = (1.0 - alpha) * initial_shape + alpha * target_shape
        errors.append(shape_error(snapshot, intermediate_target))
    return float(np.mean(errors))


def path_smoothness_penalty(snapshots: np.ndarray) -> float:
    """Penalize sudden jumps between neighboring DLO configurations."""

    if snapshots.shape[0] <= 1:
        return 0.0
    delta = np.diff(snapshots, axis=0)
    return float(np.mean(np.sum(delta * delta, axis=2)))


def monotonic_progress_penalty(snapshots: np.ndarray, target_shape: np.ndarray) -> float:
    """Penalize rollout steps that move farther away from the target shape."""

    errors = np.asarray([aligned_shape_error(snapshot, target_shape) for snapshot in snapshots])
    increases = np.maximum(0.0, np.diff(errors))
    return float(np.mean(increases * increases)) if len(increases) else 0.0


def force_magnitude_penalty(force_sequence: np.ndarray) -> float:
    """Mean squared force magnitude across all steps and control nodes."""

    return float(np.mean(np.sum(force_sequence * force_sequence, axis=2)))


def force_smoothness_penalty(force_sequence: np.ndarray) -> float:
    """Penalize sudden force changes between neighboring time steps."""

    if force_sequence.shape[0] <= 1:
        return 0.0
    force_delta = np.diff(force_sequence, axis=0)
    return float(np.mean(np.sum(force_delta * force_delta, axis=2)))


def net_force_penalty(force_sequence: np.ndarray) -> float:
    """Penalize nonzero resultant force at each planning step."""

    net_force = np.sum(force_sequence, axis=1)
    return float(np.mean(np.sum(net_force * net_force, axis=1)))


def force_torque_penalty(
    force_locations: np.ndarray,
    force_sequence: np.ndarray,
    initial_shape: np.ndarray,
    config: DemoConfig,
) -> float:
    """Penalize net out-of-plane torque from the applied planar forces."""

    force_points = force_application_points(force_locations, initial_shape, config)
    center = np.mean(initial_shape, axis=0)
    torques = []
    for step in range(force_sequence.shape[0]):
        moment_arm = force_points - center
        force = force_sequence[step]
        torque_z = moment_arm[:, 0] * force[:, 1] - moment_arm[:, 1] * force[:, 0]
        torques.append(float(np.sum(torque_z)))
    torques = np.asarray(torques)
    return float(np.mean(torques * torques))


def force_application_points(
    force_locations: np.ndarray,
    reference_shape: np.ndarray,
    config: DemoConfig,
) -> np.ndarray:
    """Map continuous force locations to interpolated points on a reference DLO."""

    locations = np.asarray(force_locations, dtype=float)
    if locations.ndim == 2:
        locations = locations[0]

    points = []
    for location in locations:
        left = int(np.floor(location))
        left = int(np.clip(left, 0, config.n_nodes - 1))
        right = min(left + 1, config.n_nodes - 1)
        alpha = float(location - left)
        point = (1.0 - alpha) * reference_shape[left] + alpha * reference_shape[right]
        points.append(point)
    return np.asarray(points)


def curvature_penalty(final_shape: np.ndarray) -> float:
    """Penalize very sharp bends in the final DLO shape."""

    curvature = discrete_curvature(final_shape)
    return float(np.mean(curvature * curvature))


def length_change_penalty(final_shape: np.ndarray, config: DemoConfig) -> float:
    """Penalize stretching or compression of neighboring node distances."""

    length_error = segment_length_change(final_shape, config)
    return float(np.mean(length_error * length_error))


def total_length_error_penalty(final_shape: np.ndarray, config: DemoConfig) -> float:
    """Strongly penalize total rod length changes."""

    final_length = arc_length(final_shape)
    return float((final_length - config.rod_length) ** 2)


def separation_penalty(force_locations: np.ndarray, config: DemoConfig) -> float:
    """Penalize force locations that are closer than the configured spacing.

    The optimizer is allowed to choose continuous locations, but this term
    discourages all three forces from collapsing onto the same small segment.
    """

    penalty = 0.0
    for i in range(len(force_locations)):
        for j in range(i + 1, len(force_locations)):
            distance = abs(float(force_locations[i] - force_locations[j]))
            overlap = max(0.0, config.min_force_separation - distance)
            penalty += overlap * overlap
    return float(penalty)


def total_cost(
    final_shape: np.ndarray,
    snapshots: np.ndarray,
    initial_shape: np.ndarray,
    target_shape: np.ndarray,
    force_locations: np.ndarray,
    force_sequence: np.ndarray,
    config: DemoConfig,
) -> tuple[float, dict[str, float]]:
    """Return weighted total cost plus a dictionary of readable components."""

    components = {
        "shape_error": configured_shape_error(final_shape, target_shape, config),
        "aligned_shape_error": aligned_shape_error(final_shape, target_shape),
        "absolute_shape_error": shape_error(final_shape, target_shape),
        "force_magnitude": force_magnitude_penalty(force_sequence),
        "force_smoothness": force_smoothness_penalty(force_sequence),
        "net_force_penalty": net_force_penalty(force_sequence),
        "net_torque_penalty": force_torque_penalty(
            force_locations,
            force_sequence,
            initial_shape,
            config,
        ),
        "curvature": curvature_penalty(final_shape),
        "length_change": length_change_penalty(final_shape, config),
        "length_error": total_length_error_penalty(final_shape, config),
        "separation": separation_penalty(force_locations, config),
        "trajectory_shape_error": trajectory_shape_error(
            snapshots,
            initial_shape,
            target_shape,
        ),
        "path_smoothness": path_smoothness_penalty(snapshots),
        "monotonic_progress_penalty": monotonic_progress_penalty(
            snapshots,
            target_shape,
        ),
    }

    length_error = segment_length_change(final_shape, config)
    curvature = discrete_curvature(final_shape)
    force_magnitude = np.linalg.norm(force_sequence, axis=2)
    initial_length = config.rod_length
    target_length = arc_length(target_shape)
    final_length = arc_length(final_shape)
    components["initial_length"] = float(initial_length)
    components["target_length"] = float(target_length)
    components["final_length"] = float(final_length)
    components["target_length_error"] = float(target_length - initial_length)
    components["final_length_error"] = float(final_length - initial_length)
    components["max_segment_length_error"] = float(np.max(length_error))
    components["mean_segment_length_error"] = float(np.mean(length_error))
    components["max_length_change"] = float(np.max(length_error))
    components["mean_length_change"] = float(np.mean(length_error))
    components["max_curvature"] = float(np.max(curvature))
    components["mean_curvature"] = float(np.mean(curvature))
    components["force_max_used"] = float(np.max(force_magnitude))
    if force_locations.ndim == 2:
        components["location_smoothness"] = float(np.mean(np.diff(force_locations, axis=0) ** 2))

    weighted_total = (
        config.shape_weight * components["shape_error"]
        + config.force_weight * components["force_magnitude"]
        + config.smooth_weight * components["force_smoothness"]
        + config.net_force_weight * components["net_force_penalty"]
        + config.net_torque_weight * components["net_torque_penalty"]
        + config.curvature_weight * components["curvature"]
        + config.length_weight * components["length_error"]
        + config.length_weight * components["length_change"]
        + config.separation_weight * components["separation"]
        + config.trajectory_weight * components["trajectory_shape_error"]
        + config.path_smoothness_weight * components["path_smoothness"]
        + config.progress_weight * components["monotonic_progress_penalty"]
    )
    components["total_cost"] = float(weighted_total)
    return float(weighted_total), components
