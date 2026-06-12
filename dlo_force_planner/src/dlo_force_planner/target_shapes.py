"""Helpers for creating initial and target DLO shapes."""

import numpy as np


def arc_length(points: np.ndarray) -> float:
    """Return the polyline length of a 2D node sequence.

    This is the length notion used by the discrete DLO: each neighboring pair of
    nodes forms one straight segment, and all segment lengths are summed.
    """

    segment_vectors = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    return float(np.sum(segment_lengths))


def make_straight_line(n_nodes: int, length: float) -> np.ndarray:
    """Return a horizontal DLO shape with ``n_nodes`` points.

    The returned array has shape ``(n_nodes, 2)``. Column 0 is x, column 1 is y.
    """

    x = np.linspace(0.0, length, n_nodes)
    y = np.zeros_like(x)
    return np.column_stack([x, y])


def make_upward_circular_arc(
    n_nodes: int,
    length: float,
    height: float,
) -> np.ndarray:
    """Return a circular arc whose endpoints match the initial straight line.

    ``height`` is the sagitta: the vertical distance from the endpoint line to
    the highest point of the arc. A larger height creates a stronger bend.
    """

    if height <= 0.0:
        raise ValueError("height must be positive for an upward arc")

    x = np.linspace(0.0, length, n_nodes)
    half_length = length / 2.0

    # Radius of a circular segment from chord length and sagitta.
    radius = (length * length) / (8.0 * height) + height / 2.0
    center_y = height - radius

    # Take the upper half of the circle so the middle node rises to ``height``.
    y = center_y + np.sqrt(np.maximum(radius * radius - (x - half_length) ** 2, 0.0))
    y -= y[0]
    return np.column_stack([x, y])


def generate_inextensible_arc_target(
    n_nodes: int,
    rod_length: float,
    arc_height: float,
    center_x: float = 0.5,
) -> np.ndarray:
    """Generate an upward circular arc with discrete length equal to rod length.

    The old target arc kept endpoints at ``x=0`` and ``x=1`` while lifting the
    middle upward, which made the target longer than an inextensible rod. This
    function instead solves for a smaller chord length when ``arc_height > 0``
    so the sum of distances between neighboring target nodes equals
    ``rod_length``.
    """

    if n_nodes < 2:
        raise ValueError("n_nodes must be at least 2")
    if rod_length <= 0.0:
        raise ValueError("rod_length must be positive")
    if arc_height < 0.0:
        raise ValueError("arc_height must be non-negative")
    if arc_height == 0.0:
        x = np.linspace(center_x - rod_length / 2.0, center_x + rod_length / 2.0, n_nodes)
        y = np.zeros_like(x)
        return np.column_stack([x, y])

    max_height = rod_length / np.pi
    if arc_height >= max_height:
        raise ValueError(
            "arc_height is too large for the requested inextensible arc. "
            f"Use arc_height < {max_height:.6f} for rod_length={rod_length}."
        )

    # Let phi be half of the circular arc angle. With radius R:
    #   arc height = R * (1 - cos(phi))
    # For the discrete DLO we match the polyline length, not the continuous
    # circle length:
    #   rod_length = (n_nodes - 1) * 2 * R * sin(phi / (n_nodes - 1))
    n_segments = n_nodes - 1

    def discrete_length_for_phi(phi: float) -> float:
        radius = arc_height / (1.0 - np.cos(phi))
        return n_segments * 2.0 * radius * np.sin(phi / n_segments)

    low = 1e-8
    high = np.pi - 1e-8
    for _ in range(80):
        mid = 0.5 * (low + high)
        if discrete_length_for_phi(mid) > rod_length:
            low = mid
        else:
            high = mid

    phi = 0.5 * (low + high)
    radius = arc_height / (1.0 - np.cos(phi))
    angles = np.linspace(-phi, phi, n_nodes)

    x = center_x + radius * np.sin(angles)
    y = radius * np.cos(angles) - radius * np.cos(phi)
    return np.column_stack([x, y])
