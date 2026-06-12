"""Helpers for creating initial and target DLO shapes."""

import numpy as np


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
