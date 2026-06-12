"""A tiny 2D DLO simulator.

This module deliberately avoids PyElastica. The DLO is represented as point
masses connected by springs, with an extra bending force that encourages each
interior node to stay near the midpoint of its neighbors.
"""

from dataclasses import dataclass

import numpy as np

from .config import DemoConfig
from .target_shapes import make_straight_line


@dataclass
class SimulationResult:
    """Container returned by ``SimpleDLOSimulator.rollout``."""

    positions: np.ndarray
    velocities: np.ndarray
    snapshots: np.ndarray


class SimpleDLOSimulator:
    """Simplified 2D flexible-line simulator for beginner-friendly demos."""

    def __init__(self, config: DemoConfig):
        """Create a simulator using values from ``DemoConfig``."""

        self.config = config
        self.initial_positions = make_straight_line(config.n_nodes, config.length)
        self.rest_length = config.length / (config.n_nodes - 1)

    def rollout(self, force_sequence: np.ndarray) -> SimulationResult:
        """Simulate the DLO under a sequence of external forces.

        Parameters
        ----------
        force_sequence:
            Array with shape ``(horizon, n_force_nodes, 2)``. For every planning
            step, each controlled node receives a 2D force ``[Fx, Fy]``.
        """

        positions = self.initial_positions.copy()
        velocities = np.zeros_like(positions)
        snapshots = [positions.copy()]

        for step_forces in force_sequence:
            for _ in range(self.config.substeps):
                forces = self._internal_forces(positions, velocities)
                self._add_external_forces(forces, step_forces)

                acceleration = forces / self.config.mass
                velocities += self.config.dt * acceleration
                positions += self.config.dt * velocities

                # Pin the two endpoints. This makes the target arc meaningful:
                # the optimizer bends the center upward while endpoints stay put.
                positions[0] = self.initial_positions[0]
                positions[-1] = self.initial_positions[-1]
                velocities[0] = 0.0
                velocities[-1] = 0.0

            snapshots.append(positions.copy())

        return SimulationResult(
            positions=positions,
            velocities=velocities,
            snapshots=np.asarray(snapshots),
        )

    def _internal_forces(self, positions: np.ndarray, velocities: np.ndarray) -> np.ndarray:
        """Compute spring, bending, and damping forces for all nodes."""

        forces = np.zeros_like(positions)
        forces += self._spring_forces(positions)
        forces += self._bending_forces(positions)
        forces += -self.config.damping * velocities
        return forces

    def _spring_forces(self, positions: np.ndarray) -> np.ndarray:
        """Pull neighboring nodes toward their original spacing."""

        forces = np.zeros_like(positions)
        for i in range(self.config.n_nodes - 1):
            edge = positions[i + 1] - positions[i]
            length = np.linalg.norm(edge)
            if length < 1e-12:
                continue
            direction = edge / length
            stretch = length - self.rest_length
            force = self.config.spring_stiffness * stretch * direction
            forces[i] += force
            forces[i + 1] -= force
        return forces

    def _bending_forces(self, positions: np.ndarray) -> np.ndarray:
        """Add a smoothness force based on the discrete second derivative."""

        forces = np.zeros_like(positions)
        for i in range(1, self.config.n_nodes - 1):
            midpoint_error = positions[i - 1] + positions[i + 1] - 2.0 * positions[i]
            forces[i] += self.config.bending_stiffness * midpoint_error
        return forces

    def _add_external_forces(self, forces: np.ndarray, step_forces: np.ndarray) -> None:
        """Add optimizer-selected forces to the configured control nodes."""

        for local_index, node_index in enumerate(self.config.force_nodes):
            forces[node_index] += step_forces[local_index]
