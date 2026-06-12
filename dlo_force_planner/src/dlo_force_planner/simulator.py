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
        self.rest_segment_length = config.rod_length / (config.n_nodes - 1)
        self.rest_length = self.rest_segment_length

    def rollout(
        self,
        force_locations: np.ndarray,
        force_sequence: np.ndarray,
    ) -> SimulationResult:
        """Simulate the DLO under a sequence of external forces.

        Parameters
        ----------
        force_locations:
            Continuous force locations with shape ``(n_forces,)``. A value such
            as 8.25 means 75 percent of the force goes to node 8 and 25 percent
            goes to node 9. Locations may also be exactly on either endpoint.
        force_sequence:
            Array with shape ``(horizon, n_forces, 2)``. For every planning
            step, each external force has a 2D value ``[Fx, Fy]``.
        """

        positions = self.initial_positions.copy()
        velocities = np.zeros_like(positions)
        snapshots = [positions.copy()]
        force_locations = self._clip_force_locations(force_locations)
        if np.max(np.linalg.norm(force_sequence, axis=2)) < 1e-8:
            snapshots.extend([positions.copy() for _ in range(force_sequence.shape[0])])
            return SimulationResult(
                positions=positions,
                velocities=velocities,
                snapshots=np.asarray(snapshots),
            )

        for step_forces in force_sequence:
            for _ in range(self.config.substeps):
                previous_positions = positions.copy()
                forces = self._internal_forces(positions, velocities)
                self._add_external_forces(forces, force_locations, step_forces)

                acceleration = forces / self.config.mass
                velocities += self.config.dt * acceleration
                velocities *= self.config.damping
                positions += self.config.dt * velocities
                positions = self._limit_node_displacement(previous_positions, positions)
                positions = self._smooth_bends(positions)
                if self.config.inextensible:
                    positions = self._project_inextensible_lengths(positions)
                else:
                    positions = self._enforce_segment_lengths(positions)
                velocities = (positions - previous_positions) / self.config.dt

                if self.config.fix_endpoints:
                    # Some experiments need fixed endpoints. The default demo
                    # leaves them free so endpoint forces can create motion.
                    positions[0] = self.initial_positions[0]
                    positions[-1] = self.initial_positions[-1]
                    velocities[0] = 0.0
                    velocities[-1] = 0.0

            if self.config.quasi_static:
                velocities *= 0.0
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
            force = self.config.stretch_stiffness * stretch * direction
            forces[i] += force
            forces[i + 1] -= force
        return forces

    def _bending_forces(self, positions: np.ndarray) -> np.ndarray:
        """Add a smoothness force based on the discrete second derivative."""

        forces = np.zeros_like(positions)
        for i in range(1, self.config.n_nodes - 1):
            midpoint_error = positions[i - 1] + positions[i + 1] - 2.0 * positions[i]
            forces[i] += self.config.bend_stiffness * midpoint_error
        return forces

    def _limit_node_displacement(
        self,
        previous_positions: np.ndarray,
        proposed_positions: np.ndarray,
    ) -> np.ndarray:
        """Limit how far any node can move during one small integration step."""

        displacement = proposed_positions - previous_positions
        distance = np.linalg.norm(displacement, axis=1)
        too_large = distance > self.config.max_node_displacement
        if np.any(too_large):
            scale = self.config.max_node_displacement / (distance[too_large] + 1e-12)
            displacement[too_large] *= scale[:, None]
        return previous_positions + displacement

    def _project_inextensible_lengths(self, positions: np.ndarray) -> np.ndarray:
        """Project all segments onto the fixed rest-segment length constraint.

        This is a simple position-based dynamics projection. Repeating it many
        times makes the discrete DLO nearly inextensible while still allowing
        the whole rod, including endpoints, to move when ``fix_endpoints`` is
        false.
        """

        corrected = positions.copy()
        fixed_nodes = np.zeros(self.config.n_nodes, dtype=bool)
        if self.config.fix_endpoints:
            fixed_nodes[0] = True
            fixed_nodes[-1] = True

        for _ in range(self.config.constraint_iterations):
            if self.config.fix_endpoints:
                corrected[0] = self.initial_positions[0]
                corrected[-1] = self.initial_positions[-1]

            for i in range(self.config.n_nodes - 1):
                self._project_one_segment(corrected, fixed_nodes, i)
            for i in range(self.config.n_nodes - 2, -1, -1):
                self._project_one_segment(corrected, fixed_nodes, i)

        return corrected

    def _project_one_segment(
        self,
        positions: np.ndarray,
        fixed_nodes: np.ndarray,
        left_index: int,
    ) -> None:
        """Project one segment so its length matches ``rest_segment_length``."""

        right_index = left_index + 1
        edge = positions[right_index] - positions[left_index]
        current_length = np.linalg.norm(edge)
        if current_length < 1e-12:
            return

        direction = edge / current_length
        error = current_length - self.rest_segment_length
        if abs(error) < 1e-9:
            return
        correction = 0.5 * error * direction

        left_fixed = fixed_nodes[left_index]
        right_fixed = fixed_nodes[right_index]

        if left_fixed and not right_fixed:
            positions[right_index] -= 2.0 * correction
        elif right_fixed and not left_fixed:
            positions[left_index] += 2.0 * correction
        elif not left_fixed and not right_fixed:
            positions[left_index] += correction
            positions[right_index] -= correction

    def _enforce_segment_lengths(self, positions: np.ndarray) -> np.ndarray:
        """Compatibility wrapper for the non-inextensible soft constraint mode."""

        return self._project_inextensible_lengths(positions)

    def _smooth_bends(self, positions: np.ndarray) -> np.ndarray:
        """Gently reduce sharp local kinks without changing the rollout API."""

        smoothed = positions.copy()
        for i in range(1, self.config.n_nodes - 1):
            neighbor_midpoint = 0.5 * (positions[i - 1] + positions[i + 1])
            smoothed[i] += self.config.bend_smoothing * (neighbor_midpoint - positions[i])
        return smoothed

    def _clip_force_locations(self, force_locations: np.ndarray) -> np.ndarray:
        """Keep continuous force locations inside valid node-index bounds."""

        return np.clip(force_locations, 0.0, self.config.n_nodes - 1.0)

    def _add_external_forces(
        self,
        forces: np.ndarray,
        force_locations: np.ndarray,
        step_forces: np.ndarray,
    ) -> None:
        """Distribute each external force to neighboring nodes by interpolation."""

        for force_index, location in enumerate(force_locations):
            left = int(np.floor(location))
            left = int(np.clip(left, 0, self.config.n_nodes - 1))
            right = min(left + 1, self.config.n_nodes - 1)
            alpha = float(location - left)
            force = self._clip_force(step_forces[force_index]).copy()
            force *= self.config.force_gain

            if left == right:
                forces[left] += force
            else:
                forces[left] += (1.0 - alpha) * force
                forces[right] += alpha * force

    def _clip_force(self, force: np.ndarray) -> np.ndarray:
        """Limit force magnitude so the optimizer cannot create explosive pulls."""

        magnitude = np.linalg.norm(force)
        if magnitude <= self.config.force_max:
            return force.copy()
        return force * (self.config.force_max / (magnitude + 1e-12))


class PyElasticaReplaySimulator:
    """Optional PyElastica replay backend for validating optimized force plans.

    This class is intentionally not used inside the pymoo objective. The simple
    optimizer first finds ``force_locations`` and ``force_sequence``; then this
    replay backend tries to run the same inputs through a PyElastica Cosserat
    rod and returns 2D snapshots with shape ``(horizon + 1, n_nodes, 2)``.
    """

    def __init__(self, config: DemoConfig):
        """Store configuration and check that PyElastica can be imported."""

        self.config = config
        self.initial_positions = make_straight_line(config.n_nodes, config.length)
        self.rest_length = config.length / (config.n_nodes - 1)
        self._elastica = self._require_pyelastica()

    def rollout(self, force_locations: np.ndarray, force_sequence: np.ndarray) -> np.ndarray:
        """Replay forces with PyElastica and return x-y rod node snapshots.

        ``force_locations`` may be either fixed with shape ``(n_forces,)`` or
        time-varying with shape ``(horizon, n_forces)``. ``force_sequence`` keeps
        the same meaning as the simple backend: ``(horizon, n_forces, 2)``.
        """

        force_locations = self._prepare_force_locations(force_locations, force_sequence.shape[0])
        nodal_forces = self._build_nodal_force_sequence(force_locations, force_sequence)
        return self._run_pyelastica_replay(nodal_forces)

    def _require_pyelastica(self):
        """Import PyElastica symbols lazily and provide a beginner-friendly error."""

        try:
            from elastica import (
                BaseSystemCollection,
                Constraints,
                CosseratRod,
                Damping,
                Forcing,
                PositionVerlet,
                finalize,
                integrate,
            )
            from elastica.dissipation import AnalyticalLinearDamper
            from elastica.external_forces import NoForces
        except ImportError as exc:
            raise ImportError(
                "PyElastica replay requested, but pyelastica is not installed. "
                "Install it with: pip install pyelastica"
            ) from exc

        return {
            "BaseSystemCollection": BaseSystemCollection,
            "Constraints": Constraints,
            "CosseratRod": CosseratRod,
            "Damping": Damping,
            "Forcing": Forcing,
            "PositionVerlet": PositionVerlet,
            "finalize": finalize,
            "integrate": integrate,
            "AnalyticalLinearDamper": AnalyticalLinearDamper,
            "NoForces": NoForces,
        }

    def _prepare_force_locations(
        self,
        force_locations: np.ndarray,
        horizon: int,
    ) -> np.ndarray:
        """Return force locations with shape ``(horizon, n_forces)``."""

        locations = np.asarray(force_locations, dtype=float)
        if locations.ndim == 1:
            locations = np.repeat(locations[None, :], horizon, axis=0)
        elif locations.shape != (horizon, self.config.n_forces):
            raise ValueError(
                "force_locations must have shape (n_forces,) or "
                "(horizon, n_forces)"
            )
        return np.clip(locations, 0.0, self.config.n_nodes - 1.0)

    def _build_nodal_force_sequence(
        self,
        force_locations: np.ndarray,
        force_sequence: np.ndarray,
    ) -> np.ndarray:
        """Convert interpolated point forces into PyElastica nodal forces."""

        nodal_forces = np.zeros((force_sequence.shape[0], 3, self.config.n_nodes))
        for step in range(force_sequence.shape[0]):
            for force_index, location in enumerate(force_locations[step]):
                left = int(np.floor(location))
                left = int(np.clip(left, 0, self.config.n_nodes - 1))
                right = min(left + 1, self.config.n_nodes - 1)
                alpha = float(location - left)

                force_xy = self._clip_force(force_sequence[step, force_index])
                force_xy = force_xy * self.config.force_gain
                force_xyz = np.array([force_xy[0], force_xy[1], 0.0])

                if left == right:
                    nodal_forces[step, :, left] += force_xyz
                else:
                    nodal_forces[step, :, left] += (1.0 - alpha) * force_xyz
                    nodal_forces[step, :, right] += alpha * force_xyz
        return nodal_forces

    def _run_pyelastica_replay(self, nodal_forces: np.ndarray) -> np.ndarray:
        """Create a straight Cosserat rod and replay the force sequence."""

        elastica = self._elastica
        base_classes = (
            elastica["BaseSystemCollection"],
            elastica["Constraints"],
            elastica["Forcing"],
            elastica["Damping"],
        )

        class ReplaySystem(*base_classes):
            """Small PyElastica system containing one rod."""

        class StepNodalForces(elastica["NoForces"]):
            """External force object that reads the current planning step."""

            step_index = 0

            def __init__(self, force_table: np.ndarray):
                self.force_table = force_table

            def apply_forces(self, system, time: float = 0.0):
                system.external_forces += self.force_table[type(self).step_index]

        simulator = ReplaySystem()
        shear_modulus = self.config.young_modulus / (2.0 * (1.0 + self.config.poisson_ratio))
        rod = elastica["CosseratRod"].straight_rod(
            n_elements=self.config.n_nodes - 1,
            start=np.array([0.0, 0.0, 0.0]),
            direction=np.array([1.0, 0.0, 0.0]),
            normal=np.array([0.0, 0.0, 1.0]),
            base_length=self.config.rod_length,
            base_radius=self.config.rod_radius,
            density=self.config.density,
            youngs_modulus=self.config.young_modulus,
            shear_modulus=shear_modulus,
        )
        simulator.append(rod)

        simulator.add_forcing_to(rod).using(StepNodalForces, nodal_forces)
        simulator.dampen(rod).using(
            elastica["AnalyticalLinearDamper"],
            damping_constant=0.05,
            time_step=self.config.dt,
        )
        elastica["finalize"](simulator)

        timestepper = elastica["PositionVerlet"]()
        snapshots = [self._rod_xy_positions(rod)]
        step_duration = self.config.dt * self.config.substeps
        total_steps = max(1, self.config.substeps)

        for step in range(nodal_forces.shape[0]):
            StepNodalForces.step_index = step
            elastica["integrate"](timestepper, simulator, step_duration, total_steps)
            snapshots.append(self._rod_xy_positions(rod))

        return np.asarray(snapshots)

    def _rod_xy_positions(self, rod) -> np.ndarray:
        """Convert PyElastica's 3D ``position_collection`` into ``(n_nodes, 2)``."""

        return rod.position_collection[:2, :].T.copy()

    def _clip_force(self, force: np.ndarray) -> np.ndarray:
        """Use the same vector magnitude limit as the simple backend."""

        magnitude = np.linalg.norm(force)
        if magnitude <= self.config.force_max:
            return force.copy()
        return force * (self.config.force_max / (magnitude + 1e-12))
