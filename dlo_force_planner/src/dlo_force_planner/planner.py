"""Force-sequence optimizer based on pymoo's genetic algorithm."""

from dataclasses import dataclass

import numpy as np

from .config import DemoConfig
from .costs import total_cost
from .simulator import SimpleDLOSimulator


@dataclass
class PlannerResult:
    """All important data produced by the planner."""

    best_force_locations: np.ndarray
    best_force_sequence: np.ndarray
    best_cost: float
    cost_components: dict[str, float]
    convergence: np.ndarray
    snapshots: np.ndarray
    final_shape: np.ndarray


def _require_pymoo():
    """Import pymoo lazily so missing dependencies produce a helpful message."""

    try:
        from pymoo.algorithms.soo.nonconvex.ga import GA
        from pymoo.core.callback import Callback
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.optimize import minimize
    except ImportError as exc:
        raise ImportError(
            "pymoo is required for this demo. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from exc

    return GA, Callback, ElementwiseProblem, minimize


def optimize_force_sequence(
    simulator: SimpleDLOSimulator,
    target_shape: np.ndarray,
    config: DemoConfig,
) -> PlannerResult:
    """Optimize a flattened force vector and return the best rollout."""

    GA, Callback, ElementwiseProblem, minimize = _require_pymoo()
    n_force_values = config.horizon * config.n_forces * 2
    n_variables = config.n_forces + n_force_values

    lower_bounds = np.concatenate(
        [
            np.full(config.n_forces, 0.0),
            np.full(n_force_values, -config.force_max),
        ]
    )
    upper_bounds = np.concatenate(
        [
            np.full(config.n_forces, config.n_nodes - 1.0),
            np.full(n_force_values, config.force_max),
        ]
    )

    def split_variables(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Split pymoo's flat vector into locations and time-varying forces."""

        force_locations = x[: config.n_forces]
        force_sequence = x[config.n_forces :].reshape(config.horizon, config.n_forces, 2)
        force_magnitude = np.linalg.norm(force_sequence, axis=2, keepdims=True)
        scale = np.minimum(1.0, config.force_max / (force_magnitude + 1e-12))
        force_sequence = force_sequence * scale
        return force_locations, force_sequence

    class DLOForceProblem(ElementwiseProblem):
        """pymoo problem wrapper around one DLO rollout."""

        def __init__(self):
            super().__init__(
                n_var=n_variables,
                n_obj=1,
                xl=lower_bounds,
                xu=upper_bounds,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            force_locations, force_sequence = split_variables(x)
            rollout = simulator.rollout(force_locations, force_sequence)
            cost, _ = total_cost(
                rollout.positions,
                rollout.snapshots,
                simulator.initial_positions,
                target_shape,
                force_locations,
                force_sequence,
                config,
            )
            out["F"] = cost

    class BestCostCallback(Callback):
        """Collect the best objective value after each GA generation."""

        def __init__(self):
            super().__init__()
            self.data["best_cost"] = []

        def notify(self, algorithm):
            self.data["best_cost"].append(float(algorithm.pop.get("F").min()))

    callback = BestCostCallback()
    algorithm = GA(pop_size=config.population_size, eliminate_duplicates=True)

    result = minimize(
        DLOForceProblem(),
        algorithm,
        ("n_gen", config.n_generations),
        seed=config.random_seed,
        callback=callback,
        verbose=False,
    )

    best_vector = np.asarray(result.X, dtype=float).reshape(-1).copy()
    best_force_locations, best_force_sequence = split_variables(best_vector)
    rollout = simulator.rollout(best_force_locations, best_force_sequence)
    best_cost, components = total_cost(
        rollout.positions,
        rollout.snapshots,
        simulator.initial_positions,
        target_shape,
        best_force_locations,
        best_force_sequence,
        config,
    )
    for index, location in enumerate(best_force_locations, start=1):
        components[f"force_location_{index}"] = float(location)
    components["force_location_start"] = float(best_force_locations[0])
    components["force_location_end"] = float(best_force_locations[-1])
    components["force_location_mean"] = float(np.mean(best_force_locations))

    return PlannerResult(
        best_force_locations=best_force_locations,
        best_force_sequence=best_force_sequence,
        best_cost=float(best_cost),
        cost_components=components,
        convergence=np.asarray(callback.data["best_cost"], dtype=float),
        snapshots=rollout.snapshots,
        final_shape=rollout.positions,
    )
