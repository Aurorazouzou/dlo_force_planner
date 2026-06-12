"""Force-sequence optimizer based on pymoo's genetic algorithm."""

from dataclasses import dataclass

import numpy as np

from .config import DemoConfig
from .costs import total_cost
from .simulator import SimpleDLOSimulator


@dataclass
class PlannerResult:
    """All important data produced by the planner."""

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
    n_force_nodes = len(config.force_nodes)
    n_variables = config.horizon * n_force_nodes * 2

    class DLOForceProblem(ElementwiseProblem):
        """pymoo problem wrapper around one DLO rollout."""

        def __init__(self):
            super().__init__(
                n_var=n_variables,
                n_obj=1,
                xl=-config.force_limit,
                xu=config.force_limit,
            )

        def _evaluate(self, x, out, *args, **kwargs):
            force_sequence = x.reshape(config.horizon, n_force_nodes, 2)
            rollout = simulator.rollout(force_sequence)
            cost, _ = total_cost(
                rollout.positions,
                target_shape,
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

    best_force_sequence = result.X.reshape(config.horizon, n_force_nodes, 2)
    rollout = simulator.rollout(best_force_sequence)
    best_cost, components = total_cost(
        rollout.positions,
        target_shape,
        best_force_sequence,
        config,
    )

    return PlannerResult(
        best_force_sequence=best_force_sequence,
        best_cost=float(best_cost),
        cost_components=components,
        convergence=np.asarray(callback.data["best_cost"], dtype=float),
        snapshots=rollout.snapshots,
        final_shape=rollout.positions,
    )
