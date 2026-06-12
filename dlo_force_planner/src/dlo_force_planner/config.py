"""Configuration values for the minimal DLO force-planning demo.

Keeping the numbers in one dataclass makes the beginner-facing code easier to
change: try editing one value here, then run ``python scripts/run_demo.py``.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DemoConfig:
    """All tunable parameters used by the demo.

    The defaults are chosen so the script finishes quickly on a laptop while
    still producing a visible upward bend in the DLO.
    """

    # Geometry of the simulated line.
    n_nodes: int = 21
    length: float = 0.05
    target_height: float = 0.012
    fix_endpoints: bool = False

    # Ecoflex-like material notes. The simulator below is still a simplified
    # 2D model, but these values document the material we are imitating.
    material_name: str = "Ecoflex 00-30"
    density: float = 1070.0
    young_modulus: float = 1.0e5
    poisson_ratio: float = 0.49
    rod_radius: float = 0.0002
    rod_length: float = 0.05
    inextensible: bool = True
    constraint_iterations: int = 20
    quasi_static: bool = True

    # Planning problem.
    horizon: int = 12
    n_forces: int = 3
    force_limit: float = 1.0e-5
    force_max: float = 1.0e-5
    min_force_separation: float = 2.0

    # Simple dynamics and Ecoflex-like regularization. One planning step is
    # split into smaller integration steps to keep the explicit Euler update
    # stable and visually rod-like.
    dt: float = 0.01
    substeps: int = 6
    mass: float = 3.2e-7
    damping: float = 0.8
    stretch_stiffness: float = 1.0
    bend_stiffness: float = 0.01
    force_gain: float = 1.0
    max_node_displacement: float = 5.0e-4
    bend_smoothing: float = 0.08

    # Cost weights. Shape error is intentionally dominant; the others make the
    # solution less violent and less jagged.
    shape_weight: float = 900.0
    force_weight: float = 0.006
    smooth_weight: float = 0.35
    curvature_weight: float = 8.0
    length_weight: float = 5000.0
    separation_weight: float = 1.0
    trajectory_weight: float = 0.0
    path_smoothness_weight: float = 100.0
    progress_weight: float = 20.0
    net_force_weight: float = 10.0
    net_torque_weight: float = 10.0

    # GA settings. Increase n_generations for prettier convergence.
    population_size: int = 40
    n_generations: int = 80
    random_seed: int = 7

    # Output folders.
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    @property
    def figure_dir(self) -> Path:
        """Folder used for PNG figures."""

        return self.output_dir / "figures"

    @property
    def animation_dir(self) -> Path:
        """Folder used for GIF animations."""

        return self.output_dir / "animations"

    @property
    def data_dir(self) -> Path:
        """Folder used for CSV files."""

        return self.output_dir / "data"
