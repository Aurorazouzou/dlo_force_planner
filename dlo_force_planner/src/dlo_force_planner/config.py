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
    length: float = 1.0
    target_height: float = 0.24

    # Planning problem.
    horizon: int = 12
    force_nodes: tuple[int, int, int] = (8, 10, 12)
    force_limit: float = 7.0

    # Simple dynamics. One planning step is split into smaller integration
    # steps to keep the explicit Euler update stable.
    dt: float = 0.025
    substeps: int = 8
    mass: float = 1.0
    damping: float = 1.8
    spring_stiffness: float = 700.0
    bending_stiffness: float = 35.0

    # Cost weights. Shape error is intentionally dominant; the others make the
    # solution less violent and less jagged.
    shape_weight: float = 250.0
    force_weight: float = 0.015
    smooth_weight: float = 0.12
    curvature_weight: float = 0.015

    # GA settings. Increase n_generations for prettier convergence.
    population_size: int = 50
    n_generations: int = 35
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
