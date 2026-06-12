"""Convenience wrapper for running the nested demo project.

This file lets you run ``python scripts/run_demo.py`` from the outer
``dlo_force_planner`` folder. The real implementation lives in
``dlo_force_planner/scripts/run_demo.py``.
"""

from pathlib import Path
import runpy


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INNER_SCRIPT = PROJECT_ROOT / "dlo_force_planner" / "scripts" / "run_demo.py"

runpy.run_path(str(INNER_SCRIPT), run_name="__main__")
