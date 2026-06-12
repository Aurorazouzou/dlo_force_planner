# DLO Force Planner

This project is a minimal runnable demo for planning external forces on a
deformable linear object (DLO). It intentionally uses a simple 2D simulator
instead of PyElastica, so the full pipeline is easy to read and run.

## Run

```bash
pip install -r requirements.txt
python scripts/run_demo.py
```

The demo writes figures, an animation, and CSV data into `outputs/`.
