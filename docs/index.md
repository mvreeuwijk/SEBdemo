# SEBdemo - Documentation

Welcome to the SEBdemo documentation. This site is short and focused: the primary deliverable is the desktop demo application (`app.py`) which provides a friendly GUI for exploring material choices and the surface energy balance. The underlying numerical model is also available as a programmatic library (`model.py`) for scripted experiments.

## Quick links

- [Model overview](model.md)
- [Governing equations](governing_equations.md)
- [Software license](LICENSE.md)

## Which interface should I use?

- GUI (recommended for interactive exploration): run `app.py`. The GUI exposes inputs for materials, forcing and solver settings, and provides Results, Term-by-term and Animation views so you can visualise Ts, fluxes and vertical temperature profiles without writing code.
- Programmatic API (recommended for scripting and batch runs): import `model.py` and call `run_simulation(mat, params, dt, tmax)`. This gives direct access to the time series, flux diagnostics and full temperature profiles so you can embed the model in tests, analyses or automated workflows.

## Quick start

1. Create a virtual environment and install requirements:

```powershell
python -m pip install -r requirements.txt
```

2. Run the desktop app:

```powershell
python app.py
```

3. Or run the model from Python (script or REPL):

```python
from model import load_material, run_simulation
mat = load_material('sandy_dry')
forcing = ...  # build forcing dict (see model.py example)
params = {'beta': 0.5, 'h': 10.0, 'forcing': forcing, 'thickness': 1.0}
out = run_simulation(mat, params, dt=1800.0, tmax=48*3600)
```

## How to cite

If you use this software in published work, please cite the project. A short citation is:

SEB Roof Material Comparator (https://github.com/mvreeuwijk/SEBdemo)

