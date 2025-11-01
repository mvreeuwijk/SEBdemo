# Model: 1D conduction + Surface Energy Balance (SEB)

This document describes the simple 1D model implemented in `model.py` and how to use it.

Overview

- The model couples a vertically discretised 1D conductive heat equation with a surface energy balance (shortwave, longwave, sensible & latent, ground conductive flux).
- The numerical integration is performed via `scipy.integrate.solve_ivp` (BDF) for stability.

Key components

- `Material` dataclass: describes material properties (k, rho, cp, albedo, emissivity, evaporation flag).
- `DEFAULTS`: sensible default parameters (time span, dt, physical constants, Nz, initial T0) so the module can be executed as a script.
- `diurnal_forcing()` / `kdown()`: convenience functions to create a simple diurnal air temperature and shortwave time series.
- `sebrhs()`: right-hand-side function used by the ODE solver; computes surface fluxes by interpolating forcing arrays and evaluating the SEB.
- `run_simulation(mat, params, dt, tmax)`: the main programmatic entry point. It returns a dictionary with arrays used by the GUI and plots.

run_simulation inputs (brief)

- `mat`: a `Material` instance (use `load_material(key)` to read from `materials.json`).
- `params`: dict with keys including `forcing` (a dict with arrays `'t'`, `'Ta'`, `'Kdown'`, `'Ldown'`), `beta`, `h`, `thickness`, optionally `Nz` and `T0`.
- `dt`, `tmax`: timestep for outputs and final simulation time (seconds).

run_simulation outputs (dict)

- `t`: time array (s)
- `Ta`: air temperature array (K)
- `Ts`: surface temperature (K)
- `Kstar`, `Kdown`, `Kup`: shortwave net and components (W/m2)
- `Lstar`, `Ldown`, `Lup`: longwave net and components (W/m2)
- `H`, `E`, `G`: sensible, latent, conductive ground flux arrays (W/m2)
- `z`: vertical coordinates (m)
- `T_profiles`: system temperature profiles (Nt x Nz)

Quick start — run the demo (local)

1. Install dependencies from the repository `requirements.txt` (recommended in a virtualenv or conda env). Example:

```powershell
python -m pip install -r requirements.txt
```

2. Run the demo script (this uses the module-level `DEFAULTS` and opens a matplotlib figure showing the Results panels):

```powershell
python model.py
```

3. Programmatic usage

```python
from model import load_material, run_simulation, DEFAULTS
mat = load_material('concrete')
# build forcing dict (example: use diurnal_forcing)
import numpy as np
from model import diurnal_forcing, hour
tmax = int(DEFAULTS['tmax'])
forcing_t = np.arange(0, tmax + 600, 600)
Ta, S0 = diurnal_forcing(forcing_t, Ta_mean=DEFAULTS['Ta_mean'], Ta_amp=DEFAULTS['Ta_amp'], Sb=DEFAULTS['Sb'], trise=DEFAULTS['trise'], tset=DEFAULTS['tset'])
forcing = {'t': forcing_t, 'Ta': Ta, 'Kdown': S0, 'Ldown': np.full_like(forcing_t, DEFAULTS['Ldown'])}
params = {'forcing': forcing, 'beta': 0.5, 'h': 10.0, 'thickness': 0.2}
out = run_simulation(mat, params, dt=DEFAULTS['dt'], tmax=tmax)
print(out['Ts'][-1])
```

Notes & caveats

- The demo script in `model.py` displays plots using Matplotlib and intentionally avoids importing the GUI (`app.py`) to prevent circular imports.
- For interactive usage inside the GUI, the `app` module constructs the forcing arrays and passes them into `run_simulation` so the GUI and model use the identical forcing times for exact comparisons.

License / attribution

- See the repository `README.md` for license and author details.

