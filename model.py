"""Simple 1D conduction + surface energy balance model.

This is a compact implementation intended for demonstration and teaching.
"""
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Any, Union, Optional
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# time unit helper
hour = 3600

@dataclass
class Material:
    name: str
    k: float
    rho: float
    cp: float
    albedo: float
    emissivity: float
    evaporation: bool

# Module-level defaults so model can run standalone without importing `app`.
DEFAULTS = {
    'thickness': 1.0,           # layer thickness (m)
    'T0': 293.15,               # initial soil temperature (K)
    'Sb': 1000.0,               # peak shortwave (W/m2)
    'trise': 7*hour,            # sunrise time (s)
    'tset': 21*hour,            # sunset time (s)
    'Ldown': 350.0,             # downwelling longwave (W/m2)
    'sigma': 5.670374419e-8,    # Stefan-Boltzmann constant (W/m2/K4)
    'Ta_mean': 293.15,          # mean air temperature (K)
    'Ta_amp': 5.0,              # amplitude of air temperature variation (K)
    'h': 20.0,                  # heat-transfer coefficient (W/m2/K)
    'beta': 0.5,                # Bowen ratio (-)
    'tmax': 48*hour,            # simulation time (s)
    'dt': 0.5*hour,             # time step for output (s)
    'Nz': 100,                  # default number of vertical cells
    'insulating_layer': True,   # switch for bottom boundary condition
}

def load_material(key: str, path: str = "materials.json") -> Material:
    """Load material definition from JSON and return a Material instance.

    This function assumes `materials.json` is correct and contains the
    required fields. If a required field is missing, a KeyError will
    naturally be raised by the dict access.
    """
    base = Path(__file__).parent
    with open(base / path, "r", encoding="utf8") as f:
        data = json.load(f)
    d = data[key]
    return Material(name=d["name"], k=d["k"], rho=d["rho"], cp=d["cp"], albedo=d["albedo"], emissivity=d["emissivity"], evaporation=d["evaporation"])


def diurnal_forcing(t, Ta_mean, Ta_amp, Sb, trise, tset):
    """Return atmospheric temperature (K) and a simple shortwave shape S0 (W/m2).

    Ta_mean : mean air temperature in K
    Ta_amp  : amplitude (half the peak-to-trough) in K
    Sb      : peak shortwave (W/m2) used only for the S0 shape returned here
    trise,tset : rise/set times in seconds used by kdown
    """
    # align Ta peak with shortwave peak: midpoint between trise and tset
    t = np.asarray(t)
    t24 = np.mod(t, 24 * hour)
    tmid = 0.5 * (trise + tset)
    phase = (t24 - tmid) / (24 * hour)
    Ta = Ta_mean + Ta_amp * np.cos(2 * np.pi * phase)

    # shortwave shape (kept simple here) - use kdown timings for parity
    S0 = kdown(t, Sb, trise, tset)
    # clip negative values to zero (night)
    S0 = np.maximum(0.0, S0)
    return Ta, S0


def kdown(t, Sb, trise, tset):
    """Compute incoming shortwave Kdown following the same timing logic used in SEB_diurnal.

    t : scalar or array in seconds
    Sb : peak shortwave (W/m2)
    trise, tset : times in seconds
    """
    # bring into 0..24h window
    t24 = np.mod(t, 24 * hour)
    # same theta mapping used previously; supports array input
    theta = (t24 - trise) / (tset - trise) * np.pi / 2 + (t24 - tset) / (tset - trise) * np.pi / 2
    theta = np.minimum(np.maximum(theta, -np.pi / 2), np.pi / 2)
    return Sb * np.cos(theta)



def sebrhs_noins(t, T, mat: Material, dx, forcing, h, beta=0.0, sigma=5.670374419e-8):
    """
    RHS that accepts a Material instance `mat` and uses an externally
    supplied `forcing` mapping for interpolation. The `forcing` object
    must provide arrays for time, air temperature and shortwave (Kdown).
    Latent heat is treated via Bowen ratio `beta` (E = Qh / beta).
    """
    # material properties
    k = mat.k
    C = mat.rho * mat.cp
    alpha = mat.albedo
    epsilon = mat.emissivity

    kappa = k / C
    Nz = len(T)
    dTdt = np.zeros_like(T)
    for i in range(1, Nz - 1):
        dTdt[i] = kappa * (T[i - 1] - 2 * T[i] + T[i + 1]) / dx ** 2

    # forcing is expected to be a mapping with keys 't', 'Ta', 'Kdown'
    times_arr = np.asarray(forcing['t'], dtype=float)
    S0_arr = np.asarray(forcing['Kdown'])
    Ta_arr = np.asarray(forcing['Ta'])
    Ldown_arr = np.asarray(forcing.get('Ldown'))

    # interpolate S0 (shortwave), Ta and Ldown at current time
    Kdown = np.interp(t, times_arr, S0_arr)
    Kup = alpha * Kdown
    Lup = epsilon * sigma * T[0] ** 4

    Ta = np.interp(t, times_arr, Ta_arr)
    Ldown = np.interp(t, times_arr, Ldown_arr)

    Qh = h * (T[0] - Ta)
    QE = Qh / beta if beta != 0 else 0.0

    QG = Kdown - Kup + Ldown - Lup - Qh - QE

    f = -QG / k
    dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)

    dTdt[-1] = 0
    return dTdt

def sebrhs_ins(t, T, mat: Material, dx, forcing, h, beta=0.0, sigma=5.670374419e-8):
    """
    RHS that accepts a Material instance `mat` and uses an externally
    supplied `forcing` mapping for interpolation. The `forcing` object
    must provide arrays for time, air temperature and shortwave (Kdown).
    Latent heat is treated via Bowen ratio `beta` (E = Qh / beta).
    """
    # material properties
    k = mat.k
    C = mat.rho * mat.cp
    alpha = mat.albedo
    epsilon = mat.emissivity

    kappa = k / C
    nz = len(T)
    dTdt = np.zeros_like(T)
    for i in range(1, nz - 1):
        dTdt[i] = kappa * (T[i - 1] - 2 * T[i] + T[i + 1]) / dx ** 2

    # forcing is expected to be a mapping with keys 't', 'Ta', 'Kdown'
    times_arr = np.asarray(forcing['t'], dtype=float)
    S0_arr = np.asarray(forcing['Kdown'])
    Ta_arr = np.asarray(forcing['Ta'])
    Ldown_arr = np.asarray(forcing.get('Ldown'))

    # interpolate S0 (shortwave), Ta and Ldown at current time
    Kdown = np.interp(t, times_arr, S0_arr)
    Kup = alpha * Kdown
    Lup = epsilon * sigma * T[0] ** 4

    Ta = np.interp(t, times_arr, Ta_arr)
    Ldown = np.interp(t, times_arr, Ldown_arr)

    Qh = h * (T[0] - Ta)
    QE = Qh / beta if beta != 0 else 0.0

    QG = Kdown - Kup + Ldown - Lup - Qh - QE

    f = -QG / k
    dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)

    # insulating bottom boundary
    dTdt[-1] = -2 * kappa / dx * ((T[-1] - T[-2]) / dx)
    return dTdt


def run_simulation(mat: Material,
                   params,
                   dt,
                   tmax,
                   ):
    """Run simulation using a stiff ODE solver (scipy.solve_ivp with BDF).

    The RHS mirrors the MATLAB `SEB_diurnal` (or `SEB_diurnal_green` when beta provided).
    Explicit finite-difference code removed; use this ODE path for stability and parity with
    MATLAB's stiff integrator (ode15s ~ BDF in scipy).
    """

    # require caller to provide these solver parameters; no defaults allowed
    required = ['beta', 'forcing', 'thickness', 'h']
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"Missing required solver_kwargs: {', '.join(missing)}")

    # read params dictionary
    hcoef = params['h']
    beta_default = params['beta']
    forcing = params['forcing']
    thickness = float(params['thickness'])

    # discretisation: allow override via params (fall back to DEFAULTS)
    Nz = DEFAULTS['Nz']
    dz = thickness / (Nz - 1)
    z = np.linspace(0.0, -thickness, Nz)

    # prepare initial condition: allow overriding initial temperature via params
    T_init = float(DEFAULTS['T0'])
    T0 = np.ones(Nz) * T_init

    # determine RHS and beta usage depending on evaporation capability
    # always pass a numeric beta to sebrhs; use 0.0 to disable evaporation
    beta_local = beta_default if mat.evaporation else 0.0

    if DEFAULTS['insulating_layer']:
        rhs = lambda t, T: sebrhs_ins(t, T, mat, dz, forcing, hcoef, beta_local)
    else:
        rhs = lambda t, T: sebrhs_noins(t, T, mat, dz, forcing, hcoef, beta_local)

    # determine solver evaluation times (t_eval) and t_span
    t_eval = np.arange(0.0, float(tmax) + dt, dt)
    t_span = (0.0, float(tmax))

    # use a stiff solver (BDF)
    sol = solve_ivp(rhs, t_span, T0, t_eval=t_eval, method='BDF', atol=1e-6, rtol=1e-6, max_step=dt)

    # times and surface temperature
    times = sol.t
    Ts = sol.y[0, :]

    # Vectorised post-processing of fluxes
    Ta_arr = np.interp(times, forcing['t'], forcing['Ta'])

    # shortwave partitioning
    Kdown = np.interp(times, forcing['t'], forcing['Kdown'])
    Kup = mat.albedo * Kdown
    Kstar = Kdown - Kup

    # longwave upwelling from surface
    sigma = DEFAULTS['sigma']
    Ldown = np.interp(times, forcing['t'], forcing['Ldown'])
    Lup = mat.emissivity * sigma * (Ts ** 4)
    Lstar = Ldown - Lup

    # sensible heat flux
    Hflux = hcoef * (Ts - Ta_arr)

    # latent flux: only compute if material allows evaporation
    if mat.evaporation:
        Eflux = Hflux / beta_local
    else:
        Eflux = np.zeros_like(Hflux)

    # conductive flux approximated by -k * (T1 - T0)/dz (vectorised)
    if sol.y.shape[0] >= 2:
        dTdz = (sol.y[1, :] - sol.y[0, :]) / dz
    else:
        dTdz = np.zeros_like(Ts)
    Gflux = -mat.k * dTdz

    # assemble outputs
    T_profiles = sol.y.T
    return {
        "t": times,
        "Ta": Ta_arr,
        "Ts": Ts,
        "Kstar": Kstar,
        "Kdown": Kdown,
        "Kup": Kup,
        "Lstar": Lstar,
        "Ldown": Ldown,
        "Lup": Lup,
        "G": Gflux,
        "H": Hflux,
        "E": Eflux,
        "L": Lstar,
        "z": z,
        "T_profile": sol.y[:, -1],
        "T_profiles": T_profiles
    }

if __name__ == "__main__":
    # import app only for the demo/run-as-script path to avoid circular
    # imports when `app` imports this module.
    m = load_material("sandy_dry")

    # build forcing time series (high-resolution for interpolation)
    tmax = int(DEFAULTS['tmax'])
    forcing_dt = 60.0  # 1 min forcing resolution
    forcing_t = np.arange(0.0, float(tmax) + forcing_dt, forcing_dt)
    Ta_arr, S0_arr = diurnal_forcing(forcing_t,
                                     Ta_mean=DEFAULTS['Ta_mean'],
                                     Ta_amp=DEFAULTS['Ta_amp'],
                                     Sb=DEFAULTS['Sb'],
                                     trise=DEFAULTS['trise'],
                                     tset=DEFAULTS['tset'])
    Ldown_arr = np.full_like(forcing_t, float(DEFAULTS['Ldown']))
    forcing = {'t': forcing_t, 'Ta': Ta_arr, 'Kdown': S0_arr, 'Ldown': Ldown_arr}

    params = {
        'beta': float(DEFAULTS['beta']),
        'h': float(DEFAULTS['h']),
        'forcing': forcing,
        'thickness': float(DEFAULTS['thickness']),
    }

    dt = float(DEFAULTS['dt'])
    out = run_simulation(m, params, dt, tmax=tmax)
    print("Done: sample run, final Ts:", out["Ts"][-1])

    # --- Plot Results (match style used in the GUI Results tab) ---
    t = out['t'] / hour

    fig, axs = plt.subplots(2, 1, figsize=(8, 6), squeeze=False)

    # Ts (top)
    ax_ts = axs[0][0]
    ax_ts.plot(t, out['Ts'] - 273.15, color='r')
    ax_ts.set_ylabel('T_surf (°C)')
    ax_ts.set_xlabel('Time (h)')
    ax_ts.grid(True, linestyle=':', alpha=0.5)

    # Energy panel (bottom)
    ax_en = axs[1][0]
    ax_en.plot(t, out['Kstar'], color='orange', label='K*')
    ax_en.plot(t, out['Lstar'], color='magenta', label='L*')
    ax_en.plot(t, out['H'], color='green', label='H')
    ax_en.plot(t, out['E'], color='blue', label='E')
    ax_en.plot(t, out['G'], color='saddlebrown', label='G')
    ax_en.set_ylabel('Flux (W/m2)')
    ax_en.set_xlabel('Time (h)')
    ax_en.legend(loc='upper right', fontsize='small')
    ax_en.grid(True, linestyle=':', alpha=0.5)

    fig.tight_layout()
    plt.show()