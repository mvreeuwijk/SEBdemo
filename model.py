"""Simple 1D conduction + surface energy balance model.

This is a compact implementation intended for demonstration and teaching.
"""
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Any, Union
from scipy.integrate import solve_ivp

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



def sebrhs(t, T, mat: Material, dx, times_arr, S0_arr, Ta_arr, Ldown, h, beta=0.0, sigma=5.670374419e-8):
    """
    RHS that accepts a Material instance `mat` and uses interpolated
    shortwave (S0) and air temperature (Ta) arrays. Latent heat is
    treated via Bowen ratio `beta` (E = Qh / beta). 
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

    # interpolate S0 (shortwave) and Ta at current time
    Kdown = float(np.interp(t, times_arr, S0_arr)) 
    Kup = alpha * Kdown
    Lup = epsilon * sigma * T[0] ** 4

    Ta = float(np.interp(t, times_arr, Ta_arr))

    Qh = h * (T[0] - Ta)
    QE = Qh / beta if beta != 0 else 0.0

    QG = Kdown - Kup + Ldown - Lup - Qh - QE

    f = -QG / k
    dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)
    dTdt[-1] = 0
    return dTdt

def run_simulation(mat: Material,
                   dt=60.0,
                   tmax=24*hour,
                   solver: str = "ode",
                   **solver_kwargs):
    """Run simulation using a stiff ODE solver (scipy.solve_ivp with BDF).

    The RHS mirrors the MATLAB `SEB_diurnal` (or `SEB_diurnal_green` when beta provided).
    Explicit finite-difference code removed; use this ODE path for stability and parity with
    MATLAB's stiff integrator (ode15s ~ BDF in scipy).
    """
    # thickness should be supplied via solver_kwargs (preferred)
    thickness = float(solver_kwargs.pop('thickness', 0.2))
    # fixed discretisation: 100 cells across the roof (as requested)
    Nz = 100
    dz = thickness / (Nz - 1)
    z = np.linspace(0.0, -thickness, Nz)

    # prepare initial condition
    T0 = np.ones(Nz) * 293.15

    # (thickness already read above)

    # accept optional solver evaluation times (t_array)
    t_array = solver_kwargs.pop('t_array', None)

    # time evaluation points for the solver (t_eval) - still uses dt
    if t_array is not None:
        t_eval = np.asarray(t_array, dtype=float)
        t_span = (float(t_eval[0]), float(t_eval[-1]))
    else:
        t_span = (0.0, float(tmax))
        t_eval = np.arange(0, tmax + dt, dt)

    # forcing parameters (defaults)
    Sb = solver_kwargs.get('Sb', 800.0)
    trise = solver_kwargs.get('trise', 7 * hour)
    tset = solver_kwargs.get('tset', 21 * hour)
    Ta_mean = solver_kwargs.get('Ta_mean', 293.15)
    Ta_amp = solver_kwargs.get('Ta_amp', 0.0)
    Ldown = solver_kwargs.get('Ldown', 350.0)
    hcoef = solver_kwargs.get('h', 10.0)
    beta_default = solver_kwargs.get('beta', 0.5)
    max_step = solver_kwargs.get('max_step', 600.0)

    # Forcing time grid: use a higher-resolution grid for interpolation (default 60s)
    forcing_t = solver_kwargs.pop('forcing_t', None)
    forcing_dt = float(solver_kwargs.pop('forcing_dt', 60.0))

    # always construct (or accept) a high-resolution forcing time grid used
    # for interpolation inside the RHS. This avoids relying on callers to
    # provide coarse Ta/S0 arrays and keeps interpolation accurate.
    if forcing_t is None:
        forcing_t = np.arange(0.0, float(tmax) + forcing_dt, forcing_dt)
    Ta_arr, S0_arr = diurnal_forcing(forcing_t, Ta_mean=Ta_mean, Ta_amp=Ta_amp, Sb=Sb, trise=trise, tset=tset)
    times_arr = np.asarray(forcing_t, dtype=float)

    # determine RHS and beta usage depending on evaporation capability
    # always pass a numeric beta to sebrhs; use 0.0 to disable evaporation
    beta_local = beta_default if mat.evaporation else 0.0
    rhs = lambda t, T: sebrhs(t, T, mat, dz, times_arr, S0_arr, Ta_arr, Ldown, hcoef, beta_local)

    # use a stiff solver (BDF)
    sol = solve_ivp(rhs, t_span, T0, t_eval=t_eval, method='BDF', atol=1e-6, rtol=1e-6, max_step=max_step)

    Ts = sol.y[0, :]
    times = sol.t

    # compute fluxes post-hoc
    G = np.zeros_like(times)
    H = np.zeros_like(times)
    E = np.zeros_like(times)
    L = np.zeros_like(times)
    Qstar = np.zeros_like(times)

    for i, t in enumerate(times):
        # interpolate Kdown and Ta from the forcing arrays
        Kdown = float(np.interp(t, times_arr, S0_arr))
        S_abs = (1 - mat.albedo) * Kdown
        Ta = float(np.interp(t, times_arr, Ta_arr))

        sigma = 5.670374419e-8
        Lup = mat.emissivity * sigma * (Ts[i]) ** 4
        L_net = Ldown - Lup
        Hflux = hcoef * (Ts[i] - Ta)

        # latent flux: only compute if material allows evaporation
        if mat.evaporation:
            Eflux = Hflux / beta_local if beta_local != 0 else 0.0
        else:
            Eflux = 0.0

        # conductive flux approximated by -k * (T1 - T0)/dz
        T_full = sol.y[:, i]
        dTdz = (T_full[1] - T_full[0]) / dz
        Gflux = -mat.k * dTdz

        G[i] = Gflux
        H[i] = Hflux
        E[i] = Eflux
        L[i] = L_net
        Qstar[i] = S_abs

    # sol.y shape is (nz, nt); return a time-major T_profiles array (nt, nz)
    T_profiles = sol.y.T

    return {
        "times": times,
        "z": z,
        "T_profile": sol.y[:, -1],
        "T_profiles": T_profiles,
        "Ts": Ts,
        "G": G,
        "H": H,
        "E": E,
        "L": L,
        "Qstar": Qstar,
    }


if __name__ == "__main__":
    m = load_material("concrete")
    out = run_simulation(m, tmax=3600 * 6, thickness=0.2)
    print("Done: sample run, final Ts:", out["Ts"][-1])
