"""Simple 1D conduction + surface energy balance model.

This is a compact implementation intended for demonstration and teaching.
"""
import numpy as np
from dataclasses import dataclass
import json
from pathlib import Path
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
    # evap_coeff removed - latent handled via Bowen ratio (beta) in solver kwargs


def load_material(key: str, path: str = "materials.json") -> Material:
    base = Path(__file__).parent
    with open(base / path, "r") as f:
        data = json.load(f)
    d = data[key]
    # construct Material explicitly (materials.json no longer contains evap_coeff)
    return Material(name=d["name"], k=d["k"], rho=d["rho"], cp=d["cp"], albedo=d.get("albedo", 0.2), emissivity=d.get("emissivity", 0.9))


def diurnal_forcing(t, Ta_mean=293.15, Ta_amp=5.0, Sb=800.0, trise=7 * hour, tset=21 * hour):
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


def _sebrhs(t, T, C, k, dx, Sb, trise, tset, Ldown, alpha, epsilon, sigma, h, Ta_mean, Ta_amp=0.0):
    """RHS following MATLAB SEB_diurnal: returns dTdt array.

    Parameters correspond to MATLAB function:
    SEB_diurnal(t, T, C, k, dx, Sb, trise, tset, Ldown, alpha, epsilon, sigma, h, Ta)
    """
    kappa = k / C
    nz = len(T)
    dTdt = np.zeros_like(T)
    # interior nodes (1..nz-2)
    for i in range(1, nz - 1):
        dTdt[i] = kappa * (T[i - 1] - 2 * T[i] + T[i + 1]) / dx ** 2

    # SEB at surface
    Kdown = kdown(t, Sb, trise, tset)
    Kup = alpha * Kdown
    Lup = epsilon * sigma * T[0] ** 4
    # compute instantaneous atmospheric temperature (K) with peak at midpoint between trise and tset
    t24 = np.mod(t, 24 * hour)
    tmid = 0.5 * (trise + tset)
    phase = (t24 - tmid) / (24 * hour)
    Ta_inst = Ta_mean + Ta_amp * np.cos(2 * np.pi * phase)
    QE = h * (T[0] - Ta_inst)

    QG = Kdown - Kup + Ldown - Lup - QE

    # BCs: f = - QG / kappa / C (MATLAB) -> simplifies to -QG/k
    f = -QG / k
    dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)
    # bottom insulated
    dTdt[-1] = 0
    return dTdt


def _sebrhs_green(t, T, C, k, dx, Sb, trise, tset, Ldown, alpha, epsilon, sigma, h, Ta_mean, Ta_amp=0.0, beta=0.5):
    # same as _sebrhs but treat latent Q as Qh/beta where Qh = h*(T(1)-Ta)
    kappa = k / C
    nz = len(T)
    dTdt = np.zeros_like(T)
    for i in range(1, nz - 1):
        dTdt[i] = kappa * (T[i - 1] - 2 * T[i] + T[i + 1]) / dx ** 2

    Kdown = kdown(t, Sb, trise, tset)
    Kup = alpha * Kdown
    Lup = epsilon * sigma * T[0] ** 4
    t24 = np.mod(t, 24 * hour)
    tmid = 0.5 * (trise + tset)
    phase = (t24 - tmid) / (24 * hour)
    Ta_inst = Ta_mean + Ta_amp * np.cos(2 * np.pi * phase)
    Qh = h * (T[0] - Ta_inst)
    QE = Qh / beta if beta != 0 else 0.0

    QG = Kdown - Kup + Ldown - Lup - Qh - QE

    f = -QG / k
    dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)
    dTdt[-1] = 0
    return dTdt


def run_simulation(material: Material,
                   thickness=0.2,
                   dt=60.0,
                   tmax=24*hour,
                   Ta_forcing=None,
                   solver: str = "ode",
                   **solver_kwargs):
    """Run simulation using a stiff ODE solver (scipy.solve_ivp with BDF).

    The RHS mirrors the MATLAB `SEB_diurnal` (or `SEB_diurnal_green` when beta provided).
    Explicit finite-difference code removed; use this ODE path for stability and parity with
    MATLAB's stiff integrator (ode15s ~ BDF in scipy).
    """
    # fixed discretisation: 100 cells across the roof (as requested)
    n_cells = 100
    nz = n_cells
    dz = thickness / (nz - 1)
    alpha = material.k / (material.rho * material.cp)
    # Set z so that the surface is at z=0 and depth increases negatively
    # (z[0] is surface, z[-1] is -thickness). This makes plotting with
    # z=0 at the top more natural (we keep the solver's T[0] as surface).
    z = np.linspace(0.0, -thickness, nz)

    # enforce solver choice

    # prepare initial condition
    T0 = np.ones(nz) * 293.15
    t_span = (0.0, float(tmax))
    t_eval = np.arange(0, tmax + dt, dt)

    # choose RHS depending on whether beta (Bowen ratio) given
    # extract Ta mean and amplitude for diurnal air temperature
    Ta_mean = solver_kwargs.get('Ta_mean', 293.15)
    Ta_amp = solver_kwargs.get('Ta_amp', 0.0)

    if "beta" in solver_kwargs:
        rhs = lambda t, T: _sebrhs_green(t, T, material.rho * material.cp, material.k, dz,
                                         solver_kwargs.get("Sb", 800), solver_kwargs.get("trise", 7*hour),
                                         solver_kwargs.get("tset", 21*hour), solver_kwargs.get("Ldown", 350),
                                         material.albedo, material.emissivity, 5.670374419e-8,
                                         solver_kwargs.get("h", 10), Ta_mean, Ta_amp,
                                         solver_kwargs.get("beta", 0.5))
    else:
        rhs = lambda t, T: _sebrhs(t, T, material.rho * material.cp, material.k, dz,
                                   solver_kwargs.get("Sb", 800), solver_kwargs.get("trise", 7*hour),
                                   solver_kwargs.get("tset", 21*hour), solver_kwargs.get("Ldown", 350),
                                   material.albedo, material.emissivity, 5.670374419e-8,
                                   solver_kwargs.get("h", 10), Ta_mean, Ta_amp)

    # use a stiff solver (BDF) with conservative max_step
    sol = solve_ivp(rhs, t_span, T0, t_eval=t_eval, method='BDF', atol=1e-6, rtol=1e-6, max_step=solver_kwargs.get('max_step', 600.0))

    Ts = sol.y[0, :]
    times = sol.t

    # compute fluxes post-hoc
    G = np.zeros_like(times)
    H = np.zeros_like(times)
    E = np.zeros_like(times)
    L = np.zeros_like(times)
    Qstar = np.zeros_like(times)

    for i, t in enumerate(times):
        # use the same Sb and timing as the RHS to compute Kdown (parity with RHS)
        Sb_val = solver_kwargs.get('Sb', 800.0)
        trise_val = solver_kwargs.get('trise', 7 * hour)
        tset_val = solver_kwargs.get('tset', 21 * hour)
        Kdown = kdown(t, Sb_val, trise_val, tset_val)
        # shortwave absorbed (net shortwave = Kdown - Kup) with Kup = alpha * Kdown
        S_abs = (1 - material.albedo) * Kdown
        # instantaneous air temperature (diurnal) using the same convention as RHS
        Ta_mean_val = solver_kwargs.get('Ta_mean', 293.15)
        Ta_amp_val = solver_kwargs.get('Ta_amp', 0.0)
        t24 = np.mod(t, 24 * hour)
        tmid = 0.5 * (trise_val + tset_val)
        phase = (t24 - tmid) / (24 * hour)
        Ta = Ta_mean_val + Ta_amp_val * np.cos(2 * np.pi * phase)

        sigma = 5.670374419e-8
        Lup = material.emissivity * sigma * (Ts[i]) ** 4
        # use incoming longwave parameter (Ldown) as provided (W/m2)
        L_down = solver_kwargs.get('Ldown', 350.0)
        L_net = L_down - Lup
        # sensible heat using h coefficient
        hcoef = solver_kwargs.get('h', 10.0)
        Hflux = hcoef * (Ts[i] - Ta)

        # latent (green roof Bowen partition)
        if 'beta' in solver_kwargs and solver_kwargs.get('beta', None) is not None:
            Qh = Hflux
            Eflux = Qh / solver_kwargs.get('beta', 0.5)
        else:
            Eflux = 0.0

        # conductive flux approximated by -k * (T1 - T0)/dz
        T_full = sol.y[:, i]
        dTdz = (T_full[1] - T_full[0]) / dz
        Gflux = -material.k * dTdz

        G[i] = Gflux
        H[i] = Hflux
        E[i] = Eflux
        L[i] = L_net
        # Qstar should represent net shortwave absorbed (K*), not include
        # longwave terms. Keep longwave in L separately.
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
    out = run_simulation(m, thickness=0.2, tmax=3600 * 6)
    print("Done: sample run, final Ts:", out["Ts"][-1])
