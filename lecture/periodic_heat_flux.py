"""Simple periodic heat-flux demo using the model's sebrhs_ins and a BDF solver.

This script simulates a single dry sandy soil layer subject to a sinusoidal
surface heat flux Q(t) = Q0 * sin(omega * t) and animates the vertical
temperature profile. It intentionally uses `sebrhs_ins` as the RHS and
`scipy.integrate.solve_ivp(method='BDF')` as the stiff solver.

Run as:
    python examples/periodic_heat_flux.py

The script is self-contained and defines material and simulation parameters
near the top for easy editing.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp

# --- Standalone small model pieces copied/adapted from model.py ---
k =  0.3                # thermal conductivity (W/m/K)
rho = 1600               # density (kg/m^3)
cp = 800                 # specific heat capacity (J/kg/K)
albedo = 0.17            # surface albedo
emissivity = 0.95        # surface emissivity
sigma = 5.670374419e-8   # Stefan-Boltzmann constant (W/m^2/K^4)
thickness = 0.5          # layer thickness (m)
Nz = 80                  # vertical resolution (cells)
T0 = 293.15              # initial temperature (K)
hour = 3600              # seconds in an hour
T = 24 * hour            # period (s)
omega = 2.0 * np.pi / T  # angular frequency (rad/s)
Q0 = 200.0               # amplitude of imposed surface heat flux (W/m2)
h = 10.0                 # heat transfer coefficient (W/m2/K)
tmax = 2 * T             # total simulation time (s)
dt = 600.0               # solver output timestep (s)

def sebrhs_ins(t, T, k, rho, cp, dx, Qgfun, h):
    """RHS implementing surface-forced heat diffusion with insulating bottom.

    This simplified RHS no longer computes radiative terms. Instead it expects
    Qgfun(t) to provide the net surface ground heat flux QG (W/m2) as a
    function of time. The function intentionally omits any direct treatment
    of air temperature or latent heat (beta) — those should be folded into
    Qgfun if needed.
    """
    C = float(rho) * float(cp)
    kappa = float(k) / C

    nz = len(T)
    dTdt = np.zeros_like(T)
    for i in range(1, nz - 1):
        dTdt[i] = kappa * (T[i - 1] - 2 * T[i] + T[i + 1]) / dx ** 2

    # obtain imposed net ground flux from user-supplied function
    QG = float(Qgfun(t))

    # map QG to an equivalent ghost-node value f = -QG / k
    f = -QG / float(k)
    dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)

    # insulating bottom boundary
    dTdt[-1] = -2 * kappa / dx * ((T[-1] - T[-2]) / dx)
    return dTdt

# define the prescribed net ground heat-flux function Qgfun(t)
# here we directly impose a sinusoidal ground flux (W/m2). The RHS no
# longer computes radiation — Qgfun should supply the net QG(t).
Qgfun = lambda t: Q0 * np.sin(omega * t)

# --- Build material and grid --------------------------------------------
# material parameters are provided as primitive variables (k, rho, cp, albedo, emissivity)

dz = thickness / (Nz - 1)
z = np.linspace(0.0, -thickness, Nz)

# characteristic penetration depth and temperature
C = rho * cp
d_omega = np.sqrt(k / (C * omega))
Theta = Q0 * d_omega / k

# initial condition
T_init = np.ones(Nz) * T0

# RHS wrapper for solve_ivp that calls sebrhs_ins with primitive material params
def rhs(t, T):
    return sebrhs_ins(t, T, k, rho, cp, dz, Qgfun, h)

# solve with BDF (stiff)
t_eval = np.arange(0.0, tmax + dt, dt)
sol = solve_ivp(rhs, (0.0, float(tmax)), T_init, method='BDF', t_eval=t_eval, atol=1e-6, rtol=1e-6)

# prepare data for animation (convert to °C for plotting)
T_profiles = sol.y.T - 273.15
times = sol.t / hour  # hours for labels

# --- Matplotlib animation -----------------------------------------------
fig, ax = plt.subplots(figsize=(5, 6))
line, = ax.plot([], [], '-r', lw=2)
marker, = ax.plot([], [], 'ko')
ax.set_xlim(np.min(T_profiles) - 1, np.max(T_profiles) + 1)
ax.set_ylim(float(np.min(z)), 0.1)
ax.set_xlabel('Temperature (°C)')
ax.set_ylabel('Depth (m)')
title = ax.set_title(f'Time = {times[0]:.2f} h — d_ω={d_omega:.3f} m, Θ={Theta:.3f}')
ax.grid(True, linestyle=':', alpha=0.4)

# dashed horizontal line showing the penetration depth d_omega (plotted at -d_omega)
# store the artists so they can be returned by the init function when blitting
hline = ax.axhline(y=-5*d_omega, color='k', linestyle='--', label=r'$5 d_{\omega}$')
# surface layer indicator at z=0
surface_line = ax.axhline(y=0.0, color='k', linewidth=1.0, label='surface layer')
legend = ax.legend(loc='upper right')

# initial plot
def init():
    line.set_data([], [])
    marker.set_data([], [])
    title.set_text(f'Time = {times[0]:.2f} h — d_ω={d_omega:.3f} m, Θ={Theta:.3f}')
    # include static artists so they are always drawn
    return line, marker, title, hline, surface_line, legend

# update function for each frame index
def update(i):
    T = T_profiles[i]
    line.set_data(T, z)
    # set_data expects sequences; provide 1-element lists for the marker
    marker.set_data([T[0]], [0.0])
    title.set_text(f'Time = {times[i]:.2f} h — d_ω={d_omega:.3f} m, Θ={Theta:.3f}')
    return line, marker, title

ani = FuncAnimation(fig, update, frames=len(times), init_func=init, blit=False, interval=80)

plt.tight_layout()
plt.show()
