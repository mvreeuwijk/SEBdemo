# Governing equations and implementation notes

This document summarizes the mathematical model implemented in `model.py` (1D heat conduction with a surface energy balance). It gives the continuous equations, boundary conditions, flux parameterisations, and how these map to the code.

## 1) Model overview

- Domain: 1D vertical soil column, \( z \in [-H, 0] \) with \(z=0\) at the surface and \(z<0\) below the surface.
- Unknown: temperature \(T(z,t)\) (K).
- The model solves transient heat conduction in the soil coupled to a surface energy balance that provides the surface boundary condition.

## 2) Heat equation (continuous)

The soil obeys the diffusion (heat) equation in 1D:

$$ C\,\frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\!\left(k\,\frac{\partial T}{\partial z}\right). $$

where \( C = \rho c_p \) is the volumetric heat capacity (J m^{-3} K^{-1}), and \(k\) is the thermal conductivity (W m^{-1} K^{-1}). For homogeneous material with constant \(k\) and \(C\):

$$
 \frac{\partial T}{\partial t} = \kappa\,\frac{\partial^2 T}{\partial z^2},\quad \kappa = \frac{k}{C}.
$$

Implementation note: the code forms a finite-difference discretisation on \(N_z\) points and advances the resulting ODE system in time using `scipy.integrate.solve_ivp` (method='BDF'). See `run_simulation` for discretisation choices.

## 3) Surface boundary condition (surface energy balance)

At the surface (\(z = 0\)) the model enforces the surface energy balance (positive downward):

$$
 K_{\downarrow} - K_{\uparrow}
 + L_{\downarrow} - L_{\uparrow}
 - H - E - G = 0,
$$

where

- \(K_{\downarrow}\): incoming shortwave (W m^{-2})
- \(K_{\uparrow} = \alpha\,K_{\downarrow}\): reflected shortwave (albedo \(\alpha\))
- \(L_{\downarrow}\): downwelling longwave (W m^{-2})
- \(L_{\uparrow} = \varepsilon\,\sigma\,T_s^4\): upwelling longwave (emissivity \(\varepsilon\), Stefanâ€“Boltzmann constant \(\sigma\))
- \(H\): sensible heat flux (W m^{-2})
- \(E\): latent heat flux (W m^{-2})
- \(G\): conductive flux into the ground (W m^{-2}, positive downward)

In code the balance is evaluated as:

```python
Kdown = ...                 # forcing shortwave (W/m2)
Kup = alpha * Kdown
Lup = epsilon * sigma * T[0]**4

Qh = h * (T[0] - Ta)                        # sensible heat (positive when Ts > Ta)
QE = Qh / beta if beta != 0 else 0.0        # latent via Bowen ratio

QG = Kdown - Kup + Ldown - Lup - Qh - QE    # net into ground
```

and the surface derivative is set so that the conductive flux into the ground matches \(Q_G\). The discrete expression in `sebrhs_*` computes the surface time derivative using the derived flux \(f = -Q_G/k\) and the finite-difference stencil.

Notes on sign convention: the code uses \(Q_G =\) (net surface input) \(-\) (non-conductive losses); then uses \(f = -Q_G/k\) in the algebra that produces `dTdt[0]`. The GUI presents \(K^*, L^*, H, E, G\) with conventional plotting signs (radiation and conduction positive downward; sensible/latent follow \(T_s-T_a\)).

## 4) Derivation: how the SEB becomes the surface boundary condition

This section shows the algebra used to map the SEB to the Neumann-type boundary condition used in the finite-difference RHS (`sebrhs_ins` / `sebrhs_noins`). The end result is the exact discrete expression used in the code:

```python
f = -QG / k
dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)
```

Step 1 â€” start from the continuous SEB (positive downward):

$$ K_{\downarrow} - K_{\uparrow} + L_{\downarrow} - L_{\uparrow} - H - E - G = 0. $$

Re-arrange to isolate the conductive ground flux (positive downward):

$$
 G = K_{\downarrow} - K_{\uparrow} + L_{\downarrow} - L_{\uparrow} - H - E \equiv Q_G.
$$

So in the code `QG` holds the net flux into the ground.

Step 2 â€” relate \(G\) to the temperature gradient at the surface by Fourier's law:

$$
 G = -k\left.\frac{\partial T}{\partial z}\right|_{z=0},
$$

where \(k\) is the thermal conductivity (W m^{-1} K^{-1}). The minus sign arises because \(z\) increases upward (\(z=0\) at the surface and \(z<0\) below) while \(G\) is positive downward.

Step 3 â€” eliminate a ghost node to form a centred FD second-derivative at the surface.

Let the vertical grid spacing be \(\Delta x\) (code: `dz`) with indices 0 (surface), 1 (first subsurface), and a ghost node at index \(-1\) to enforce the boundary. The central FD for the second derivative at node 0 is:

$$
 \left.\frac{\partial^2 T}{\partial z^2}\right|_{0}
 \approx \frac{T_{-1} - 2T_0 + T_1}{\Delta x^2} .
$$

Use the centred first derivative to express the surface gradient:

$$
 \left.\frac{\partial T}{\partial z}\right|_0 \approx \frac{T_1 - T_{-1}}{2 \, \Delta x} .
$$

Plug this into Fourier's law and solve for \(T_{-1}\):

$$
 G = -k\frac{T_1 - T_{-1}}{2 \, \Delta x}
 \quad\Rightarrow\quad
 T_{-1} = T_1 + \frac{2 G}{k} \, \Delta x .
$$

Substitute into the second derivative:

$$
 \frac{T_{-1} - 2T_0 + T_1}{\Delta x^2}
 = \frac{\big(T_1 + \frac{2G}{k}\Delta x\big) - 2T_0 + T_1}{\Delta x^2}
 = \frac{2(T_1 - T_0)}{\Delta x^2} + \frac{2G}{k}\frac{1}{\Delta x} .
$$

Multiply by \(\kappa\) to get the surface ODE term:

$$
 \frac{d T_0}{d t} = \kappa\left[\frac{2(T_1 - T_0)}{\Delta x^2} + \frac{2G}{k}\frac{1}{\Delta x}\right]
 = \frac{2\kappa}{ \Delta x}\left(\frac{T_1 - T_0}{ \Delta x} - f\right),
$$

where \(f \equiv -\,G/k\) (and in code \(f = -Q_G/k\)). This is the expression used in `sebrhs_*`.

## 5) Flux parameterisations used in the code

- Shortwave partitioning (code):
  - \(K_{\downarrow}(t)\) provided by forcing (`kdown` / `diurnal_forcing`).
  - \(K_{\uparrow} = \alpha\,K_{\downarrow}\) (\(\alpha\) is material albedo).
  - Net shortwave: \(K^* = K_{\downarrow} - K_{\uparrow}\).

- Longwave (code):
  - \(L_{\uparrow} = \varepsilon\,\sigma\,T_s^4\) (Stefanâ€“Boltzmann law, emissivity \(\varepsilon\)).
  - \(L^* = L_{\downarrow} - L_{\uparrow}\).

- Sensible heat (H):
  - \(H = h\,(T_s - T_a)\), where \(h\) is a heat-transfer coefficient (W m^{-2} K^{-1}) and \(T_a\) is the air temperature.

- Latent heat (E):
  - Bowen-ratio approach: \(E = H / \beta\) (if evaporation allowed). When \(\beta = 0\) or material evaporation is disabled, \(E\equiv 0\).

- Ground conductive flux (G):
  - At the surface: enforced to balance the remainder of the SEB.
  - Post-processing estimate for plotting: \(G \approx -k\,(T_1 - T_0)/\Delta z\) (see `run_simulation`).

## 6) Bottom boundary condition

- Two options (controlled by `DEFAULTS['insulating_layer']`):
  - Insulating bottom (zero-flux): in `sebrhs_ins`, set the bottom node derivative to impose zero flux, i.e. \(\partial T/\partial z|_{z=-H}=0\).
  - Non-insulating (open) bottom: `sebrhs_noins` sets `dTdt[-1] = 0` (simplified demo behaviour).

## 7) Discretisation and solver

- Vertical grid: \(N_z\) points from surface (index 0) to bottom (index \(N_z-1\)). Spacing \(\Delta z = H/(N_z-1)\).
- Interior nodes: central second-difference for \(\partial^2 T/\partial z^2\), i.e.

  \( \displaystyle \frac{dT_i}{dt} = \kappa\, \frac{T_{i-1} - 2T_i + T_{i+1}}{\Delta z^2}\), for \(i=1..N_z-2\).

- Surface node: treated by combining the FD representation with the SEB to obtain `dTdt[0]`.
- Time integration: stiff ODE solver (BDF) via `solve_ivp` with adaptive stepping; `t_eval` is supplied to create regularly spaced outputs.

## 8) How this maps to the code

- `run_simulation(mat, params, dt, tmax)` â€” sets up grid, ICs and calls `solve_ivp`.
- `sebrhs_ins` / `sebrhs_noins` â€” RHS used by the integrator; implements interior update and boundary conditions.
- Forcing helpers:
  - `diurnal_forcing` â€” build Ta(t) and Kdown(t)
  - `kdown` â€” shortwave temporal shape used by the demo
- Outputs returned by `run_simulation` include:
  - `t`, `Ta`, `Ts` (surface), `Kstar`, `Kup`, `Kdown`, `Lstar`, `Lup`, `Ldown`, `G`, `H`, `E`, `z`, `T_profiles`, ...

## 9) References and further reading

- [Stefanâ€“Boltzmann law (radiative emission)](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)
- [Fourier's law / heat equation (conduction)](https://en.wikipedia.org/wiki/Fourier%27s_law_of_heat_conduction)
- [Heat transfer coefficient](https://en.wikipedia.org/wiki/Heat_transfer_coefficient)
- [Albedo](https://en.wikipedia.org/wiki/Albedo)

## 10) Quick notes / caveats

- Signs: Some texts use upward-positive conventions. Here, radiation and conduction are presented positive downward; sensible/latent follow \(T_s-T_a\).
- Latent heat is enabled only for materials with `evaporation=True` in `materials.json` and via the Bowen ratio parameter `beta` supplied to `run_simulation`.
- The surface longwave uses bulk \(T_s^4\) (gray-body assumption); atmospheric `Ldown` is provided by forcing and is not computed with a radiative transfer scheme.


