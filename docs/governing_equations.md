# Governing equations and implementation notes

This document summarizes the mathematical model implemented in `model.py` (1D heat conduction with a surface energy balance). It gives the continuous equations, the boundary conditions used at the surface and bottom, the parameterisations of the surface flux terms, and how these map to the code.

## 1) Model overview

- Domain: 1D vertical soil column, z ∈ [−H, 0] with z=0 at the surface and z negative below the surface.
- Unknown: temperature T(z,t) (K).
- The model solves transient heat conduction in the soil coupled to a surface energy balance that provides the surface boundary condition.

## 2) Heat equation (continuous)

The soil obeys the diffusion (heat) equation in 1D:

$$C\frac{\partial T}{\partial t} = \frac{\partial}{\partial z}\left(k\frac{\partial T}{\partial z}\right) ,$$

where C = ρ c_p is the volumetric heat capacity (J m−3 K−1), k is thermal conductivity (W m−1 K−1). For homogeneous material with constant k and C this simplifies to

$$\frac{\partial T}{\partial t} = \kappa\frac{\partial^2 T}{\partial z^2},\quad \kappa = \frac{k}{C}. $$

Implementation note: the code forms a finite-difference discretisation on Nz points and advances the resulting ODE system in time using scipy.integrate.solve_ivp (method='BDF'). See `run_simulation` for discretisation choices.

## 3) Surface boundary condition (surface energy balance)

At the surface z = 0 the model enforces the surface energy balance (positive downward convention used in the code for flux bookkeeping):

$$K_{\downarrow} - K_{\uparrow} + L_{\downarrow} - L_{\uparrow} - H - E - G = 0,$$

where

- K↓ : incoming shortwave (W m−2)
- K↑ = α K↓ : reflected shortwave (albedo α)
- L↓ : downwelling longwave (W m−2)
- L↑ = ε σ T_s^4 : surface longwave upwelling (emissivity ε, Stefan–Boltzmann constant σ)
- H : sensible heat flux (W m−2)
- E : latent heat flux (W m−2)
- G : conductive flux into the ground (positive downward) (W m−2)

The code computes the net surface conductive gradient needed to satisfy this balance and converts that to a Neumann-type condition on the surface temperature node; the FD expression implemented in `sebrhs_ins` / `sebrhs_noins` leads to the surface ODE term:

In code the balance is evaluated as

```py
# shortwave partition
Kdown = forcing Kdown
Kup = alpha * Kdown
Lup = epsilon * sigma * T[0]**4

Qh = h * (T[0] - Ta)      # sensible (positive when Ts > Ta)
QE = Qh / beta if beta != 0 else 0.0  # latent via Bowen ratio

QG = Kdown - Kup + Ldown - Lup - Qh - QE
```

and the surface derivative is set so that the conductive flux into the ground matches QG. The discrete expression in `sebrhs_*` computes the surface time derivative using the derived flux f = −QG/k and the finite-difference stencil (see the code for the exact algebra).

Notes on sign convention: the code uses QG = (net surface input) − (non-conductive losses), then uses f = −QG/k in the algebra that produces dTdt[0]. The plotted diagnostics in the GUI present K*, L*, H, E, G with signs consistent with the usual energy-balance plotting conventions (positive downward for radiative and conductive terms; sensible/latent sign follows Ts−Ta convention).

## Derivation: how the SEB becomes the surface boundary condition (G = -k dT/dz)

This section shows the algebra used to map the surface energy balance (SEB) to the Neumann-type boundary condition used in the finite-difference RHS (`sebrhs_ins` / `sebrhs_noins`). The end result is the exact discrete expression used in the code:

```python
f = -QG / k
dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)
```

Step 1 — start from the continuous SEB (positive downward convention):

$$K_{\downarrow} - K_{\uparrow} + L_{\downarrow} - L_{\uparrow} - H - E - G = 0.$$

Re-arrange to isolate the conductive ground flux G (positive downward):

$$G = K_{\downarrow} - K_{\uparrow} + L_{\downarrow} - L_{\uparrow} - H - E \equiv Q_G.$$

So in the code `QG` holds the net flux into the ground.

Step 2 — relate G to the temperature gradient at the surface by Fourier's law:

$$G = -k\left.\frac{\partial T}{\partial z}\right|_{z=0},$$

where k is the thermal conductivity (W m^{-1} K^{-1}). The minus sign arises because we use z increasing upward (z=0 at surface and z negative downwards) and we choose G positive downward (consistent with the code's bookkeeping).

Step 3 — eliminate a ghost-node to form a centred FD second-derivative at the surface.

Let the vertical grid spacing be `dx` (code: `dz`) and indices 0 (surface), 1 (first subsurface), and a ghost node at index −1 used to enforce the Neumann condition. The central FD for the second derivative at node 0 is:

$$\left.\frac{\partial^2 T}{\partial z^2}\right|_{0} \approx \frac{T_{-1} - 2T_0 + T_1}{\Delta x^2} .$$

Use the first-derivative centred approximation to express the surface gradient in terms of the ghost node:

$$\left.\frac{\partial T}{\partial z}\right|_0 \approx \frac{T_1 - T_{-1}}{2 \Delta x} .$$

Plug this into Fourier's law and solve for the ghost node value T_{-1}:

$$G = -k\frac{T_1 - T_{-1}}{2 \Delta x} \quad\Rightarrow\quad T_{-1} = T_1 + \frac{2 G}{k} \Delta x .$$

Step 4 — substitute the ghost-node expression into the second-difference:

$$\frac{T_{-1} - 2T_0 + T_1}{\Delta x^2} = \frac{\big(T_1 + \frac{2G}{k}\Delta x\big) - 2T_0 + T_1}{\Delta x^2}$$
$$= \frac{2(T_1 - T_0)}{\Delta x^2} + \frac{2G}{k}\frac{1}{\Delta x}.$$

Step 5 — multiply by thermal diffusivity κ = k / C to obtain the time derivative at the surface node (continuum → discrete RHS):

$$\frac{d T_0}{d t} = \kappa\left[\frac{2(T_1 - T_0)}{\Delta x^2} + \frac{2G}{k}\frac{1}{\Delta x}\right].$$

Factor the expression and substitute the code variable definitions. The code computes

```py
f = -QG / k
```

which, since `QG` == G, gives `f = -G/k`. Re-writing the discrete surface RHS in the same form used in the code:

$$\frac{d T_0}{d t} = \frac{2\kappa}{ \Delta x}\left(\frac{T_1 - T_0}{ \Delta x} - f\right).$$

This matches the implementation in `sebrhs_ins` / `sebrhs_noins`:

```py
# inside sebrhs_*
f = -QG / k
dTdt[0] = 2 * kappa / dx * ((T[1] - T[0]) / dx - f)
```

Step 6 — sign check and interpretation

- If `QG` (code) is positive, the net SEB input goes into the ground (G positive downward). From Fourier's law this corresponds to a negative near-surface temperature gradient (surface warmer than subsurface), and the ghost-node algebra ensures the surface node derivative responds accordingly.
- If evaporation is disabled or `beta` is zero the code sets `QE=0` and `QG` reduces accordingly.

This derivation therefore shows precisely how the continuous SEB is enforced as a Neumann boundary condition in the finite-difference ODE system solved by `solve_ivp` and reproduces the algebra used in the code.

## 4) Flux parameterisations used in the code

- Shortwave partitioning (code):
  - Kdown(t) provided by forcing (function `kdown` / `diurnal_forcing`).
  - Kup = α Kdown (α is material albedo).
  - Net shortwave (K*) = Kdown − Kup.

- Longwave (code):
  - Lup = ε σ T_s^4 (Stefan–Boltzmann law, emissivity ε).
  - Lstar = Ldown − Lup.

- Sensible heat (H):
  - H = h (T_s − T_a), where h is an external heat-transfer coefficient (W m−2 K−1) and T_a is the air temperature (forcing).

- Latent heat (E):
  - The demo uses the Bowen-ratio approach: E = H / β (if evaporation allowed). When β == 0 or material evaporation is disabled, E is zero.

- Ground conductive flux (G):
  - At the surface the model enforces that G balances the remainder of the SEB.
  - The post-processing estimate for plotting is G ≈ −k (T[1] − T[0]) / dz (see `run_simulation` vectorised `dTdz` and `Gflux`).

## 5) Bottom boundary condition

- Two options are provided in code (controlled by `DEFAULTS['insulating_layer']`):
  - insulating bottom (zero-flux): implemented in `sebrhs_ins` by setting the bottom node derivative to impose zero flux (Neumann 0), i.e. dTdt[-1] = −2 κ / dx * ((T[-1] − T[-2]) / dx).
  - non-insulating (open) bottom: `sebrhs_noins` sets dTdt[-1] = 0 (effectively a fixed state for the bottom node in this simplified demo).

## 6) Discretisation and solver

- Vertical grid: Nz points from surface (index 0) to bottom (index Nz−1). Spacing dz = thickness / (Nz−1).
- Interior nodes: central second-difference for ∂^2T/∂z^2, giving

  dTdt[i] = κ (T[i−1] − 2 T[i] + T[i+1]) / dz^2 for i = 1..Nz−2

- Surface node: treated by combining the finite-difference representation and the surface energy balance to obtain dTdt[0] (see `sebrhs_*`).
- Time integration: stiff ODE solver BDF via scipy.integrate.solve_ivp with adaptive stepping; t_eval is supplied to create regularly spaced outputs used by the GUI.

## 7) How this maps to the code

- `run_simulation(mat, params, dt, tmax)` — main entry that sets up grid, initial conditions and calls solve_ivp.
- `sebrhs_ins` / `sebrhs_noins` — right-hand side implementations used by the integrator; they implement the FD interior update and the surface/bottom BC algebra.
- Forcing functions:
  - `diurnal_forcing` — convenience to build Ta(t) and Kdown(t)
  - `kdown` — simple shortwave temporal shape used by the demo.
- Post-processing and outputs returned in the result dict from `run_simulation`:
  - t, Ta, Ts (surface), Kstar, Kup, Kdown, Lstar, Lup, Ldown, G, H, E, z, T_profiles, ...

## 8) References and further reading

- [Stefan–Boltzmann law (radiative emission)](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law)
- [Fourier's law / heat equation (conduction)](https://en.wikipedia.org/wiki/Fourier%27s_law_of_heat_conduction)
- [Heat transfer coefficient and simple convective parameterisations](https://en.wikipedia.org/wiki/Heat_transfer_coefficient)
- [Shortwave partition and albedo (practical notes)](https://en.wikipedia.org/wiki/Albedo)

These are general references; the code implements simplified parameterisations suitable for demonstration and teaching (Bowen-ratio latent heat, fixed heat-transfer coefficient for H, simple albedo partition). For research-grade SEB models you would replace the simple parameterisations with more comprehensive treatment of aerodynamic transfer, surface resistance networks, humidity coupling, and spectral albedo/emissivity.

## 9) Quick notes / caveats

- Signs: be careful when comparing sign conventions between different models or texts. The GUI and code present fluxes with a conventional plotting sign (radiation and conduction positive downward), while some textbooks use upward-positive conventions.
- Latent heat is enabled only for materials with `evaporation=True` in `materials.json` and via the Bowen ratio parameter `beta` supplied to `run_simulation`.
- The surface longwave uses bulk T_s**4 via ε σ T^4 (gray/grey-body assumption); atmospheric Ldown is taken from forcing and is not computed from a radiative transfer scheme.
