# About

**SEBdemo** is a compact 1D surface energy balance (SEB) and conduction demo
with both a programmatic API (`model.py`) and a simple GUI (`app.py`). It is
designed for experimenting with different material properties and diurnal
forcing to explore how surface temperature and energy fluxes respond.

Key features

- 1D conductive heat equation coupled to a surface energy balance (shortwave,
  longwave, sensible, latent and ground conductive flux).
- GUI tabs: Inputs (materials & parameters), Results, Term-by-term, Animation,
  and About.
- Compare two materials side-by-side and animate vertical temperature
  profiles with SEB arrows.
- Programmatic entry-point: `run_simulation(mat, params, dt, tmax)`.

For more information, see the project site [https://mvreeuwijk.github.io/SEBdemo/](https://mvreeuwijk.github.io/SEBdemo/)

Contact & license

- Maintainer: [maintainer@example.org](mailto:maintainer@example.org)
- License: This project is released under the GNU General Public License v3.0 (GPL-3.0) — see `docs/LICENSE.md` for the full text.
