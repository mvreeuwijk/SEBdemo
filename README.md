# SEB Roof Material Comparator (demo)

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.txt) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

This is a small demo app to compare roof materials using a simple surface-energy-balance + 1D conduction model.

Requirements

- Python 3.8+
- Install from requirements.txt (prefer inside a venv)

Run (desktop)

1. Install requirements (inside a venv recommended):
   python -m pip install -r requirements.txt
2. Start the desktop app:
   python app.py

# Urban Surface Energy Balance Demo

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.txt) [![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)

This is a small demo app to compare roof materials using a simple surface-energy-balance + 1D conduction model.

Requirements

- Python 3.8+
- Install from requirements.txt (prefer inside a venv)

Run (desktop)

1. Install requirements (inside a venv recommended):
   python -m pip install -r requirements.txt
2. Start the desktop app:
   python app.py

What's new (recent changes)

- The animation view now shows the current time in the plot title as it plays and after using the Reset button.
- A Reset button was added to the animation controls to rewind the animation to t=0.
- Material tooltips now display volumetric heat capacity C = rho*cp in MJ/m^3/K for clearer unit presentation.

Defaults

- The app defaults to a 48-hour simulation length. Use the input field to change it.

Notes

- This is a simplified educational model for demonstration in lectures.

License

- This project is licensed under the GNU General Public License v3.0 (GPL-3.0). See `docs/LICENSE.md` for details.
