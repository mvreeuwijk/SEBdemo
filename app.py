"""SEB demo GUI (tkinter) with compare + animation support.

This file implements three requested features:
 - Compare default disabled; Material B controls are greyed when compare is off.
 - Animation tab that shows a single panel when compare is off, and two panels when on.
 - A small scaffold to detect `customtkinter` and optionally switch to a
     CustomTkinter-based UI (we keep tkinter as the primary implementation
     here so the app runs in the provided venv without extra installs).

The implementation is deliberately incremental and testable. After this
change `import app` should succeed and the UI will support compare/animation.
"""
from __future__ import annotations

import importlib
import json
import threading
from pathlib import Path
from tkinter import messagebox
import tkinter as tk
import tkinter.font as tkfont
from typing import Optional, Any, Mapping, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk
import logging
import traceback
import sys
import re
import webbrowser

from model import load_material, run_simulation, diurnal_forcing, DEFAULTS as MODEL_DEFAULTS, hour


def load_material_keys(path: str = "materials.json"):
    base = Path(__file__).parent
    with open(base / path, "r") as f:
        data = json.load(f)
    return list(data.keys())


def customtk_available() -> bool:
    try:
        importlib.import_module('customtkinter')
        return True
    except Exception:
        return False

class App(ctk.CTk):

    # GUI-only defaults (keep UI-only flags separate from model DEFAULTS)
    GUIDEFAULTS = {
        'compare': False,
        'speed': 2.0,
        'forcing_dt': 60.0,
    }

    # annotate some optional attributes to quiet static checkers (Pylance)
    _tooltip_win: Optional[tk.Toplevel] = None
    _tooltip_after_id: Optional[str] = None
    _lbl_thickB_color_default: Optional[str] = None
    def __init__(self) -> None:
        ctk.set_appearance_mode('System')
        ctk.set_default_color_theme('blue')
        super().__init__()
        self.title('SEB demo')
        # default size and place the window near the top of the screen so the
        # whole window is visible on most displays. Prefer centering horizontally
        # and a small top margin.
        width, height = 1000, 560
        try:
            # ensure geometry info is up to date
            self.update_idletasks()
            sw = self.winfo_screenwidth()
            x = max(0, (sw - width) // 2)
            y = 30
            self.geometry(f"{width}x{height}+{x}+{y}")
        except Exception:
            # fallback to a reasonable position
            self.geometry(f"{width}x{height}+50+30")

        # Tabview (CTkTabview)
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill='both', expand=True, padx=12, pady=12)
        self.tabview.add('Inputs')
        self.tabview.add('Animation')
        self.tabview.add('Results')
        self.tabview.add('Term by term')
        self.tabview.add('About')

        # references to tab frames
        self.tab_inputs = self.tabview.tab('Inputs')
        self.tab_animation = self.tabview.tab('Animation')
        self.tab_results = self.tabview.tab('Results')
        self.tab_tempseb = self.tabview.tab('Term by term')
        self.tab_about = self.tabview.tab('About')

        # build tab content
        self._build_inputs_tab()
        self._build_results_tab()
        self._build_tempseb_tab()
        self._build_animation_tab()
        self._build_about_tab()

        # track tab changes: run simulation automatically when leaving Inputs tab
        try:
            self._last_tab = self.tabview.get()
        except Exception:
            self._last_tab = 'Inputs'
        # keep the poll after-id so we can cancel it on exit
        try:
            self._poll_after_id = self.after(500, self._poll_tab)
        except Exception:
            self._poll_after_id = None

        # ensure we clean up scheduled callbacks when the window is closed
        self.protocol('WM_DELETE_WINDOW', self._on_closing)

    # --- builders for each tab to keep __init__ concise ---
    def _build_inputs_tab(self):
        """Create widgets for the Inputs tab (materials, parameters, previews)."""
        keys = load_material_keys()
        frm = ctk.CTkFrame(self.tab_inputs)
        frm.pack(fill='both', expand=True, padx=8, pady=8)
        # tune grid column weights so the optionmenus and entries fit the window
        frm.grid_columnconfigure(0, weight=1, minsize=120, uniform='param')
        frm.grid_columnconfigure(1, weight=1, minsize=120, uniform='param')
        frm.grid_columnconfigure(2, weight=1, minsize=120, uniform='param')

        # capture the inputs frame background color (used to color Matplotlib figs)
        self._input_frame_bg = self._mpl_color_from_ctk(frm.cget('bg_color'))


        # Top row: Material selectors and compare checkbox above Material B
        frameA = ctk.CTkFrame(frm, fg_color='transparent')
        frameA.grid(row=1, column=0, sticky='nsew', padx=6, pady=(6, 2))
        frameA.grid_columnconfigure(0, weight=0)
        frameA.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frameA, text='Material A:').grid(row=0, column=0, sticky='w')
        self.matA = ctk.StringVar(value=keys[0] if keys else 'concrete')
        self.opt_matA = ctk.CTkOptionMenu(frameA, values=keys, variable=self.matA, width=150)
        self.opt_matA.grid(row=0, column=1, sticky='e', padx=(6, 0))
        # tooltip support for material A: show properties on hover
        self._tooltip_after_id = None
        self._tooltip_win = None
        self.opt_matA.bind('<Enter>', lambda e, w=self.opt_matA: self._schedule_show_material_tooltip(w, self.matA))
        self.opt_matA.bind('<Leave>', lambda e: self._hide_material_tooltip())
        # Some CTkOptionMenu implementations contain child widgets that receive
        # enter/leave events instead of the container. Attach the same handlers
        # to known children shortly after creation so tooltips reliably hide.
        def _attach_children_for_optmenu(w, v):
            try:
                for ch in w.winfo_children():
                    try:
                        ch.bind('<Enter>', lambda e, ww=w, vv=v: self._schedule_show_material_tooltip(ww, vv))
                        ch.bind('<Leave>', lambda e: self._hide_material_tooltip())
                    except Exception:
                        pass
            except Exception:
                pass
        try:
            self.after(50, lambda w=self.opt_matA, v=self.matA: _attach_children_for_optmenu(w, v))
        except Exception:
            pass
        ctk.CTkLabel(frameA, text='Thickness A (m):').grid(row=1, column=0, sticky='w', pady=(6, 0))
        self.thickA = ctk.DoubleVar(value=MODEL_DEFAULTS['thickness'])
        self.entry_thickA = ctk.CTkEntry(frameA, textvariable=self.thickA, width=120, justify='right')
        self.entry_thickA.grid(row=1, column=1, sticky='e', padx=(6, 0), pady=(6, 0))

        # Compare control in third column above Material B
        # GUI compare default: prefer GUIDEFAULTS (UI-only); fall back to model DEFAULTS
        cmp_def = self.GUIDEFAULTS['compare']
        self.compare_var = ctk.BooleanVar(value=cmp_def)
        cmp_frame = ctk.CTkFrame(frm, fg_color='transparent')
        # place the compare control inside the third column (column=2).
        # The label will be left-aligned and the checkbox right-aligned within this column.
        cmp_frame.grid(row=0, column=2, sticky='nsew', padx=(6, 0), pady=(6, 0))
        # give the right-hand column weight so the checkbox can expand
        # and be flush with the right edge of the window
        cmp_frame.grid_columnconfigure(0, weight=1)
        cmp_frame.grid_columnconfigure(1, weight=1)
        # show checkbox on the left and description on the right
        # command toggles Material B controls immediately
        self.cb_compare = ctk.CTkCheckBox(cmp_frame, text='', variable=self.compare_var, command=self._on_compare_toggle)
        self.cb_compare.grid(row=0, column=0, sticky='w', padx=(0, 6))
        ctk.CTkLabel(cmp_frame, text='Compare with second material:').grid(row=0, column=1, sticky='w')

        # Frame for Material B (aligned with A)
        frameB = ctk.CTkFrame(frm, fg_color='transparent')
        frameB.grid(row=1, column=2, sticky='nsew', padx=6, pady=(6, 2))
        frameB.grid_columnconfigure(0, weight=0)
        frameB.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(frameB, text='Material B:').grid(row=0, column=0, sticky='w')
        self.matB = ctk.StringVar(value=(keys[1] if len(keys) > 1 else keys[0]))
        self.opt_matB = ctk.CTkOptionMenu(frameB, values=keys, variable=self.matB, width=150)
        self.opt_matB.grid(row=0, column=1, sticky='e', padx=(6, 0))
        # tooltip support for material B
        self.opt_matB.bind('<Enter>', lambda e, w=self.opt_matB: self._schedule_show_material_tooltip(w, self.matB))
        self.opt_matB.bind('<Leave>', lambda e: self._hide_material_tooltip())
        try:
            self.after(50, lambda w=self.opt_matB, v=self.matB: _attach_children_for_optmenu(w, v))
        except Exception:
            pass
        self.lbl_thickB = ctk.CTkLabel(frameB, text='Thickness B (m):')
        self.lbl_thickB.grid(row=1, column=0, sticky='w', pady=(6, 0))
        # remember default color for toggling
        try:
            self._lbl_thickB_color_default = self.lbl_thickB.cget('text_color')
        except Exception:
            self._lbl_thickB_color_default = None
        self.thickB = ctk.DoubleVar(value=MODEL_DEFAULTS['thickness'])
        self.entry_thickB = ctk.CTkEntry(frameB, textvariable=self.thickB, width=120, justify='right')
        self.entry_thickB.grid(row=1, column=1, sticky='e', padx=(6, 0), pady=(6, 0))
        # remember entry default text color so we can restore it reliably
        try:
            self._entry_thickB_text_color_default = self.entry_thickB.cget('text_color')
        except Exception:
            self._entry_thickB_text_color_default = None

        sep = ctk.CTkFrame(frm, height=2, fg_color='#7f7f7f')
        sep.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(6, 8))

        # runtime flags
        self._running = False
        self._closed = False
        # storage for validated parameters (filled by validators)
        self._params = {
            'thickness_A': float(MODEL_DEFAULTS['thickness']),
            'thickness_B': float(MODEL_DEFAULTS['thickness']),
            'Sb': float(MODEL_DEFAULTS['Sb']),
            # model DEFAULTS stores trise/tset in seconds
            'trise': int(MODEL_DEFAULTS['trise']),
            'tset': int(MODEL_DEFAULTS['tset']),
            'Ldown': float(MODEL_DEFAULTS['Ldown']),
            # model uses key 'h' for heat-transfer coefficient
            'h': float(MODEL_DEFAULTS['h']),
            # model Ta_mean is in Kelvin; keep internal param in K
            'Ta_mean': float(MODEL_DEFAULTS['Ta_mean']),
            'Ta_amp': float(MODEL_DEFAULTS['Ta_amp']),
            'beta': float(MODEL_DEFAULTS['beta']),
        }

        # Parameters arranged in three vertical columns under the materials
        self.param_col0 = ctk.CTkFrame(frm)
        self.param_col0.grid(row=3, column=0, sticky='nsew', padx=6, pady=4)
        self.param_col1 = ctk.CTkFrame(frm)
        self.param_col1.grid(row=3, column=1, sticky='nsew', padx=6, pady=4)
        self.param_col2 = ctk.CTkFrame(frm)
        self.param_col2.grid(row=3, column=2, sticky='nsew', padx=6, pady=4)
        self.param_col0.grid_columnconfigure(0, weight=1)
        self.param_col0.grid_columnconfigure(1, weight=1)
        self.param_col1.grid_columnconfigure(0, weight=1)
        self.param_col1.grid_columnconfigure(1, weight=1)
        self.param_col2.grid_columnconfigure(0, weight=1)
        self.param_col2.grid_columnconfigure(1, weight=1)

        frm.grid_rowconfigure(6, weight=1)
        frm.grid_rowconfigure(7, weight=1)

        # Column 0
        ctk.CTkLabel(self.param_col0, text='Peak shortwave (W/m2):').grid(row=0, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.Sb = ctk.DoubleVar(value=MODEL_DEFAULTS['Sb'])
        self.entry_Sb = ctk.CTkEntry(self.param_col0, textvariable=self.Sb, width=80, justify='right')
        self.entry_Sb.grid(row=0, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col0, text='Sunrise time (h):').grid(row=1, column=0, sticky='w', padx=(0,6), pady=(0,4))
        # model stores trise in seconds; present GUI field in hours
        trise_hours = float(MODEL_DEFAULTS['trise']) / float(hour)
        self.trise_hr = ctk.DoubleVar(value=trise_hours)
        self.entry_trise = ctk.CTkEntry(self.param_col0, textvariable=self.trise_hr, width=80, justify='right')
        self.entry_trise.grid(row=1, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col0, text='Sunset time (h):').grid(row=2, column=0, sticky='w', padx=(0,6), pady=(0,4))
        tset_hours = float(MODEL_DEFAULTS['tset']) / float(hour)
        self.tset_hr = ctk.DoubleVar(value=tset_hours)
        self.entry_tset = ctk.CTkEntry(self.param_col0, textvariable=self.tset_hr, width=80, justify='right')
        self.entry_tset.grid(row=2, column=1, sticky='e', pady=(0,4))

        # Column 1
        ctk.CTkLabel(self.param_col1, text='Incoming longwave (W/m2):').grid(row=0, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.Ldown = ctk.DoubleVar(value=MODEL_DEFAULTS['Ldown'])
        self.entry_Ldown = ctk.CTkEntry(self.param_col1, textvariable=self.Ldown, width=80, justify='right')
        self.entry_Ldown.grid(row=0, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col1, text='Heat transfer coeff (W/m2K):').grid(row=1, column=0, sticky='w', padx=(0,6), pady=(0,4))
        # model uses key 'h' for the heat transfer coefficient
        hval = float(MODEL_DEFAULTS['h'])
        self.hcoef = ctk.DoubleVar(value=hval)
        self.entry_hcoef = ctk.CTkEntry(self.param_col1, textvariable=self.hcoef, width=80, justify='right')
        self.entry_hcoef.grid(row=1, column=1, sticky='e', pady=(0,4))

        # Bowen ratio moved here (swapped with Air mean temp)
        ctk.CTkLabel(self.param_col1, text='Bowen ratio (-):').grid(row=2, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.beta_var = ctk.DoubleVar(value=MODEL_DEFAULTS['beta'])
        self.entry_beta = ctk.CTkEntry(self.param_col1, textvariable=self.beta_var, width=80, justify='right')
        self.entry_beta.grid(row=2, column=1, sticky='e', pady=(0,4))

        # Column 2
        ctk.CTkLabel(self.param_col2, text='Air temperature amp (°C):').grid(row=0, column=0, sticky='w', padx=(0,6), pady=(0,4))
        # model Ta_amp is in K (same scale as °C for amplitudes)
        ta_amp = float(MODEL_DEFAULTS['Ta_amp'])
        self.Ta_amp_C = ctk.DoubleVar(value=ta_amp)
        self.entry_Ta_amp = ctk.CTkEntry(self.param_col2, textvariable=self.Ta_amp_C, width=80, justify='right')
        self.entry_Ta_amp.grid(row=0, column=1, sticky='e', pady=(0,4))

        # Air mean temperature moved here (swapped with Bowen ratio)
        ctk.CTkLabel(self.param_col2, text='Air mean temp (°C):').grid(row=1, column=0, sticky='w', padx=(0,6), pady=(0,4))
        ta_mean_k = float(MODEL_DEFAULTS['Ta_mean'])
        # present mean air temperature in °C in the GUI
        self.Ta_mean_C = ctk.DoubleVar(value=ta_mean_k - 273.15)
        self.entry_Ta_mean = ctk.CTkEntry(self.param_col2, textvariable=self.Ta_mean_C, width=80, justify='right')
        self.entry_Ta_mean.grid(row=1, column=1, sticky='e', pady=(0,4))

        # Preview figures
        self.fig_Ta = Figure(figsize=(8, 2))
        self.fig_Ta.patch.set_facecolor(self._input_frame_bg)
        self.canvas_Ta = FigureCanvasTkAgg(self.fig_Ta, master=frm)
        self.canvas_Ta.get_tk_widget().grid(row=6, column=0, columnspan=3, padx=8, pady=(8, 4), sticky='nsew')
        self.ax_Ta = self.fig_Ta.subplots(1, 1)

        self.fig_K = Figure(figsize=(8, 2))
        self.fig_K.patch.set_facecolor(self._input_frame_bg)
        self.canvas_K = FigureCanvasTkAgg(self.fig_K, master=frm)
        self.canvas_K.get_tk_widget().grid(row=7, column=0, columnspan=3, padx=8, pady=(4, 8), sticky='nsew')
        self.ax_K = self.fig_K.subplots(1, 1)

        # initial Inputs-tab state: enable/disable compare controls and
        # install preview traces so changing inputs updates the preview.
        # ensure preview scheduling handle exists
        self._preview_after_id = None

        # initialize compare state so Material B controls are set correctly
        try:
            val = self.compare_var.get()
            if isinstance(val, str):
                enabled = val.lower() in ("1", "true", "t", "yes", "y")
            else:
                enabled = bool(val)
        except Exception:
            enabled = False

        state = 'normal' if enabled else 'disabled'

        # configure Material B optionmenu if present
        if getattr(self, 'opt_matB', None) is not None:
            try:
                self.opt_matB.configure(state=state)
            except Exception:
                try:
                    import logging
                    logging.getLogger(__name__).exception('Failed to configure opt_matB')
                except Exception:
                    pass

        # configure Material B thickness entry (and visual hint)
        if getattr(self, 'entry_thickB', None) is not None:
            try:
                self.entry_thickB.configure(state=state)
            except Exception:
                try:
                    import logging
                    logging.getLogger(__name__).exception('Failed to configure entry_thickB state')
                except Exception:
                    pass
            # visual hint: gray text when disabled (best-effort)
            try:
                if not enabled:
                    self.entry_thickB.configure(text_color='gray')
                else:
                    # some CTkEntry implementations may not accept None; ignore failures
                    self.entry_thickB.configure(text_color=None)
            except Exception:
                pass

        # update animation layout (ensure static background matches new settings)
        try:
            # configure_animation_axes was removed; redraw static animation background
            self.draw_anim_static()
        except Exception:
            try:
                import logging
                logging.getLogger(__name__).exception('draw_anim_static failed')
            except Exception:
                pass

        # draw an initial preview (best-effort)
        self.update_preview()
            # install traces on the input variables to debounce preview updates
        self._install_preview_traces()

        # bind focus-out validation for inputs so correctness is checked when the
        # user leaves the box (and before we run the simulation).
        self.entry_Sb.bind('<FocusOut>', lambda e: self._validate_and_store('Sb'))
        self.entry_trise.bind('<FocusOut>', lambda e: self._validate_and_store('trise'))
        self.entry_tset.bind('<FocusOut>', lambda e: self._validate_and_store('tset'))
        self.entry_Ldown.bind('<FocusOut>', lambda e: self._validate_and_store('Ldown'))
        self.entry_hcoef.bind('<FocusOut>', lambda e: self._validate_and_store('h'))
        self.entry_Ta_mean.bind('<FocusOut>', lambda e: self._validate_and_store('Ta_mean'))
        self.entry_Ta_amp.bind('<FocusOut>', lambda e: self._validate_and_store('Ta_amp'))
        self.entry_beta.bind('<FocusOut>', lambda e: self._validate_and_store('beta'))
        # thickness entries already exist; bind them too
        self.entry_thickA.bind('<FocusOut>', lambda e: self._validate_and_store('thickness_A'))
        self.entry_thickB.bind('<FocusOut>', lambda e: self._validate_and_store('thickness_B'))

        # run an initial validation to populate self._params
        self._validate_and_store()

    def _build_results_tab(self):
        """Create the Results tab figure and axes."""
        # Results: create a 2x2 grid. When only one material is present we use
        # the left column; when comparing we fill both columns.
        self.fig_res = Figure(figsize=(8, 9))
        self.fig_res.patch.set_facecolor(self._input_frame_bg)
        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=self.tab_results)
        self.canvas_res.get_tk_widget().pack(fill='both', expand=True)
        # 2 rows x 2 cols, keep as 2D array for easy indexing
        self.axs_res = self.fig_res.subplots(2, 2, squeeze=False)

    def _build_tempseb_tab(self):
        """Create the Temp & SEB (term-by-term) tab contents."""
        # small Temp profile figure exists for programmatic use but is not
        # shown in the UI (Term-by-term grid is the visible summary)
        self.fig_temp = Figure(figsize=(6, 3))
        self.fig_temp.patch.set_facecolor(self._input_frame_bg)
        self.canvas_temp = FigureCanvasTkAgg(self.fig_temp, master=self.tab_tempseb)
        self.ax_temp = self.fig_temp.subplots(1, 1)


        # time slider for Temp & SEB (inactive until a run populates anim_data)
        self.temp_time_var = ctk.IntVar(value=0)
        # slider and playback controls are intentionally not packed into the
        # UI (we don't show animation controls in the Term-by-term tab)
        self.temp_slider = ctk.CTkSlider(self.tab_tempseb, from_=0, to=1, number_of_steps=1, command=lambda v: self.update_tempseb(int(float(v))))
        # playback controls for Temp & SEB
        self.temp_playing = False
        self.temp_play_btn = ctk.CTkButton(self.tab_tempseb, text='Play', command=self.toggle_temp_playback, state='disabled')

        # Also provide a copy of the Results plots in the Term by Term tab so
        # users can see the summary plots alongside the temporal profile view.
        # Use a 3x3 grid so each panel can show both Material A and B together
        # (with a legend) as requested.
        self.fig_temp_res = Figure(figsize=(9, 9))
        self.fig_temp_res.patch.set_facecolor(self._input_frame_bg)
        self.canvas_temp_res = FigureCanvasTkAgg(self.fig_temp_res, master=self.tab_tempseb)
        self.canvas_temp_res.get_tk_widget().pack(fill='both', expand=True)
        # term-by-term comparison: 3 rows x 2 cols. We'll plot the main
        # terms (Ts, K*, L*, H, E, G) in the first two rows and use the third
        # row for derived comparisons (residual, Ts diff, E diff).
        self.axs_temp_res = self.fig_temp_res.subplots(3, 2, squeeze=False)

    def _build_animation_tab(self):
        """Create the Animation tab figure and controls.

        Controls are placed in a top bar; the figure is a single axes
        (always one panel). Store the axes in `self.ax_anim` and keep the
        FigureCanvas below the control bar.
        """
        # create figure and single axes for animation
        self.fig_anim = Figure(figsize=(8, 3))
        self.fig_anim.patch.set_facecolor(self._input_frame_bg)
        self.ax_anim = self.fig_anim.subplots(1, 1)

        # controls across the top
        self.anim_ctrl_frame = ctk.CTkFrame(self.tab_animation)
        self.anim_ctrl_frame.pack(side='top', fill='x', padx=8, pady=6)
        self.time_label = ctk.CTkLabel(self.anim_ctrl_frame, text='Time: -- h')
        self.time_label.grid(row=0, column=0, padx=(4, 8))
        # animation speed: prefer GUI-only default
        try:
            sv = float(self.GUIDEFAULTS.get('speed', 1.0))
        except Exception:
            sv = 1.0
        self.speed_var = ctk.DoubleVar(value=sv)
        ctk.CTkLabel(self.anim_ctrl_frame, text='Speed:').grid(row=0, column=1, sticky='w')
        ctk.CTkEntry(self.anim_ctrl_frame, textvariable=self.speed_var, width=80).grid(row=0, column=2, padx=(4, 8))
        self.btn_start = ctk.CTkButton(self.anim_ctrl_frame, text='Start', command=self.toggle_animation, state='disabled')
        self.btn_start.grid(row=0, column=3, padx=(8, 4))

        # place the figure below the controls
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=self.tab_animation)
        self.canvas_anim.get_tk_widget().pack(side='top', fill='both', expand=True)

        # animation state placeholders
        self.animating = False
        self.anim_idx = 0
        self.anim_data = None

    def _build_about_tab(self):
        """Create the About tab and display contents of about.md.

        This implements a small, dependency-free markdown renderer that
        handles headings (#), unordered lists (-), and inline **bold** and
        *italic* markup. The Text widget background is set to match the
        app input frame color (converted via _mpl_color_from_ctk) for a
        consistent appearance with the rest of the UI.
        """
        frm = ctk.CTkFrame(self.tab_about)
        try:
            if getattr(self, '_input_frame_bg', None) is not None:
                frm.configure(fg_color=self._input_frame_bg)
        except Exception:
            pass
        frm.pack(fill='both', expand=True, padx=8, pady=8)

    
        # Load about.md from repository root
        try:
            p = Path(__file__).parent / 'about.md'
            content = p.read_text(encoding='utf-8')
        except Exception:
            content = 'About information not available.'

        # Compute tkinter background color from CTk color (if available)
        
        tk_bg = frm.cget('bg_color')

        # Prepare fonts
        try:
            base_font = tkfont.nametofont('TkDefaultFont')
        except Exception:
            base_font = tkfont.Font()
        body_size = max(10, int(base_font.cget('size')) + 2)
        h1_size = max(12, int(base_font.cget('size')) + 4)

        # If CTkTextbox exists we could use it, but to keep styling consistent
        # we'll render the markdown into a stack of CTkLabel widgets. This
        # avoids dealing with Text tags and guarantees CTk-based appearance.
        # determine readable text color against background (fallback to CTk default)
        text_color = None
        try:
            if tk_bg and isinstance(tk_bg, str) and tk_bg.startswith('#') and len(tk_bg) >= 7:
                rr = int(tk_bg[1:3], 16) / 255.0
                gg = int(tk_bg[3:5], 16) / 255.0
                bb = int(tk_bg[5:7], 16) / 255.0
                lum = 0.299 * rr + 0.587 * gg + 0.114 * bb
                text_color = 'white' if lum < 0.5 else 'black'
        except Exception:
            text_color = None

        # create font objects for labels and links
        body_font = tkfont.Font(font=base_font)
        try:
            body_font.configure(size=body_size)
        except Exception:
            pass
        h1_font = tkfont.Font(font=base_font)
        try:
            h1_font.configure(size=h1_size, weight='bold')
        except Exception:
            pass
        # link font will be a simple tuple for CTkLabel; underline may not be
        # widely supported so we use blue text + hand cursor for affordance.

        # Render about.md into a single, read-only Tk Text widget so we get
        # reliable wrapping and layout across platforms. We keep inline bold
        # and clickable links by using text tags.
        link_re = re.compile(
            r'\[([^\]]+)\]\(([^)]+)\)|'      # [text](url)
            r'<(https?://[^>]+)>|'                # <https://...>
            r'<([\w\.-]+@[\w\.-]+\.[\w]+)>|' # <user@example.org>
            r'(https?://\S+)|'                   # bare https://...
            r'([\w\.-]+@[\w\.-]+\.[\w]+)'
        )

        # create a read-only Text widget
        txt = tk.Text(frm, wrap='word', bd=0, relief='flat')
        # background color should match the CTk frame if available
        try:
            if tk_bg:
                txt.configure(bg=tk_bg)
        except Exception:
            pass

        # prepare fonts and tags
        bold_font = tkfont.Font(font=body_font)
        try:
            bold_font.configure(weight='bold')
        except Exception:
            pass

        txt.tag_configure('h1', font=h1_font)
        txt.tag_configure('bold', font=bold_font)
        txt.tag_configure('para', font=body_font)
        txt.tag_configure('bullet', font=body_font)

        link_count = 0

        def insert_with_bold(target, s, tag='para'):
            # simple **bold** scanner
            i = 0
            while i < len(s):
                if s.startswith('**', i):
                    j = s.find('**', i + 2)
                    if j != -1:
                        inner = s[i+2:j]
                        if inner:
                            target.insert('end', inner, ('bold', tag))
                        i = j + 2
                        continue
                # no bold start -> insert until next ** or end
                next_pos = s.find('**', i)
                if next_pos == -1:
                    target.insert('end', s[i:], (tag,))
                    break
                else:
                    target.insert('end', s[i:next_pos], (tag,))
                    i = next_pos

        for ln in content.splitlines():
            ln_stripped = ln.strip()
            if ln_stripped.startswith('# '):
                txt.insert('end', ln_stripped[2:].strip() + '\n\n', ('h1',))
                continue
            if ln_stripped == '':
                txt.insert('end', '\n')
                continue

            # lists
            if ln_stripped.startswith('- '):
                item = ln_stripped[2:].strip()
                txt.insert('end', '• ', ('bullet',))
                # process inline links and bold inside item
                pos = 0
                for m in link_re.finditer(item):
                    if m.start() > pos:
                        insert_with_bold(txt, item[pos:m.start()])
                    if m.group(1):
                        label_text = m.group(1); url = m.group(2)
                    elif m.group(3):
                        label_text = m.group(3); url = m.group(3)
                    elif m.group(4):
                        label_text = m.group(4); url = f'mailto:{m.group(4)}'
                    elif m.group(5):
                        label_text = m.group(5); url = m.group(5)
                    else:
                        label_text = m.group(6); url = f'mailto:{m.group(6)}'
                    tag_name = f'link{link_count}'
                    txt.insert('end', label_text, (tag_name,))
                    # configure tag and binding
                    txt.tag_configure(tag_name, foreground='blue', underline=True)
                    txt.tag_bind(tag_name, '<Button-1>', lambda e, u=url: webbrowser.open(u))
                    link_count += 1
                    pos = m.end()
                if pos < len(item):
                    insert_with_bold(txt, item[pos:])
                txt.insert('end', '\n')
                continue

            # non-list line: handle links and inline bold
            pos = 0
            for m in link_re.finditer(ln):
                if m.start() > pos:
                    insert_with_bold(txt, ln[pos:m.start()])
                if m.group(1):
                    label_text = m.group(1); url = m.group(2)
                elif m.group(3):
                    label_text = m.group(3); url = m.group(3)
                elif m.group(4):
                    label_text = m.group(4); url = f'mailto:{m.group(4)}'
                elif m.group(5):
                    label_text = m.group(5); url = m.group(5)
                else:
                    label_text = m.group(6); url = f'mailto:{m.group(6)}'
                tag_name = f'link{link_count}'
                txt.insert('end', label_text, (tag_name,))
                txt.tag_configure(tag_name, foreground='blue', underline=True)
                txt.tag_bind(tag_name, '<Button-1>', lambda e, u=url: webbrowser.open(u))
                link_count += 1
                pos = m.end()
            if pos < len(ln):
                insert_with_bold(txt, ln[pos:])
            txt.insert('end', '\n')

        # make readonly
        try:
            txt.configure(state='disabled')
        except Exception:
            pass
        txt.pack(fill='both', expand=True, padx=4, pady=4)

    def _poll_tab(self):
        try:
            cur = self.tabview.get()
        except Exception:
            # fallback: no action
            cur = self._last_tab
        # guard: don't run if the app is closed or widget destroyed
        if getattr(self, '_closed', False):
            return
        if hasattr(self, 'winfo_exists') and not self.winfo_exists():
            return
        if cur != getattr(self, '_last_tab', None):
            # if we just left the Inputs tab, start a simulation
            if getattr(self, '_last_tab', None) == 'Inputs':
                # start background run if not already running
                if not getattr(self, '_running', False):
                    self._on_run()
        self._last_tab = cur
        try:
            self._poll_after_id = self.after(300, self._poll_tab)
        except Exception:
            self._poll_after_id = None

    def update_preview(self):
        """Update the small preview plot on the Inputs tab showing Kdown and Ta."""
        # guard: don't run if window is closed/destroyed
        if getattr(self, '_closed', False):
            return
        if hasattr(self, 'winfo_exists') and not self.winfo_exists():
            return
        
        # Read inputs (already validated elsewhere) — just retrieve values
        Sb = float(self.Sb.get())
        trise = int(float(self.trise_hr.get()) * hour)
        tset = int(float(self.tset_hr.get()) * hour)
        Ta_mean = float(self.Ta_mean_C.get()) + 273.15
        Ta_amp = float(self.Ta_amp_C.get())
        Ldown = float(self.Ldown.get())

        # plot over a 24 h window with a 10-min timestep
        tmax = 24 * hour
        dt = 600.0
        t = np.arange(0, tmax + dt, dt)
        TaK, S0 = diurnal_forcing(t, Ta_mean=Ta_mean, Ta_amp=Ta_amp, Sb=Sb, trise=trise, tset=tset)

        TaC = TaK - 273.15

        # Ta figure
        self.ax_Ta.clear()
        self.ax_Ta.plot(t / hour, TaC, color='tab:blue', label='Ta (°C)')
        self.ax_Ta.set_xlabel('Time (h)')
        self.ax_Ta.set_ylabel('Air temp (°C)')
        self.ax_Ta.set_xlim(0, 24)
        self.ax_Ta.grid(True, linestyle=':', alpha=0.5)
        self.ax_Ta.legend(loc='upper right')
        # avoid tight_layout clipping the right spine
        self.fig_Ta.subplots_adjust(right=0.98)
        # use tight_layout but reserve a small right margin so legends/spines are not clipped
        self.fig_Ta.tight_layout(rect=(0, 0, 0.98, 1))
        self.canvas_Ta.draw()

        # Kdown + Ldown figure
        self.ax_K.clear()
        self.ax_K.plot(t / hour, S0, color='tab:orange', label='Kdown (W/m2)')
        self.ax_K.plot(t / hour, np.full_like(t, Ldown), color='tab:purple', linestyle='--', label='Ldown (W/m2)')
        self.ax_K.set_xlabel('Time (h)')
        self.ax_K.set_ylabel('Flux (W/m2)')
        self.ax_K.set_xlim(0, 24)
        self.ax_K.grid(True, linestyle=':', alpha=0.5)
        self.ax_K.legend(loc='upper right')
        self.fig_K.subplots_adjust(right=0.98)
        # reserve right margin for legend
        self.fig_K.tight_layout(rect=(0, 0, 0.98, 1))
        self.canvas_K.draw()

    # --- preview auto-update helpers ---
    def _schedule_preview_update(self, delay_ms: int = 200):
        """Debounced schedule for preview updates (cancels previous scheduled call)."""
        pid = getattr(self, '_preview_after_id', None)
        if pid is not None:
            self.after_cancel(pid)
        self._preview_after_id = self.after(delay_ms, self.update_preview)

    def _install_preview_traces(self):
        """Install traces on input variables so changing them auto-updates the preview."""
        vars_to_trace = [
            getattr(self, 'Sb', None), getattr(self, 'trise_hr', None), getattr(self, 'tset_hr', None),
            getattr(self, 'Ldown', None), getattr(self, 'hcoef', None), getattr(self, 'Ta_mean_C', None),
            getattr(self, 'Ta_amp_C', None), getattr(self, 'beta_var', None),
            getattr(self, 'thickA', None), getattr(self, 'thickB', None),
            getattr(self, 'matA', None), getattr(self, 'matB', None), getattr(self, 'compare_var', None)
        ]
        for v in vars_to_trace:
            if v is None:
                continue
            try:
                # tkinter variables support trace_add in modern Python
                v.trace_add('write', lambda *a, _=v: self._schedule_preview_update())
            except Exception:
                # fallback to older trace
                v.trace('w', lambda *a, _=v: self._schedule_preview_update())

    # --- generic validators -------------------------------------------------
    def _var_for_key(self, key: str):
        """Return the tkinter Variable associated with a validation key.

        This maps logical parameter names (keys used in validators) to the
        tkinter variable attributes stored on the App instance.
        """
        mapping = {
            'Sb': 'Sb',
            'trise': 'trise_hr',
            'tset': 'tset_hr',
            'Ldown': 'Ldown',
            'h': 'hcoef',
            'Ta_mean': 'Ta_mean_C',
            'Ta_amp': 'Ta_amp_C',
            'beta': 'beta_var',
            'thickness_A': 'thickA',
            'thickness_B': 'thickB',
        }
        attr = mapping.get(key)
        if not attr:
            return None
        return getattr(self, attr, None)

    def _validate_nonneg(self, key: str):
        """Validate a non-negative numeric field (>= 0). Returns (ok, value)."""
        var = self._var_for_key(key)
        if var is None:
            return False, f"Internal error: missing UI variable for {key}."
        try:
            v = float(var.get())
            if v < 0:
                raise ValueError
        except Exception:
            return False, f"Invalid {key} â€” must be a non-negative number."
        return True, float(v)

    def _validate_positive(self, key: str):
        """Validate a strictly positive numeric field (> 0)."""
        var = self._var_for_key(key)
        if var is None:
            return False, f"Internal error: missing UI variable for {key}."
        try:
            v = float(var.get())
            if v <= 0:
                raise ValueError
        except Exception:
            return False, f"Invalid {key} â€” must be a positive number."
        return True, float(v)

    def _validate_hours(self, key: str):
        """Validate an hours-of-day field (0..24) and return seconds (int).

        The GUI stores hours; model uses seconds so the canonical value returned
        is in seconds.
        """
        var = self._var_for_key(key)
        if var is None:
            return False, f"Internal error: missing UI variable for {key}."
        try:
            vh = float(var.get())
            if not (0.0 <= vh <= 24.0):
                raise ValueError
            return True, int(vh * hour)
        except Exception:
            return False, f"Invalid {key} â€” enter hours between 0 and 24."

    def _validate_tempC(self, key: str):
        """Validate an air temperature in °C and return Kelvin."""
        var = self._var_for_key(key)
        if var is None:
            return False, f"Internal error: missing UI variable for {key}."
        try:
            v_c = float(var.get())
        except Exception:
            return False, f"Invalid {key} â€” enter a numeric Â°C value."
        return True, float(v_c) + 273.15

    def _validate_any_number(self, key: str):
        """Generic numeric validator with no sign constraints."""
        var = self._var_for_key(key)
        if var is None:
            return False, f"Internal error: missing UI variable for {key}."
        try:
            v = float(var.get())
        except Exception:
            return False, f"Invalid {key} â€” enter a numeric value."
        return True, float(v)

    def _validate_and_store(self, key: Optional[str] = None, event: Optional[object] = None):
        """Validate input fields on focus-out and store canonical values in self._params.

        This method is deliberately conservative: invalid values are replaced by
        sensible defaults from `DEFAULTS` and the corresponding tkinter variable
        is updated so the user sees the corrected value.
        """
        # helper to safely set a doublevar (and clamp/convert)
        def set_doublevar(var, val):
            var.set(val)

        # helper to map validation keys to entry widgets and readable names
        key_map = {
            'Sb': (self.entry_Sb, 'Peak shortwave (W/m2)'),
            'trise': (self.entry_trise, 'Sunrise time (h)'),
            'tset': (self.entry_tset, 'Sunset time (h)'),
            'Ldown': (self.entry_Ldown, 'Incoming longwave (W/m2)'),
            'h': (self.entry_hcoef, 'Heat transfer coeff (W/m2K)'),
            'Ta_mean': (self.entry_Ta_mean, 'Air mean temp (°C)'),
            'Ta_amp': (self.entry_Ta_amp, 'Air temperature amp (°C)'),
            'beta': (self.entry_beta, 'Bowen ratio (-)'),
            'thickness_A': (self.entry_thickA, 'Thickness A (m)'),
            'thickness_B': (self.entry_thickB, 'Thickness B (m)'),
        }

        def focus_and_select(entry_widget):
            entry_widget.focus_set()
            try:
                # CTkEntry exposes selection methods via underlying tk widget
                entry_widget.select_range(0, 'end')
            except Exception:
                w = getattr(entry_widget, 'entry', None) or getattr(entry_widget, 'master', None)
                if w is not None and hasattr(w, 'select_range'):
                    w.select_range(0, 'end')

        def show_fix_message(entry_widget, label, msg):
            # show modal dialog and return focus to the offending widget
            messagebox.showerror('Invalid input', msg)
            try:
                # restore focus to the widget after the dialog closes
                self.after(1, lambda: focus_and_select(entry_widget))
            except Exception:
                focus_and_select(entry_widget)

        # validation routines for each key. Use the generic validators above
        # so each validator only needs the logical parameter name.
        validators = {
            'Sb': lambda: self._validate_nonneg('Sb'),
            'trise': lambda: self._validate_hours('trise'),
            'tset': lambda: self._validate_hours('tset'),
            'Ldown': lambda: self._validate_nonneg('Ldown'),
            'h': lambda: self._validate_nonneg('h'),
            'Ta_mean': lambda: self._validate_tempC('Ta_mean'),
            'Ta_amp': lambda: self._validate_nonneg('Ta_amp'),
            'beta': lambda: self._validate_any_number('beta'),
            'thickness_A': lambda: self._validate_positive('thickness_A'),
            'thickness_B': lambda: self._validate_positive('thickness_B'),
        }

        # if no specific key given, validate all and stop at first error
        keys_to_check = [key] if key else list(validators.keys())
        for kk in keys_to_check:
            validator = validators.get(kk)
            if validator is None:
                continue
            ok, res = validator()
            if not ok:
                # show dialog and return focus to offending entry, do NOT overwrite the user's input
                entry_widget, label = key_map.get(kk, (None, kk))
                msg = f"{label}: {res}\n\nPlease correct the value."
                if entry_widget is not None:
                    show_fix_message(entry_widget, label, msg)
                else:
                    messagebox.showerror('Invalid input', msg)
                    return False
                    
            # on success, store canonical value
            self._params[kk if kk not in ('thickness_A', 'thickness_B') else kk] = res
            self._params[kk] = res
        return True

    # --- simple tooltip implementation (CTk-styled when possible, fallback to Tk)
    def _schedule_show_material_tooltip(self, widget, var, delay=300):
        """Schedule showing a tooltip for the material referenced by the tkinter variable `var`.

        We debounce using `after` so the tooltip only appears when the cursor lingers.
        """
        # if this is Material B's widget and compare is disabled, don't show
        if widget is getattr(self, 'opt_matB', None) and not bool(getattr(self, 'compare_var', ctk.BooleanVar(value=False)).get()):
            return
        # cancel any existing schedule
        pid = getattr(self, '_tooltip_after_id', None)
        if pid is not None:
            self.after_cancel(pid)
 
        # schedule new show
        self._tooltip_after_id = self.after(delay, lambda: self._show_material_tooltip(widget, var))
 
        # immediate fallback: show now
        self._show_material_tooltip(widget, var)

    def _show_material_tooltip(self, widget, var):
        """Create and show a small tooltip window near `widget` with material properties.

        `var` is a tkinter variable whose value is the material key name; we load
        the material and display a few fields (k, rho, cp, albedo, emissivity, evaporation).
        """
        # ensure any scheduled show id is cleared
        try:
            self._tooltip_after_id = None
        except Exception:
            pass
        # destroy existing tooltip if present (assign to local var so
        # static checkers can reason about None-ness)
        tw = getattr(self, '_tooltip_win', None)
        if tw is not None:
            tw.destroy()
        self._tooltip_win = None
 
        # defensive: if this is Material B and compare is off, don't show
        if widget is getattr(self, 'opt_matB', None) and not bool(getattr(self, 'compare_var', ctk.BooleanVar(value=False)).get()):
            return

        mat_key = None
        try:
            mat_key = var.get()
        except Exception:
            try:
                mat_key = str(var)
            except Exception:
                mat_key = None
        if not mat_key:
            return

        # attempt to load material (best-effort; load_material may raise)
        try:
            mat = load_material(mat_key)
        except Exception:
            mat = None

        # build tooltip text
        lines = [f"{mat_key}"]
        try:
            if mat is None:
                lines.append('(no data)')
            else:
                # prefer mapping access when the material is a dict
                mat_map: Optional[Mapping[str, Any]] = mat if isinstance(mat, Mapping) else None

                def g(k, fmt='{:.3g}'):
                    try:
                        if mat_map is not None:
                            v = mat_map.get(k)
                        else:
                            v = getattr(mat, k)
                        return fmt.format(v) if v is not None else 'â€”'
                    except Exception:
                        return 'â€”'

                k_val = g('k')
                rho = g('rho')
                cp = g('cp')
                alb = g('albedo')
                emis = g('emissivity')
                evap = False
                try:
                    if mat_map is not None:
                        evap = bool(mat_map.get('evaporation', False))
                    else:
                        evap = bool(getattr(mat, 'evaporation', False))
                except Exception:
                    evap = False

                lines.append(f"k={k_val} W/mK  rho={rho} kg/m3  cp={cp} J/kgK")
                lines.append(f"albedo={alb}  emissivity={emis}  evap={evap}")
        except Exception:
            lines.append('(error fetching properties)')

        text = '\n'.join(lines)

        # create tooltip window
        tw = tk.Toplevel(self)
        tw.wm_overrideredirect(True)
        tw.wm_attributes('-topmost', True)
        # style with CTkLabel if available so it matches the app theme
        try:
            lbl = ctk.CTkLabel(tw, text=text, justify='left', padx=8, pady=6)
            lbl.pack()
        except Exception:
            lbl = tk.Label(tw, text=text, justify='left', bg='#ffffe0', relief='solid', bd=1)
            lbl.pack()
        # position near the widget (to the right and slightly below)
        try:
            wx = widget.winfo_rootx()
            wy = widget.winfo_rooty()
            ww = widget.winfo_width()
            wh = widget.winfo_height()
            tw.geometry(f'+{wx + ww + 8}+{wy + max(0, wh//2 - 8)}')
        except Exception:
            # geometry failed; still show the tooltip at default location
            pass

        # keep a reference so hide can destroy it; bind leave on the tooltip
        # itself so moving into the tooltip still hides it when the pointer
        # leaves the tooltip window.
        try:
            tw.bind('<Leave>', lambda e: self._hide_material_tooltip())
            tw.bind('<Enter>', lambda e: None)
        except Exception:
            pass
        self._tooltip_win = tw
        # remember origin and bind a motion handler so we hide the tooltip
        # when the pointer leaves both the origin widget and the tooltip.
        try:
            self._tooltip_origin = widget
        except Exception:
            self._tooltip_origin = None
        try:
            self.bind('<Motion>', self._tooltip_motion_handler)
        except Exception:
            pass

    def _hide_material_tooltip(self):
        """Hide/destroy any visible tooltip and cancel scheduled shows."""
        try:
            pid = getattr(self, '_tooltip_after_id', None)
            if pid is not None:
                self.after_cancel(pid)
                self._tooltip_after_id = None
            tw = getattr(self, '_tooltip_win', None)
            if tw is not None:
                tw.destroy()
        except Exception:
            pass
        # clear origin and unbind motion handler
        try:
            self._tooltip_origin = None
        except Exception:
            pass
        try:
            self.unbind('<Motion>')
        except Exception:
            pass
        self._tooltip_win = None

    def _tooltip_motion_handler(self, event=None):
        """Hide tooltip when mouse leaves both the origin widget and tooltip."""
        tw = getattr(self, '_tooltip_win', None)
        if tw is None:
            try:
                self.unbind('<Motion>')
            except Exception:
                pass
            return
        origin = getattr(self, '_tooltip_origin', None)
        try:
            px = self.winfo_pointerx()
            py = self.winfo_pointery()
        except Exception:
            # unable to query pointer; be conservative and hide
            self._hide_material_tooltip()
            return
        # check tooltip geometry
        try:
            tx = tw.winfo_rootx(); ty = tw.winfo_rooty(); tw_w = tw.winfo_width(); tw_h = tw.winfo_height()
            if tx <= px <= tx + tw_w and ty <= py <= ty + tw_h:
                return
        except Exception:
            pass
        # check origin geometry
        if origin is not None:
            try:
                ox = origin.winfo_rootx(); oy = origin.winfo_rooty(); ow = origin.winfo_width(); oh = origin.winfo_height()
                if ox <= px <= ox + ow and oy <= py <= oy + oh:
                    return
            except Exception:
                pass
        # pointer is outside both -> hide
        self._hide_material_tooltip()

    def _mpl_color_from_ctk(self, c):
        """Convert a CTk color (hex string or tuple) to an MPL-compatible color.

        CTk may return colors as HEX strings like '#rrggbb' or as tuples.
        Handle common cases and fall back to the input unchanged.
        """
        if c is None:
            return None
        # if it's a string, try to convert Tk-style color names (e.g. 'gray86')
        # to an RGB tuple Matplotlib accepts. If it's a hex string, return it
        # unchanged. Fall back to returning the original input on failure.
        if isinstance(c, str):
            try:
                # hex strings are already acceptable for Matplotlib
                if c.startswith('#'):
                    return c
                # try to resolve the color via the underlying Tk color parser
                # winfo_rgb returns 0..65535 values
                rgb16 = self.winfo_rgb(c)
                r = rgb16[0] / 65535.0
                g = rgb16[1] / 65535.0
                b = rgb16[2] / 65535.0
                return (r, g, b)
            except Exception:
                # fallback: return original string (may be a matplotlib name)
                return c
        # tuple-like: try to coerce to an RGB or RGBA tuple of floats in 0..1
        if isinstance(c, tuple) and all(isinstance(x, (int, float)) for x in c):
            vals = [float(x) / 255.0 if (isinstance(x, (int,)) and x > 1) else float(x) for x in c]
            # ensure length 3 or 4
            if len(vals) >= 3:
                if len(vals) == 3:
                    return (vals[0], vals[1], vals[2])
                else:
                    # keep first 4 components as RGBA
                    return (vals[0], vals[1], vals[2], vals[3])
            # fallthrough
        return None

    def _mat_allows_evap(self, mat: Optional[object]) -> bool:
        """Return True if the given material (dict-like or object) enables evaporation.

        This helper accepts either a mapping (with .get) or an object with an
        attribute `evaporation` so it's robust to the material representation
        returned by load_material.
        """
        try:
            if mat is None:
                return False
            # prefer Mapping checks so static checkers know .get exists
            if isinstance(mat, Mapping):
                return bool(mat.get('evaporation', False))
            return bool(getattr(mat, 'evaporation', False))
        except Exception:
            return False

    def _on_compare_toggle(self):
        """Enable/disable Material B controls when the Compare checkbox changes."""
        try:
            enabled = bool(self.compare_var.get())
        except Exception:
            enabled = False
        state = 'normal' if enabled else 'disabled'
        try:
            if getattr(self, 'opt_matB', None) is not None:
                self.opt_matB.configure(state=state)
        except Exception:
            pass
        try:
            if getattr(self, 'entry_thickB', None) is not None:
                self.entry_thickB.configure(state=state)
        except Exception:
            pass
        # visually indicate Material B entry and label color when compare toggles
        try:
            if not enabled:
                # set both the entry text and the label to gray when disabled
                try:
                    self.entry_thickB.configure(text_color='gray')
                except Exception:
                    pass
                try:
                    if getattr(self, '_lbl_thickB_color_default', None) is not None:
                        self.lbl_thickB.configure(text_color='gray')
                    else:
                        self.lbl_thickB.configure(text_color='gray')
                except Exception:
                    pass
            else:
                # restore label and entry to their saved defaults (best-effort)
                try:
                    if getattr(self, '_entry_thickB_text_color_default', None) is not None:
                        self.entry_thickB.configure(text_color=self._entry_thickB_text_color_default)
                    else:
                        # try removing the override
                        self.entry_thickB.configure(text_color=None)
                except Exception:
                    pass
                try:
                    if getattr(self, '_lbl_thickB_color_default', None) is not None:
                        self.lbl_thickB.configure(text_color=self._lbl_thickB_color_default)
                    else:
                        self.lbl_thickB.configure(text_color=None)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            # update animation background/layout for new compare state
            self.draw_anim_static()
        except Exception:
            pass

    # status text stored internally (no visible widget)
    # any code that previously called _set_status now writes to _status_text

    # Compare UI handling removed: comparison checkbox exists but behavior is
    # handled when running the simulation; Material B controls are enabled/disabled
    # during initial setup below.

    # NOTE: configure_animation_axes removed — animation tab now uses a single
    # axes stored in `self.ax_anim`. Calls that previously invoked
    # configure_animation_axes have been replaced with `draw_anim_static()`.

    def draw_anim_static(self):
        # draw a simple background showing material depths
        # guard: avoid drawing if app is closed
        if getattr(self, '_closed', False):
            return
        # require that self.ax_anim exists
        if not getattr(self, 'ax_anim', None):
            return
        if hasattr(self, 'winfo_exists') and not self.winfo_exists():
            return
        ax = self.ax_anim
        ax.clear()

        # determine thickness to show; prefer max of A and B when comparing
        thickA = float(self.thickA.get())
        thickB = float(self.thickB.get()) if getattr(self, 'compare_var', None) and bool(self.compare_var.get()) else 0.0
        thick = max(thickA, thickB, 0.1)

        # y-limits per request: [-thickness, thickness/2]
        ymin = -thick
        ymax = thick / 2.0
        ax.set_ylim(ymin, ymax)

        # X limits: if we have anim_data, set to full range of profile temps
        try:
            anim = getattr(self, 'anim_data', None)
            if anim and anim.get('A') is not None:
                outA = anim.get('A')
                tps = np.asarray(outA.get('T_profiles', [])) - 273.15
                xmin = float(np.nanmin(tps)) if tps.size else -10.0
                xmax = float(np.nanmax(tps)) if tps.size else 40.0
                # include B if present
                if anim.get('B') is not None and bool(self.compare_var.get()):
                    outB = anim.get('B')
                    tpsb = np.asarray(outB.get('T_profiles', [])) - 273.15
                    if tpsb.size:
                        xmin = min(xmin, float(np.nanmin(tpsb)))
                        xmax = max(xmax, float(np.nanmax(tpsb)))
                # pad slightly and ensure left limit shows cold temps (at least -10°C)
                span = max(0.5, xmax - xmin)
                xmin_pad = xmin - 0.05 * span
                xmin_pad = min(xmin_pad, -10.0)
                ax.set_xlim(xmin_pad, xmax + 0.05 * span)
            else:
                ax.set_xlim(0, 30)
        except Exception:
            try:
                ax.set_xlim(0, 30)
            except Exception:
                pass

        # soil patch: light grey behind the temperature profile from surface (0) down to -thick
        try:
            ax.axhspan(ymin=-thick, ymax=0.0, facecolor='lightgrey', zorder=0, alpha=0.5)
        except Exception:
            pass

        # solid surface line at z=0
        ax.axhline(y=0.0, color='k', linewidth=1.0, zorder=3)

        # if comparing and thicknesses differ, draw a dash-dot line at the bottom of
        # the thinner layer to denote its base
        if self.compare_var.get():
            thickA = float(self.thickA.get())
            thickB = float(self.thickB.get())
            # choose the larger for the filled patch (already used); show a dash-dot
            # at the bottom of the smaller (if different by more than eps)
            if abs(thickA - thickB) > 1e-6:
                other_thick = min(thickA, thickB)
                ax.axhline(y=-other_thick, color='k', linestyle='-.', linewidth=1.0, zorder=3)

        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Depth (m)')
        # initial title (time will be updated each frame) — show time in axes title
        try:
            ax.set_title('Time: -- h')
        except Exception:
            pass

        # draw initial temperature profile (time index 0) if available so the
        # user sees the initial condition immediately when the screen loads.
        try:
            anim = getattr(self, 'anim_data', None)
            if anim and anim.get('A') is not None:
                outA = anim.get('A')
                if 'T_profiles' in outA and len(outA['T_profiles']):
                    z = outA.get('z', None)
                    if z is not None:
                        try:
                            ax.plot(outA['T_profiles'][0] - 273.15, z, '-r', alpha=0.9, zorder=2, label='A')
                        except Exception:
                            pass
                # plot B initial if present and comparing
                if anim.get('B') is not None and bool(getattr(self, 'compare_var', ctk.BooleanVar(value=False)).get()):
                    outB = anim.get('B')
                    if 'T_profiles' in outB and len(outB['T_profiles']):
                        zB = outB.get('z', None)
                        if zB is not None:
                            try:
                                ax.plot(outB['T_profiles'][0] - 273.15, zB, '-b', alpha=0.9, zorder=2, label='B')
                            except Exception:
                                pass
                try:
                    ax.legend(loc='upper right', fontsize='small')
                except Exception:
                    pass
        except Exception:
            pass
        # keep surface at top (invert so 0 is near the top)
        # keep normal y-axis orientation so depth increases downward
        try:
            self.canvas_anim.draw()
        except Exception:
            pass

    # --- Animation control ---
    def toggle_animation(self):
        if not getattr(self, 'anim_data', None):
            messagebox.showerror('Error', 'Run a simulation first')
            return
        self.animating = not self.animating
        self.btn_start.configure(text='Stop' if self.animating else 'Start')
        if self.animating:
            # start loop
            self.animate_step()

    def animate_step(self):
        # guard: stop if app closed or animation flag cleared
        if getattr(self, '_closed', False):
            self.animating = False
            return
        if not self.animating:
            return
        if hasattr(self, 'winfo_exists') and not self.winfo_exists():
            self.animating = False
            return
        anim_data = getattr(self, 'anim_data', None)
        if anim_data is None:
            # nothing to animate
            self.animating = False
            return
        outA = anim_data.get('A')
        matA = anim_data.get('A_mat')
        outB = anim_data.get('B') if self.compare_var.get() else None
        matB = anim_data.get('B_mat') if self.compare_var.get() else None
        times = outA['t']
        i = int(self.anim_idx % len(times))
        ax = getattr(self, 'ax_anim', None)
        if ax is None:
            self.animating = False
            return
        ax.clear()

        thickA = float(self.thickA.get())
        thickB = float(self.thickB.get()) if getattr(self, 'compare_var', None) and bool(self.compare_var.get()) else 0.0
        thick = max(thickA, thickB, 0.1)

        ymin = -thick
        ymax = thick / 2.0

        # compute x-limits from data (A and optionally B)
        xmin, xmax = None, None
        try:
            tA = np.asarray(outA.get('T_profiles', [])) - 273.15
            if tA.size:
                xmin = float(np.nanmin(tA))
                xmax = float(np.nanmax(tA))
        except Exception:
            pass
        if outB is not None:
            try:
                tB = np.asarray(outB.get('T_profiles', [])) - 273.15
                if tB.size:
                    val_min = float(np.nanmin(tB))
                    val_max = float(np.nanmax(tB))
                    xmin = val_min if xmin is None else min(xmin, val_min)
                    xmax = val_max if xmax is None else max(xmax, val_max)
            except Exception:
                pass
        if xmin is None or xmax is None or not np.isfinite(xmin) or not np.isfinite(xmax):
            xmin, xmax = -10.0, 30.0
        # pad a little and ensure left bound at least -10°C so cold profiles are visible
        span = max(0.5, xmax - xmin)
        xmin = xmin - 0.05 * span
        xmin = min(xmin, -10.0)
        xmax = xmax + 0.05 * span

        # draw soil patch first (so it's behind) and set limits
        try:
            ax.axhspan(ymin=-thick, ymax=0.0, facecolor='lightgrey', zorder=0, alpha=0.5)
        except Exception:
            pass

        # solid surface line at z=0
        ax.axhline(y=0.0, color='k', linewidth=1.0, zorder=3)
        # dash-dot line for bottom of thinner layer when comparing
        if self.compare_var.get():
            thickA = float(self.thickA.get())
            thickB = float(self.thickB.get())
            # choose the larger for the filled patch (already used); show a dash-dot
            # at the bottom of the smaller (if different by more than eps)
            if abs(thickA - thickB) > 1e-6:
                other_thick = min(thickA, thickB)
                ax.axhline(y=-other_thick, color='k', linestyle='-.', linewidth=1.0, zorder=3)
 
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        try:
            ax.set_ylim(ymin, ymax)
        except Exception:
            pass

        # plot A on top of the patch
        try:
            ax.plot(outA['T_profiles'][i] - 273.15, outA['z'], '-r', label='A', zorder=2)
        except Exception:
            pass
        # plot B (on same axes) if present
        if outB is not None:
            try:
                ax.plot(outB['T_profiles'][i] - 273.15, outB['z'], '-b', label='B', zorder=2)
            except Exception:
                pass

        # axis labels and orientation
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('Depth (m)')
        # keep normal y-axis orientation so depth increases downward
        # legend
        try:
            ax.legend(loc='upper right', fontsize='small')
        except Exception:
            pass

        # draw SEB arrows for A and B (A first so arrows overlay nicely)
        try:
            self.draw_seb_arrows(ax, outA, i, mat=matA)
        except Exception:
            pass
        if outB is not None:
            try:
                self.draw_seb_arrows(ax, outB, i, mat=matB)
            except Exception:
                pass

        # update time label (show hours if available) and figure title
        try:
            tval = float(times[i]) / float(hour)
            self.time_label.configure(text=f'Time: {tval:.2f} h')
            try:
                # put time in the axes title
                ax.set_title(f'Time: {tval:.2f} h')
            except Exception:
                pass
        except Exception:
            pass

        try:
            self.canvas_anim.draw()
        except Exception:
            pass

        self.anim_idx += 1
        # schedule next (store id so it can be cancelled)
        delay = int(max(50, 500 / max(0.001, float(self.speed_var.get() or 1.0))))
        try:
            self._anim_after_id = self.after(delay, self.animate_step)
        except Exception:
            self._anim_after_id = None
            self.animating = False

    # draw SEB arrows helper (reusable)
    def draw_seb_arrows(self, ax, out, idx, mat: Optional[object] = None):
        # draw labeled arrows in axes-fraction coordinates so they appear at
        # a consistent location regardless of data limits. Direction mapping:
        #  - K* (net shortwave) : downward when incoming
        #  - L* (net longwave)  : downward when net incoming
        #  - G (ground)         : downward when positive into the ground
        #  - H (sensible)       : upward when positive (surface->air)
        #  - E (latent)         : upward when positive
        Kstar = out['Kstar'][idx]
        Lstar = out['Lstar'][idx]
        H = out['H'][idx]
        E = out['E'][idx]
        G = out['G'][idx]
        # order values to match desired label order: K*, L*, G, H, E
        vals = [Kstar, Lstar, G, H, E]
        maxv = max(1.0, max(abs(v) for v in vals))
        # Arrange arrows side-by-side at the surface (z=0) in data coordinates.
        # If this `out` corresponds to Material A, center the arrows at T=0°C
        # so they appear around x=0; if it's Material B, place them on the
        # right-hand side. Use a tighter horizontal packing so arrows are
        # visually closer together.
        # show arrows in requested order: K*, L*, G, H, E
        labels = ['K*', 'L*', 'G', 'H', 'E']
        # map colors to the same order
        colors = ['orange', 'magenta', 'saddlebrown', 'green', 'blue']
        try:
            x0, x1 = ax.get_xlim()
            y0, y1 = ax.get_ylim()
            x_width = x1 - x0 if (x1 - x0) != 0 else 1.0
            y_height = y1 - y0 if (y1 - y0) != 0 else 1.0
        except Exception:
            x0, x1 = 0.0, 1.0
            y0, y1 = 0.0, 1.0
            x_width = 1.0
            y_height = 1.0

        # tighter packing than before
        block_frac = 0.04  # 4% of axis width per arrow
        right_margin_frac = 0.02

        centers = []
        # Decide whether this out is Material A by identity (fallback to False)
        try:
            ad = getattr(self, 'anim_data', None)
            isA = (ad is not None and out is ad.get('A'))
        except Exception:
            isA = False

        thickA = float(self.thickA.get())
        thickB = float(self.thickB.get()) if getattr(self, 'compare_var', None) and bool(self.compare_var.get()) else 0.0
        thick = thickA if isA else thickB

        if isA:
            # center arrows around x=0 (data coordinate). Use small dx spacing
            dx = block_frac * x_width
            mid = 0.0
            # centers left-to-right around mid
            for k in range(len(labels)):
                cx = mid + (k - (len(labels) - 1) / 2.0) * dx
                centers.append(cx)
        else:
            # place arrows near the right edge, arranged right-to-left
            for k in range(len(labels)):
                center_frac = 1.0 - right_margin_frac - (k + 0.5) * block_frac
                cx = x0 + center_frac * x_width
                centers.append(cx)

        for k, v in enumerate(vals):
            # skip near-zero terms to avoid clutter (no arrow or descriptor)
            eps = 1e-6
            if abs(v) <= eps:
                continue
            # arrow length as fraction of axis height, scaled by flux magnitude
            frac = 0.08 * (abs(v) / maxv)
            # make arrows 25% longer as requested
            dy = frac * 1.25 * y_height
            cx = centers[k]
            # start at surface z=0 in data coordinates
            start = (cx, 0.0)
            # determine direction based on sign and term type
            # For K*, L*, G (indices 0,1,2): positive -> downward, negative -> upward
            # For H, E (indices 3,4): positive -> upward, negative -> downward
            if k in (0, 1, 2):
                end = (cx, -abs(dy) if v >= 0 else abs(dy))
                va = 'top' if v >= 0 else 'bottom'
            else:
                end = (cx, abs(dy) if v >= 0 else -abs(dy))
                va = 'bottom' if v >= 0 else 'top'
            # draw arrow in data coordinates so it anchors at z=0
            try:
                ax.annotate('', xy=end, xytext=start, xycoords='data', textcoords='data', arrowprops=dict(arrowstyle='-|>', color=colors[k], lw=2))
            except Exception:
                # fallback to axes-fraction drawing
                fx = (cx - x0) / x_width if x_width != 0 else 0.9
                ay = 0.5
                ay2 = ay - (0.05 if end[1] < 0 else -0.05)
                ax.annotate('', xy=(fx, ay), xytext=(fx, ay2), xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color=colors[k], lw=2))

            # label and value: place slightly offset vertically from a fixed baseline near the tip
            try:
                txt = f"{labels[k]} {v:+.0f} W/m2"
                label_offset = 0.02 * y_height
                tip_label_y = end[1] - label_offset if end[1] < 0 else end[1] + label_offset
                ax.text(cx, tip_label_y, txt, ha='center', va='center', fontsize=9, color=colors[k])
            except Exception:
                pass

        # place material label above the arrow group at a height that is
        # consistent for both A and B (use the axes y-limits which were set
        # based on the maximum thickness). This ensures labels for A and B
        # are plotted at the same vertical position even when thicknesses
        # differ.
        try:
            # use current y-limits to infer the largest thickness shown
            try:
                ay0, ay1 = ax.get_ylim()
            except Exception:
                ay0, ay1 = y0, y1
            # ay0 is usually the negative depth baseline (e.g. -thickness)
            if ay0 is not None and ay0 < 0:
                total_thick = abs(ay0)
            else:
                total_thick = abs(ay1) if ay1 is not None else abs(thick)
            mat_label_y = 0.4 * total_thick
            group_x = sum(centers) / len(centers) if centers else (x0 + x1) / 2.0
            mat_name = 'Material A' if isA else 'Material B'
            ax.text(group_x, mat_label_y, mat_name, ha='center', va='bottom', fontsize=10, fontweight='bold')
        except Exception:
            pass

    def update_tempseb(self, idx: int):
        # draw temperature profile and SEB arrows at given time index (index into out['times'])
        anim_data = getattr(self, 'anim_data', None)
        if anim_data is None:
            return
        outA = anim_data.get('A')
        outB = anim_data.get('B') if self.compare_var.get() else None
        self.ax_temp.clear()
        self.ax_temp.plot(outA['T_profiles'][idx] - 273.15, outA['z'], '-r', label='A')
        if outB is not None:
            self.ax_temp.plot(outB['T_profiles'][idx] - 273.15, outB['z'], '-b', label='B')
        # set y-limits consistent with animation view: [-thickness, thickness/2]
        try:
            thickA = float(self.thickA.get())
        except Exception:
            thickA = 0.2
        try:
            thickB = float(self.thickB.get()) if getattr(self, 'compare_var', None) and bool(self.compare_var.get()) else 0.0
        except Exception:
            thickB = 0.0
        thick = max(thickA, thickB, 0.1)
        try:
            self.ax_temp.set_ylim(-thick, thick / 2.0)
        except Exception:
            pass
        # draw soil patch and surface line similar to animation view
        try:
            self.ax_temp.axhspan(ymin=-thick, ymax=0.0, facecolor='lightgrey', zorder=0, alpha=0.5)
        except Exception:
            pass
        try:
            self.ax_temp.axhline(y=0.0, color='k', linewidth=1.0, zorder=3)
        except Exception:
            pass
        try:
            if getattr(self, 'compare_var', None) and bool(self.compare_var.get()):
                if abs(thickA - thickB) > 1e-6:
                    other_thick = min(thickA, thickB)
                    self.ax_temp.axhline(y=-other_thick, color='k', linestyle='-.', linewidth=1.0, zorder=3)
        except Exception:
            pass
        # keep normal y-axis orientation so depth increases downward
        self.ax_temp.set_xlabel('Temperature (Â°C)')
        self.ax_temp.set_ylabel('Depth (m)')
        # draw arrows for A on same axis (offset x slightly)
        self.draw_seb_arrows(self.ax_temp, outA, idx)
        if outB is not None:
            # for B, draw arrows slightly left by translating transform
            self.draw_seb_arrows(self.ax_temp, outB, idx)
        try:
            self.canvas_temp.draw()
        except Exception:
            pass

    # --- Temp & SEB playback ---
    def toggle_temp_playback(self):
        if not getattr(self, 'anim_data', None):
            messagebox.showerror('Error', 'Run a simulation first')
            return
        self.temp_playing = not self.temp_playing
        self.temp_play_btn.configure(text='Stop' if self.temp_playing else 'Play')
        if self.temp_playing:
            self.temp_play_step()

    def temp_play_step(self):
        # guard: stop if app closed or playback flag cleared
        if getattr(self, '_closed', False):
            self.temp_playing = False
            return
        if not self.temp_playing:
            return
        if hasattr(self, 'winfo_exists') and not self.winfo_exists():
            self.temp_playing = False
            return
        anim_data = getattr(self, 'anim_data', None)
        if anim_data is None:
            self.temp_playing = False
            return
        outA = anim_data.get('A')
        n = len(outA['times'])
        # advance slider
        try:
            cur = int(float(self.temp_slider.get()))
        except Exception:
            cur = 0
        nxt = (cur + 1) % max(1, n)
        self.temp_slider.set(nxt)
        self.update_tempseb(nxt)
        delay = int(max(100, 1000 / max(0.001, float(self.speed_var.get() or 1.0))))
        try:
            self._temp_after_id = self.after(delay, self.temp_play_step)
        except Exception:
            self._temp_after_id = None
            self.temp_playing = False

    # --- Run simulation and populate results + animation data ---
    def _on_run(self):
        # Ensure the latest inputs are validated and stored before starting
        # the background run. _validate_and_store reads current tkinter
        # variable values (no need to rely on focus-out) and will show an
        # error dialog if values are invalid. If validation fails do not
        # start the simulation thread.
        try:
            ok = self._validate_and_store()
        except Exception:
            ok = False
        if not ok:
            return

        t = threading.Thread(target=self._run_thread, daemon=True)
        t.start()

    def _run_thread(self):
        # mark running and update status (store internally only)
        self._running = True
        self._status_text = 'Running simulation...'
        try:
            # inputs are validated when the user leaves each input box and saved
            # into self._params by _validate_and_store. Here we simply use those
            # canonical values and proceed to run the model.
            p = getattr(self, '_params', {})
            Sb = p.get('Sb', MODEL_DEFAULTS['Sb'])
            trise = p.get('trise', int(MODEL_DEFAULTS['trise']))
            tset = p.get('tset', int(MODEL_DEFAULTS['tset']))
            Ldown = p.get('Ldown', MODEL_DEFAULTS['Ldown'])
            hcoef = p.get('h', MODEL_DEFAULTS['h'])

            # MODEL_DEFAULT stores Ta_mean in Kelvin and Ta_amp as amplitude
            Ta_mean = p.get('Ta_mean', float(MODEL_DEFAULTS['Ta_mean']))
            Ta_amp = p.get('Ta_amp', float(MODEL_DEFAULTS['Ta_amp']))

            # solver time grid (coarse) and forcing resolution (high-res for interpolation)
            tmax = MODEL_DEFAULTS['tmax']
            dt = MODEL_DEFAULTS['dt']

            # prefer passing a high-resolution forcing interval to the model
            # and supply precomputed forcing arrays (t, Ta, Kdown). This keeps
            # the forcing construction in the GUI and ensures the same forcing
            # is used for both A and B runs.
            forcing_dt = float(self.GUIDEFAULTS['forcing_dt'])
            forcing_t = np.arange(0.0, float(tmax) + forcing_dt, forcing_dt)
            Ta_arr, S0_arr = diurnal_forcing(forcing_t, Ta_mean=Ta_mean, Ta_amp=Ta_amp, Sb=Sb, trise=trise, tset=tset)
            # include Ldown as a time series (constant array) in the forcing
            Ldown_arr = np.full_like(forcing_t, float(Ldown))
            forcing = {'t': forcing_t, 'Ta': Ta_arr, 'Kdown': S0_arr, 'Ldown': Ldown_arr}

            params = {
                'beta': float(p.get('beta', MODEL_DEFAULTS['beta'])),
                'h': float(p.get('h', MODEL_DEFAULTS['h'])),
                'forcing': forcing,
                'thickness': float(p['thickness_A'])
            }
            mA = load_material(self.matA.get())
            # run_simulation expects (mat, params, dt, tmax) where params is a dict
            outA = run_simulation(mA, params, dt, tmax)
            outB = None
            mB = None
            if self.compare_var.get():
                mB = load_material(self.matB.get())
                params_b = params.copy()
                params_b['thickness'] = float(p['thickness_B'])
                outB = run_simulation(mB, params_b, dt, tmax)

            # store for animation (include material metadata so we know if
            # evaporation is enabled per material)
            self.anim_data = {'A': outA, 'A_mat': mA, 'B': outB, 'B_mat': mB}

            # update UI on main thread (pass material dicts so results plotting
            # can suppress latent flux where evaporation is disabled)
            self.after(0, lambda: self._show_results(outA, outB, mA, mB))
            self.after(0, lambda: self.draw_anim_static())
            # configure temp slider to match number of time steps
            nsteps = len(outA['t'])
            # configure CTkSlider range (from_ remains 0)
            self.temp_slider.configure(to=max(1, nsteps - 1), number_of_steps=max(1, nsteps - 1))
            self.temp_slider.set(0)
            self.btn_start.configure(state='normal')
            self.after(0, lambda: setattr(self, '_status_text', 'Simulation complete'))
        except Exception as exc:
            # print full traceback to terminal so users/developers can see details
            try:
                tb = traceback.format_exc()
                print(tb, file=sys.stderr)
            except Exception:
                pass
            # also log via standard logging (captures stack trace)
            try:
                logging.getLogger(__name__).exception('Simulation failed in background thread')
            except Exception:
                pass

            # show compact GUI error and update status
            self.after(0, lambda: messagebox.showerror('Error', f'Simulation failed:\n{exc}'))
            self.after(0, lambda: setattr(self, '_status_text', 'Error during simulation'))
        finally:
            # clear running flag on main thread
            try:
                self.after(0, lambda: setattr(self, '_running', False))
            except Exception:
                self._running = False

    def _on_closing(self):
        """Clean shutdown: cancel scheduled after callbacks and stop loops before destroying the window."""
        # hide any tooltip immediately to avoid orphaned Toplevels
        self._hide_material_tooltip()
        # stop loops
        self.animating = False
        self.temp_playing = False

        # cancel scheduled after callbacks if present
        pid = getattr(self, '_preview_after_id', None)
        if pid is not None:
            self.after_cancel(pid)
            self._preview_after_id = None
        pid = getattr(self, '_poll_after_id', None)
        if pid is not None:
            self.after_cancel(pid)
            self._poll_after_id = None
        pid = getattr(self, '_anim_after_id', None)
        if pid is not None:
            self.after_cancel(pid)
            self._anim_after_id = None
        pid = getattr(self, '_temp_after_id', None)
        if pid is not None:
            self.after_cancel(pid)
            self._temp_after_id = None

        # attempt to cancel any remaining Tk 'after' callbacks (including CTk internals)
        try:
            info = self.tk.call('after', 'info')
        except Exception:
            info = None
        if info:
            # info can be a tuple of ids or a single string
            ids = []
            if isinstance(info, (list, tuple)):
                ids = list(info)
            else:
                # string with whitespace-separated ids
                try:
                    ids = str(info).split()
                except Exception:
                    ids = [info]
            for aid in ids:
                try:
                    self.after_cancel(aid)
                except Exception:
                    # ignore failures on cancelling internal callbacks
                    pass

        # attempt to destroy the window (ends mainloop)
        try:
            self.destroy()
        except Exception:
            self.quit()

    def _show_results(self, outA, outB: Optional[dict], mA: Optional[object] = None, mB: Optional[object] = None):
        # split plotting into dedicated tab plotters for clarity
        try:
            self._plot_results_tab(outA, outB, mA, mB)
        except Exception:
            # keep the UI responsive on exceptions
            logging.getLogger(__name__).exception('Failed to plot Results tab')

        try:
            self._plot_term_by_term(outA, outB, mA, mB)
        except Exception:
            logging.getLogger(__name__).exception('Failed to plot Term-by-term tab')

        # update Temp & SEB panel (time index 0 default)
        try:
            self.update_tempseb(0)
        except Exception:
            pass

    def _plot_results_tab(self, outA, outB: Optional[dict], mA: Optional[object] = None, mB: Optional[object] = None):
        """Populate the Results tab (2x1 or 2x2 layout)."""
        t = outA['t'] / hour
        try:
            try:
                self.fig_res.clf()
            except Exception:
                pass
            if outB is None:
                self.axs_res = self.fig_res.subplots(2, 1, squeeze=False)
            else:
                self.axs_res = self.fig_res.subplots(2, 2, squeeze=False)
        except Exception:
            pass

        def plot_energy(ax, out, mat_obj):
            ax.clear()
            ax.plot(t, out['Kstar'], color='orange', label='K*')
            ax.plot(t, out['Lstar'], color='magenta', label='L*')
            ax.plot(t, out['H'], color='green', label='H')
            if self._mat_allows_evap(mat_obj):
                ax.plot(t, out['E'], color='blue', label='E')
            else:
                ax.plot(t, np.zeros_like(out['t']), color='blue', label='E')
            ax.plot(t, out['G'], color='saddlebrown', label='G')
            ax.set_ylabel('Flux (W/m2)')
            ax.set_xlabel('Time (h)')
            ax.legend(loc='upper right', fontsize='small')

        def _mat_title(mat_obj, fallback_name: str):
            try:
                if mat_obj is None:
                    return fallback_name
                if isinstance(mat_obj, Mapping):
                    return str(mat_obj.get('name', fallback_name))
                return str(getattr(mat_obj, 'name', fallback_name))
            except Exception:
                return fallback_name

        def plot_ts(ax, out, title=None):
            ax.clear()
            # plot surface temperature (Ts)
            ax.plot(t, out['Ta'] - 273.15, color='k', label='Ta')
            ax.plot(t, out['Ts'] - 273.15, color='r', label='Ts')
            ax.set_ylabel('T (°C)')
            ax.set_xlabel('Time (h)')
            if title:
                ax.set_title(title)
            ax.legend(loc='upper right', fontsize='small')

        try:
            axs = self.axs_res
        except Exception:
            return

        if outB is None:
            for j in (0, 1):
                try:
                    axs[j][1].clear(); axs[j][1].set_visible(False)
                except Exception:
                    pass
            matA_name = _mat_title(mA, self.matA.get() if getattr(self, 'matA', None) is not None else 'Material A')
            plot_ts(axs[0][0], outA, title=matA_name)
            plot_energy(axs[1][0], outA, mA)
        else:
            for j in (0, 1):
                try:
                    axs[j][1].set_visible(True)
                except Exception:
                    pass
            matA_name = _mat_title(mA, self.matA.get() if getattr(self, 'matA', None) is not None else 'Material A')
            matB_name = _mat_title(mB, self.matB.get() if getattr(self, 'matB', None) is not None else 'Material B')
            plot_ts(axs[0][0], outA, title=matA_name)
            plot_ts(axs[0][1], outB, title=matB_name)
            plot_energy(axs[1][0], outA, mA)
            plot_energy(axs[1][1], outB, mB)

            tsA = np.asarray(outA.get('Ts', [])) - 273.15
            tsB = np.asarray(outB.get('Ts', [])) - 273.15
            if tsA.size and tsB.size:
                ymin = float(np.nanmin([np.nanmin(tsA), np.nanmin(tsB)]))
                ymax = float(np.nanmax([np.nanmax(tsA), np.nanmax(tsB)]))
                if ymax == ymin:
                    pad = 0.5
                else:
                    pad = 0.05 * (ymax - ymin)
                for a in (axs[0][0], axs[0][1]):
                    a.set_ylim(ymin - pad, ymax + pad)

            def _energy_bounds(o, mat_obj):
                tarr = o['t']
                if len(tarr) == 0:
                    return None
                K = np.asarray(o.get('Kstar', np.zeros_like(tarr)))
                L = np.asarray(o.get('Lstar', np.zeros_like(tarr)))
                H = np.asarray(o.get('H', np.zeros_like(tarr)))
                E = np.asarray(o.get('E', np.zeros_like(tarr)))
                if not self._mat_allows_evap(mat_obj):
                    E = np.zeros_like(E)
                G = np.asarray(o.get('G', np.zeros_like(tarr)))
                allv = np.concatenate([K, L, H, E, G]) if any(v.size for v in (K, L, H, E, G)) else np.array([0.0])
                return float(np.nanmin(allv)), float(np.nanmax(allv))

            bA = _energy_bounds(outA, mA)
            bB = _energy_bounds(outB, mB)
            if bA is not None and bB is not None:
                ymin = min(bA[0], bB[0])
                ymax = max(bA[1], bB[1])
                if ymax == ymin:
                    pad = 1.0
                else:
                    pad = 0.05 * (ymax - ymin)
                for a in (axs[1][0], axs[1][1]):
                    a.set_ylim(ymin - pad, ymax + pad)
        try:
            self.canvas_res.draw()
        except Exception:
            pass

    def _plot_term_by_term(self, outA, outB: Optional[dict], mA: Optional[object] = None, mB: Optional[object] = None):
        """Populate the Term-by-term (Temp & SEB) tab copy (3x2 grid)."""
        axs_tr = getattr(self, 'axs_temp_res', None)
        can_tr = getattr(self, 'canvas_temp_res', None)
        if axs_tr is None:
            return
        try:
            styleA = '-'
            styleB = '--'
            color_ta = 'black'
            color_kdown = 'tab:orange'
            color_kup = 'magenta'
            color_knet = 'black'
            color_ldown = 'tab:orange'
            color_lup = 'magenta'
            color_lnet = 'black'
            color_h = 'black'
            color_e = 'black'
            color_g = 'black'

            # TS subplot
            ax_ta = axs_tr[0][0]
            ax_ta.clear()
            # use time in hours for plotting
            tA = outA['t'] / hour
            tB = outB['t'] / hour if outB is not None else None
            # plot surface temperature (Ts) in °C
            ax_ta.plot(tA, outA['Ta'] - 273.15, color=color_kdown, linestyle=styleA, label='Ta')
            ax_ta.plot(tA, outA['Ts'] - 273.15, color=color_ta, linestyle=styleA, label='Ts A')
            if outB is not None:
                ax_ta.plot(tB, outB['Ts'] - 273.15, color=color_ta, linestyle=styleB, label='Ts B')
            ax_ta.set_ylabel('Ts (°C)')
            ax_ta.set_xlabel('Time (h)')
            ax_ta.grid(True, linestyle=':', alpha=0.4)
            ax_ta.legend(loc='upper left', fontsize='small')

            # K components subplot
            ax_k = axs_tr[0][1]
            ax_k.clear()
            ax_k.plot(tA, outA['Kstar'], color=color_knet, linestyle=styleA, label='K*')
            ax_k.plot(tA, outA['Kdown'], color=color_kdown, linestyle=styleA, label='Kdown')
            ax_k.plot(tA, -outA['Kup'], color=color_kup, linestyle=styleA, label='-Kup')
            if outB is not None:
                ax_k.plot(tB, outB['Kdown'], color=color_knet, linestyle=styleB)
                ax_k.plot(tB, outB['Kdown'], color=color_kdown, linestyle=styleB)
                ax_k.plot(tB, -outB['Kup'], color=color_kup, linestyle=styleB)
            ax_k.set_ylabel('K (W/m2)')
            ax_k.set_xlabel('Time (h)')
            ax_k.grid(True, linestyle=':', alpha=0.4)
            ax_k.legend(loc='upper left', fontsize='small')

            # L components subplot
            ax_l = axs_tr[1][0]
            ax_l.clear()
            ax_l.plot(tA, outA['Lstar'], color=color_lnet, linestyle=styleA, label='L*')
            ax_l.plot(tA, outA['Ldown'], color=color_ldown, linestyle=styleA, label='Ldown')
            ax_l.plot(tA, -outA['Lup'], color=color_lup, linestyle=styleA, label='-Lup')
            if outB is not None:
                ax_l.plot(tB, outB['Lstar'], color=color_lnet, linestyle=styleB)
                ax_l.plot(tB, outB['Ldown'], color=color_ldown, linestyle=styleB)
                ax_l.plot(tB, -outB['Lup'], color=color_lup, linestyle=styleB)
            ax_l.set_ylabel('L (W/m2)')
            ax_l.set_xlabel('Time (h)')
            ax_l.grid(True, linestyle=':', alpha=0.4)
            ax_l.legend(loc='upper left', fontsize='small')

            # H subplot
            ax_h = axs_tr[1][1]
            ax_h.clear()
            ax_h.plot(tA, outA['H'], color=color_h, linestyle=styleA)
            if outB is not None:
                ax_h.plot(tB, outB['H'], color=color_h, linestyle=styleB)
            ax_h.set_ylabel('H (W/m2)')
            ax_h.set_xlabel('Time (h)')
            ax_h.grid(True, linestyle=':', alpha=0.4)

            # E subplot
            ax_e = axs_tr[2][0]
            ax_e.clear()
            E_A = outA['E'] if 'E' in outA else np.zeros_like(outA['t'])
            if not self._mat_allows_evap(mA):
                E_A = np.zeros_like(E_A)
            ax_e.plot(tA, E_A, color=color_e, linestyle=styleA)
            if outB is not None:
                E_B = outB['E'] if 'E' in outB else np.zeros_like(outB['t'])
                if not self._mat_allows_evap(mB):
                    E_B = np.zeros_like(E_B)
                ax_e.plot(tB, E_B, color=color_e, linestyle=styleB)
            ax_e.set_ylabel('E (W/m2)')
            ax_e.set_xlabel('Time (h)')
            ax_e.grid(True, linestyle=':', alpha=0.4)

            # G subplot
            ax_g = axs_tr[2][1]
            ax_g.clear()
            ax_g.plot(tA, outA['G'], color=color_g, linestyle=styleA)
            if outB is not None:
                ax_g.plot(tB, outB['G'], color=color_g, linestyle=styleB)
            ax_g.set_ylabel('G (W/m2)')
            ax_g.set_xlabel('Time (h)')
            ax_g.grid(True, linestyle=':', alpha=0.4)

        except Exception:
            tb = traceback.format_exc()

        try:
            if can_tr is not None:
                try:
                    can_tr.draw()
                except Exception:
                    try:
                        cav = getattr(self, 'canvas_temp_res', None)
                        if cav is not None:
                            cav.draw()
                    except Exception:
                        pass
        except Exception:
            pass

if __name__ == '__main__':
    app = App()
    app.mainloop()
