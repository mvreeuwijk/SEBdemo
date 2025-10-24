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
from typing import Optional, Any, Mapping, cast

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import customtkinter as ctk

from model import load_material, run_simulation, diurnal_forcing


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
    # centralized default parameters for the UI and model
    hour = 3600.0

    DEFAULTS = {
        'thickA': 1.0,
        'thickB': 1.0,
        'compare': False,
        'Sb': 1000.0,
        'trise_hr': 7.0,
        'tset_hr': 21.0,
        'Ldown': 350.0,
        'hcoef': 10.0,
        'Ta_mean_C': 20.0,
        'Ta_amp_C': 5.0,
        'beta': 0.5,
        'speed': 1.0,
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
        width, height = 800, 560
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

        # references to tab frames
        self.tab_inputs = self.tabview.tab('Inputs')
        self.tab_animation = self.tabview.tab('Animation')
        self.tab_results = self.tabview.tab('Results')
        self.tab_tempseb = self.tabview.tab('Term by term')

        # build tab content
        self._build_inputs_tab()
        self._build_results_tab()
        self._build_tempseb_tab()
        self._build_animation_tab()

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
        try:
            self.protocol('WM_DELETE_WINDOW', self._on_closing)
        except Exception:
            pass

    # --- builders for each tab to keep __init__ concise ---
    def _build_inputs_tab(self):
        """Create widgets for the Inputs tab (materials, parameters, previews)."""
        keys = load_material_keys()
        frm = ctk.CTkFrame(self.tab_inputs)
        frm.pack(fill='both', expand=True, padx=8, pady=8)
        # tune grid column weights so the optionmenus and entries fit the window
        try:
            frm.grid_columnconfigure(0, weight=1, minsize=120, uniform='param')
            frm.grid_columnconfigure(1, weight=1, minsize=120, uniform='param')
            frm.grid_columnconfigure(2, weight=1, minsize=120, uniform='param')
        except Exception:
            pass

        # capture the inputs frame background color (used to color Matplotlib figs)
        try:
            self._input_frame_bg = frm.cget('fg_color')
        except Exception:
            try:
                self._input_frame_bg = frm.cget('bg')
            except Exception:
                self._input_frame_bg = None

        # Top row: Material selectors and compare checkbox above Material B
        frameA = ctk.CTkFrame(frm, fg_color='transparent')
        frameA.grid(row=1, column=0, sticky='nsew', padx=6, pady=(6, 2))
        try:
            frameA.grid_columnconfigure(0, weight=0)
            frameA.grid_columnconfigure(1, weight=1)
        except Exception:
            pass
        ctk.CTkLabel(frameA, text='Material A:').grid(row=0, column=0, sticky='w')
        self.matA = ctk.StringVar(value=keys[0] if keys else 'concrete')
        self.opt_matA = ctk.CTkOptionMenu(frameA, values=keys, variable=self.matA, width=150)
        self.opt_matA.grid(row=0, column=1, sticky='e', padx=(6, 0))
        # tooltip support for material A: show properties on hover
        try:
            self._tooltip_after_id = None
            self._tooltip_win = None
            self.opt_matA.bind('<Enter>', lambda e, w=self.opt_matA: self._schedule_show_material_tooltip(w, self.matA))
            self.opt_matA.bind('<Leave>', lambda e: self._hide_material_tooltip())
        except Exception:
            pass
        ctk.CTkLabel(frameA, text='Thickness A (m):').grid(row=1, column=0, sticky='w', pady=(6, 0))
        self.thickA = ctk.DoubleVar(value=self.DEFAULTS['thickA'])
        self.entry_thickA = ctk.CTkEntry(frameA, textvariable=self.thickA, width=120, justify='right')
        self.entry_thickA.grid(row=1, column=1, sticky='e', padx=(6, 0), pady=(6, 0))

        # Compare control in third column above Material B
        self.compare_var = ctk.BooleanVar(value=self.DEFAULTS['compare'])
        cmp_frame = ctk.CTkFrame(frm, fg_color='transparent')
        # place the compare control inside the third column (column=2).
        # The label will be left-aligned and the checkbox right-aligned within this column.
        cmp_frame.grid(row=0, column=2, sticky='nsew', padx=(6, 0), pady=(6, 0))
        try:
            # give the right-hand column weight so the checkbox can expand
            # and be flush with the right edge of the window
            cmp_frame.grid_columnconfigure(0, weight=1)
            cmp_frame.grid_columnconfigure(1, weight=1)
        except Exception:
            pass
        ctk.CTkLabel(cmp_frame, text='Compare with second material:').grid(row=0, column=0, sticky='w')
        self.cb_compare = ctk.CTkCheckBox(cmp_frame, text='', variable=self.compare_var, command=self.toggle_compare)
    # remove left padding and allow the checkbox column to expand so the
    # control sits flush with the right edge of the window
        self.cb_compare.grid(row=0, column=1, sticky='e', padx=(0, 0))

        # Frame for Material B (aligned with A)
        frameB = ctk.CTkFrame(frm, fg_color='transparent')
        frameB.grid(row=1, column=2, sticky='nsew', padx=6, pady=(6, 2))
        try:
            frameB.grid_columnconfigure(0, weight=0)
            frameB.grid_columnconfigure(1, weight=1)
        except Exception:
            pass
        ctk.CTkLabel(frameB, text='Material B:').grid(row=0, column=0, sticky='w')
        self.matB = ctk.StringVar(value=keys[0] if len(keys) > 1 else keys[0])
        self.opt_matB = ctk.CTkOptionMenu(frameB, values=keys, variable=self.matB, width=150)
        self.opt_matB.grid(row=0, column=1, sticky='e', padx=(6, 0))
        # tooltip support for material B
        try:
            self.opt_matB.bind('<Enter>', lambda e, w=self.opt_matB: self._schedule_show_material_tooltip(w, self.matB))
            self.opt_matB.bind('<Leave>', lambda e: self._hide_material_tooltip())
        except Exception:
            pass
        self.lbl_thickB = ctk.CTkLabel(frameB, text='Thickness B (m):')
        self.lbl_thickB.grid(row=1, column=0, sticky='w', pady=(6, 0))
        # remember default color for toggling
        try:
            self._lbl_thickB_color_default = self.lbl_thickB.cget('text_color')
        except Exception:
            self._lbl_thickB_color_default = None
        self.thickB = ctk.DoubleVar(value=self.DEFAULTS['thickB'])
        self.entry_thickB = ctk.CTkEntry(frameB, textvariable=self.thickB, width=120, justify='right')
        self.entry_thickB.grid(row=1, column=1, sticky='e', padx=(6, 0), pady=(6, 0))

        sep = ctk.CTkFrame(frm, height=2, fg_color='#7f7f7f')
        sep.grid(row=2, column=0, columnspan=3, sticky='ew', pady=(6, 8))

        # runtime flags
        self._running = False
        self._closed = False
        # storage for validated parameters (filled by validators)
        self._params = {
            'thickness_A': float(self.DEFAULTS['thickA']),
            'thickness_B': float(self.DEFAULTS['thickB']),
            'Sb': float(self.DEFAULTS['Sb']),
            'trise': int(self.DEFAULTS['trise_hr'] * self.hour),
            'tset': int(self.DEFAULTS['tset_hr'] * self.hour),
            'Ldown': float(self.DEFAULTS['Ldown']),
            'h': float(self.DEFAULTS['hcoef']),
            'Ta_mean': float(self.DEFAULTS['Ta_mean_C']) + 273.15,
            'Ta_amp': float(self.DEFAULTS['Ta_amp_C']),
            'beta': float(self.DEFAULTS['beta']),
        }

        # Parameters arranged in three vertical columns under the materials
        self.param_col0 = ctk.CTkFrame(frm)
        self.param_col0.grid(row=3, column=0, sticky='nsew', padx=6, pady=4)
        self.param_col1 = ctk.CTkFrame(frm)
        self.param_col1.grid(row=3, column=1, sticky='nsew', padx=6, pady=4)
        self.param_col2 = ctk.CTkFrame(frm)
        self.param_col2.grid(row=3, column=2, sticky='nsew', padx=6, pady=4)
        try:
            self.param_col0.grid_columnconfigure(0, weight=1)
            self.param_col0.grid_columnconfigure(1, weight=1)
            self.param_col1.grid_columnconfigure(0, weight=1)
            self.param_col1.grid_columnconfigure(1, weight=1)
            self.param_col2.grid_columnconfigure(0, weight=1)
            self.param_col2.grid_columnconfigure(1, weight=1)
        except Exception:
            pass

        try:
            frm.grid_rowconfigure(6, weight=1)
            frm.grid_rowconfigure(7, weight=1)
        except Exception:
            pass

        # Column 0
        ctk.CTkLabel(self.param_col0, text='Peak shortwave Sb (W/m2):').grid(row=0, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.Sb = ctk.DoubleVar(value=self.DEFAULTS['Sb'])
        self.entry_Sb = ctk.CTkEntry(self.param_col0, textvariable=self.Sb, width=80, justify='right')
        self.entry_Sb.grid(row=0, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col0, text='Sunrise trise (h):').grid(row=1, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.trise_hr = ctk.DoubleVar(value=self.DEFAULTS['trise_hr'])
        self.entry_trise = ctk.CTkEntry(self.param_col0, textvariable=self.trise_hr, width=80, justify='right')
        self.entry_trise.grid(row=1, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col0, text='Sunset tset (h):').grid(row=2, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.tset_hr = ctk.DoubleVar(value=self.DEFAULTS['tset_hr'])
        self.entry_tset = ctk.CTkEntry(self.param_col0, textvariable=self.tset_hr, width=80, justify='right')
        self.entry_tset.grid(row=2, column=1, sticky='e', pady=(0,4))

        # Column 1
        ctk.CTkLabel(self.param_col1, text='Incoming Ldown (W/m2):').grid(row=0, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.Ldown = ctk.DoubleVar(value=self.DEFAULTS['Ldown'])
        self.entry_Ldown = ctk.CTkEntry(self.param_col1, textvariable=self.Ldown, width=80, justify='right')
        self.entry_Ldown.grid(row=0, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col1, text='Heat transfer h (W/m2K):').grid(row=1, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.hcoef = ctk.DoubleVar(value=self.DEFAULTS['hcoef'])
        self.entry_hcoef = ctk.CTkEntry(self.param_col1, textvariable=self.hcoef, width=80, justify='right')
        self.entry_hcoef.grid(row=1, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col1, text='Air mean temp (°C):').grid(row=2, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.Ta_mean_C = ctk.DoubleVar(value=self.DEFAULTS['Ta_mean_C'])
        self.entry_Ta_mean = ctk.CTkEntry(self.param_col1, textvariable=self.Ta_mean_C, width=80, justify='right')
        self.entry_Ta_mean.grid(row=2, column=1, sticky='e', pady=(0,4))

        # Column 2
        ctk.CTkLabel(self.param_col2, text='Air amp (°C):').grid(row=0, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.Ta_amp_C = ctk.DoubleVar(value=self.DEFAULTS['Ta_amp_C'])
        self.entry_Ta_amp = ctk.CTkEntry(self.param_col2, textvariable=self.Ta_amp_C, width=80, justify='right')
        self.entry_Ta_amp.grid(row=0, column=1, sticky='e', pady=(0,4))

        ctk.CTkLabel(self.param_col2, text='Bowen beta (optional):').grid(row=1, column=0, sticky='w', padx=(0,6), pady=(0,4))
        self.beta_var = ctk.DoubleVar(value=self.DEFAULTS['beta'])
        self.entry_beta = ctk.CTkEntry(self.param_col2, textvariable=self.beta_var, width=80, justify='right')
        self.entry_beta.grid(row=1, column=1, sticky='e', pady=(0,4))

        # Preview figures
        self.fig_Ta = Figure(figsize=(8, 2))
        self.canvas_Ta = FigureCanvasTkAgg(self.fig_Ta, master=frm)
        self.canvas_Ta.get_tk_widget().grid(row=6, column=0, columnspan=3, padx=8, pady=(8, 4), sticky='nsew')
        self.ax_Ta = self.fig_Ta.subplots(1, 1)
        try:
            if getattr(self, '_input_frame_bg', None):
                mcol = self._mpl_color_from_ctk(self._input_frame_bg)
                if mcol is not None:
                    try:
                        self.fig_Ta.patch.set_facecolor(mcol)
                        for a in self.fig_Ta.axes:
                            a.set_facecolor(mcol)
                    except Exception:
                        pass
        except Exception:
            pass

        self.fig_K = Figure(figsize=(8, 2))
        self.canvas_K = FigureCanvasTkAgg(self.fig_K, master=frm)
        self.canvas_K.get_tk_widget().grid(row=7, column=0, columnspan=3, padx=8, pady=(4, 8), sticky='nsew')
        self.ax_K = self.fig_K.subplots(1, 1)
        try:
            if getattr(self, '_input_frame_bg', None):
                mcol = self._mpl_color_from_ctk(self._input_frame_bg)
                if mcol is not None:
                    try:
                        self.fig_K.patch.set_facecolor(mcol)
                        for a in self.fig_K.axes:
                            a.set_facecolor(mcol)
                    except Exception:
                        pass
        except Exception:
            pass

        # initial Inputs-tab state: enable/disable compare controls and
        # install preview traces so changing inputs updates the preview.
        try:
            # ensure preview scheduling handle exists
            self._preview_after_id = None
        except Exception:
            self._preview_after_id = None
        try:
            # toggle compare so Material B controls are set correctly
            self.toggle_compare()
        except Exception:
            pass
        try:
            # draw an initial preview (best-effort)
            self.update_preview()
        except Exception:
            pass
        try:
            # install traces on the input variables to debounce preview updates
            self._install_preview_traces()
        except Exception:
            pass

        # bind focus-out validation for inputs so correctness is checked when the
        # user leaves the box (and before we run the simulation).
        try:
            self.entry_Sb.bind('<FocusOut>', lambda e: self._validate_and_store('Sb'))
            self.entry_trise.bind('<FocusOut>', lambda e: self._validate_and_store('trise'))
            self.entry_tset.bind('<FocusOut>', lambda e: self._validate_and_store('tset'))
            self.entry_Ldown.bind('<FocusOut>', lambda e: self._validate_and_store('Ldown'))
            self.entry_hcoef.bind('<FocusOut>', lambda e: self._validate_and_store('h'))
            self.entry_Ta_mean.bind('<FocusOut>', lambda e: self._validate_and_store('Ta_mean'))
            self.entry_Ta_amp.bind('<FocusOut>', lambda e: self._validate_and_store('Ta_amp'))
            self.entry_beta.bind('<FocusOut>', lambda e: self._validate_and_store('beta'))
            # thickness entries already exist; bind them too
            try:
                self.entry_thickA.bind('<FocusOut>', lambda e: self._validate_and_store('thickness_A'))
            except Exception:
                pass
            try:
                self.entry_thickB.bind('<FocusOut>', lambda e: self._validate_and_store('thickness_B'))
            except Exception:
                pass
        except Exception:
            pass

        # run an initial validation to populate self._params
        try:
            self._validate_and_store()
        except Exception:
            pass

    def _build_results_tab(self):
        """Create the Results tab figure and axes."""
        self.fig_res = Figure(figsize=(8, 9))
        try:
            if getattr(self, '_input_frame_bg', None):
                mcol = self._mpl_color_from_ctk(self._input_frame_bg)
                if mcol is not None:
                    try:
                        self.fig_res.patch.set_facecolor(mcol)
                    except Exception:
                        pass
        except Exception:
            pass
        self.canvas_res = FigureCanvasTkAgg(self.fig_res, master=self.tab_results)
        self.canvas_res.get_tk_widget().pack(fill='both', expand=True)
        self.axs_res = self.fig_res.subplots(6, 1)

    def _build_tempseb_tab(self):
        """Create the Temp & SEB (term-by-term) tab contents."""
        self.fig_temp = Figure(figsize=(6, 3))
        self.canvas_temp = FigureCanvasTkAgg(self.fig_temp, master=self.tab_tempseb)
        self.canvas_temp.get_tk_widget().pack(fill='both', expand=True)
        self.ax_temp = self.fig_temp.subplots(1, 1)
        try:
            if getattr(self, '_input_frame_bg', None):
                mcol = self._mpl_color_from_ctk(self._input_frame_bg)
                if mcol is not None:
                    try:
                        self.fig_temp.patch.set_facecolor(mcol)
                        for a in self.fig_temp.axes:
                            a.set_facecolor(mcol)
                    except Exception:
                        pass
        except Exception:
            pass

        # time slider for Temp & SEB (inactive until a run populates anim_data)
        self.temp_time_var = ctk.IntVar(value=0)
        self.temp_slider = ctk.CTkSlider(self.tab_tempseb, from_=0, to=1, number_of_steps=1, command=lambda v: self.update_tempseb(int(float(v))))
        self.temp_slider.pack(fill='x', padx=10, pady=6)
        # playback controls for Temp & SEB
        self.temp_playing = False
        self.temp_play_btn = ctk.CTkButton(self.tab_tempseb, text='Play', command=self.toggle_temp_playback, state='disabled')
        self.temp_play_btn.pack(padx=10, pady=(0, 6))

    def _build_animation_tab(self):
        """Create the Animation tab figure and controls."""
        self.fig_anim = Figure(figsize=(8, 3))
        try:
            if getattr(self, '_input_frame_bg', None):
                mcol = self._mpl_color_from_ctk(self._input_frame_bg)
                if mcol is not None:
                    try:
                        self.fig_anim.patch.set_facecolor(mcol)
                    except Exception:
                        pass
        except Exception:
            pass
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=self.tab_animation)
        self.canvas_anim.get_tk_widget().pack(side='left', fill='both', expand=True)

        self.anim_ctrl_frame = ctk.CTkFrame(self.tab_animation)
        self.anim_ctrl_frame.pack(side='left', fill='y', padx=8, pady=6)
        self.time_label = ctk.CTkLabel(self.anim_ctrl_frame, text='Time: -- h')
        self.time_label.grid(row=0, column=0)
        self.speed_var = ctk.DoubleVar(value=1.0)
        ctk.CTkLabel(self.anim_ctrl_frame, text='Speed:').grid(row=1, column=0, sticky='w')
        ctk.CTkEntry(self.anim_ctrl_frame, textvariable=self.speed_var, width=80).grid(row=2, column=0)
        self.btn_start = ctk.CTkButton(self.anim_ctrl_frame, text='Start', command=self.toggle_animation, state='disabled')
        self.btn_start.grid(row=3, column=0, pady=(8, 0))

        # animation state placeholders
        self.animating = False
        self.anim_idx = 0
        self.anim_data = None
        self.axes_anim = tuple()

    def _poll_tab(self):
        try:
            cur = self.tabview.get()
        except Exception:
            # fallback: no action
            cur = self._last_tab
        # guard: don't run if the app is closed or widget destroyed
        try:
            if getattr(self, '_closed', False):
                return
            if hasattr(self, 'winfo_exists') and not self.winfo_exists():
                return
        except Exception:
            pass
        if cur != getattr(self, '_last_tab', None):
            # if we just left the Inputs tab, start a simulation
            if getattr(self, '_last_tab', None) == 'Inputs':
                # start background run if not already running
                if not getattr(self, '_running', False):
                    try:
                        self._on_run()
                    except Exception:
                        pass
        self._last_tab = cur
        try:
            self._poll_after_id = self.after(300, self._poll_tab)
        except Exception:
            self._poll_after_id = None

    def update_preview(self):
        """Update the small preview plot on the Inputs tab showing Kdown and Ta."""
        # guard: don't run if window is closed/destroyed
        try:
            if getattr(self, '_closed', False):
                return
            if hasattr(self, 'winfo_exists') and not self.winfo_exists():
                return
        except Exception:
            pass
        # Read inputs (use sensible defaults if parsing fails)
        try:
            Sb = float(self.Sb.get())
        except Exception:
            Sb = self.DEFAULTS['Sb']
        try:
            trise = int(float(self.trise_hr.get()) * self.hour)
        except Exception:
            trise = int(self.DEFAULTS['trise_hr'] * self.hour)
        try:
            tset = int(float(self.tset_hr.get()) * self.hour)
        except Exception:
            tset = int(self.DEFAULTS['tset_hr'] * self.hour)
        try:
            Ta_mean = float(self.Ta_mean_C.get()) + 273.15
        except Exception:
            Ta_mean = 293.15
        try:
            Ta_amp = float(self.Ta_amp_C.get())
        except Exception:
            Ta_amp = self.DEFAULTS['Ta_amp_C']
        try:
            Ldown = float(self.Ldown.get())
        except Exception:
            Ldown = self.DEFAULTS['Ldown']

        # plot over a 24 h window with a 10-min timestep
        tmax = 24 * self.hour
        dt = 600.0
        t = np.arange(0, tmax + dt, dt)
        TaK, S0 = diurnal_forcing(t, Ta_mean=Ta_mean, Ta_amp=Ta_amp, Sb=Sb, trise=trise, tset=tset)

        TaC = TaK - 273.15

        # Ta figure
        try:
            self.ax_Ta.clear()
            self.ax_Ta.plot(t / self.hour, TaC, color='tab:blue', label='Ta (°C)')
            self.ax_Ta.set_xlabel('Time (h)')
            self.ax_Ta.set_ylabel('Air temp (°C)')
            self.ax_Ta.set_xlim(0, 24)
            # apply background and spine styling to match app
            try:
                bg = getattr(self, '_input_frame_bg', None)
                if bg is not None:
                    mcol = self._mpl_color_from_ctk(bg)
                    if mcol is not None:
                        self.fig_Ta.patch.set_facecolor(mcol)
                        self.ax_Ta.set_facecolor(mcol)
                        # compute perceived luminance for contrast
                        dark = False
                        try:
                            if isinstance(mcol, (list, tuple)) and len(mcol) >= 3:
                                lum = 0.299 * float(mcol[0]) + 0.587 * float(mcol[1]) + 0.114 * float(mcol[2])
                                dark = lum < 0.5
                        except Exception:
                            dark = False
                        spcol = 'white' if dark else 'black'
                        for s in self.ax_Ta.spines.values():
                            s.set_color(spcol)
                            s.set_linewidth(0.8)
                        # ticks and labels
                        self.ax_Ta.tick_params(colors=spcol)
                        self.ax_Ta.xaxis.label.set_color(spcol)
                        self.ax_Ta.yaxis.label.set_color(spcol)
            except Exception:
                pass
            self.ax_Ta.grid(True, linestyle=':', alpha=0.5)
            self.ax_Ta.legend(loc='upper right')
            # avoid tight_layout clipping the right spine
            try:
                self.fig_Ta.subplots_adjust(right=0.98)
            except Exception:
                pass
            try:
                # use tight_layout but reserve a small right margin so legends/spines are not clipped
                self.fig_Ta.tight_layout(rect=(0, 0, 0.98, 1))
            except Exception:
                pass
            try:
                self.canvas_Ta.draw()
            except Exception:
                pass
        except Exception:
            pass

        # Kdown + Ldown figure
        try:
            self.ax_K.clear()
            self.ax_K.plot(t / self.hour, S0, color='tab:orange', label='Kdown (W/m2)')
            self.ax_K.plot(t / self.hour, np.full_like(t, Ldown), color='tab:purple', linestyle='--', label='Ldown (W/m2)')
            self.ax_K.set_xlabel('Time (h)')
            self.ax_K.set_ylabel('Flux (W/m2)')
            self.ax_K.set_xlim(0, 24)
            try:
                bg = getattr(self, '_input_frame_bg', None)
                if bg is not None:
                    mcol = self._mpl_color_from_ctk(bg)
                    if mcol is not None:
                        self.fig_K.patch.set_facecolor(mcol)
                        self.ax_K.set_facecolor(mcol)
                        dark = False
                        try:
                            if isinstance(mcol, (list, tuple)) and len(mcol) >= 3:
                                lum = 0.299 * float(mcol[0]) + 0.587 * float(mcol[1]) + 0.114 * float(mcol[2])
                                dark = lum < 0.5
                        except Exception:
                            dark = False
                        spcol = 'white' if dark else 'black'
                        for s in self.ax_K.spines.values():
                            s.set_color(spcol)
                            s.set_linewidth(0.8)
                        self.ax_K.tick_params(colors=spcol)
                        self.ax_K.xaxis.label.set_color(spcol)
                        self.ax_K.yaxis.label.set_color(spcol)
            except Exception:
                pass
            self.ax_K.grid(True, linestyle=':', alpha=0.5)
            self.ax_K.legend(loc='upper right')
            try:
                self.fig_K.subplots_adjust(right=0.98)
            except Exception:
                pass
            try:
                # reserve right margin for legend
                self.fig_K.tight_layout(rect=(0, 0, 0.98, 1))
            except Exception:
                pass
            try:
                self.canvas_K.draw()
            except Exception:
                pass
        except Exception:
            pass

    # --- preview auto-update helpers ---
    def _schedule_preview_update(self, delay_ms: int = 200):
        """Debounced schedule for preview updates (cancels previous scheduled call)."""
        try:
            pid = getattr(self, '_preview_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self._preview_after_id = self.after(delay_ms, self.update_preview)
        except Exception:
            self._preview_after_id = None
            # immediate fallback
            try:
                self.update_preview()
            except Exception:
                pass

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
                try:
                    # fallback to older trace
                    v.trace('w', lambda *a, _=v: self._schedule_preview_update())
                except Exception:
                    pass

    def _validate_and_store(self, key: Optional[str] = None, event: Optional[object] = None):
        """Validate input fields on focus-out and store canonical values in self._params.

        This method is deliberately conservative: invalid values are replaced by
        sensible defaults from `DEFAULTS` and the corresponding tkinter variable
        is updated so the user sees the corrected value.
        """
        # helper to safely set a doublevar (and clamp/convert)
        def set_doublevar(var, val):
            try:
                var.set(val)
            except Exception:
                pass

        # helper to map validation keys to entry widgets and readable names
        key_map = {
            'Sb': (self.entry_Sb, 'Peak shortwave Sb (W/m2)'),
            'trise': (self.entry_trise, 'Sunrise trise (h)'),
            'tset': (self.entry_tset, 'Sunset tset (h)'),
            'Ldown': (self.entry_Ldown, 'Incoming Ldown (W/m2)'),
            'h': (self.entry_hcoef, 'Heat transfer h (W/m2K)'),
            'Ta_mean': (self.entry_Ta_mean, 'Air mean temp (°C)'),
            'Ta_amp': (self.entry_Ta_amp, 'Air amp (°C)'),
            'beta': (self.entry_beta, 'Bowen beta'),
            'thickness_A': (self.entry_thickA, 'Thickness A (m)'),
            'thickness_B': (self.entry_thickB, 'Thickness B (m)'),
        }

        def focus_and_select(entry_widget):
            try:
                entry_widget.focus_set()
            except Exception:
                pass
            try:
                # CTkEntry exposes selection methods via underlying tk widget
                entry_widget.select_range(0, 'end')
            except Exception:
                try:
                    w = getattr(entry_widget, 'entry', None) or getattr(entry_widget, 'master', None)
                    if w is not None and hasattr(w, 'select_range'):
                        w.select_range(0, 'end')
                except Exception:
                    pass

        def show_fix_message(entry_widget, label, msg):
            # show modal dialog and return focus to the offending widget
            try:
                messagebox.showerror('Invalid input', msg)
            except Exception:
                pass
            try:
                # restore focus to the widget after the dialog closes
                self.after(1, lambda: focus_and_select(entry_widget))
            except Exception:
                focus_and_select(entry_widget)

        # validation routines for each key. Return (ok, canonical_value)
        def validate_Sb():
            try:
                v = float(self.Sb.get())
                if v < 0:
                    raise ValueError
            except Exception:
                return False, f"Invalid Sb — must be a non-negative number."
            return True, float(v)

        def validate_trise():
            try:
                vh = float(self.trise_hr.get())
                if not (0.0 <= vh <= 24.0):
                    raise ValueError
                return True, int(vh * self.hour)
            except Exception:
                return False, f"Invalid sunrise time — enter hours between 0 and 24."

        def validate_tset():
            try:
                vh = float(self.tset_hr.get())
                if not (0.0 <= vh <= 24.0):
                    raise ValueError
                return True, int(vh * self.hour)
            except Exception:
                return False, f"Invalid sunset time — enter hours between 0 and 24."

        def validate_Ldown():
            try:
                v = float(self.Ldown.get())
                if v < 0:
                    raise ValueError
            except Exception:
                return False, f"Invalid Ldown — must be a non-negative number."
            return True, float(v)

        def validate_h():
            try:
                v = float(self.hcoef.get())
                if v < 0:
                    raise ValueError
            except Exception:
                return False, f"Invalid heat-transfer coefficient — must be non-negative."
            return True, float(v)

        def validate_Ta_mean():
            try:
                v_c = float(self.Ta_mean_C.get())
            except Exception:
                return False, f"Invalid air mean temperature — enter a numeric °C value."
            return True, float(v_c) + 273.15

        def validate_Ta_amp():
            try:
                v = float(self.Ta_amp_C.get())
                if v < 0:
                    raise ValueError
            except Exception:
                return False, f"Invalid air temperature amplitude — must be non-negative."
            return True, float(v)

        def validate_beta():
            try:
                v = float(self.beta_var.get())
            except Exception:
                return False, f"Invalid Bowen beta — enter a numeric value."
            return True, float(v)

        def validate_thickness_A():
            try:
                v = float(self.thickA.get())
                if v <= 0:
                    raise ValueError
            except Exception:
                return False, f"Invalid Thickness A — must be a positive number."
            return True, float(v)

        def validate_thickness_B():
            try:
                v = float(self.thickB.get())
                if v <= 0:
                    raise ValueError
            except Exception:
                return False, f"Invalid Thickness B — must be a positive number."
            return True, float(v)

        validators = {
            'Sb': validate_Sb,
            'trise': validate_trise,
            'tset': validate_tset,
            'Ldown': validate_Ldown,
            'h': validate_h,
            'Ta_mean': validate_Ta_mean,
            'Ta_amp': validate_Ta_amp,
            'beta': validate_beta,
            'thickness_A': validate_thickness_A,
            'thickness_B': validate_thickness_B,
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
                    try:
                        messagebox.showerror('Invalid input', msg)
                    except Exception:
                        pass
                return False
            # on success, store canonical value
            self._params[kk if kk not in ('thickness_A', 'thickness_B') else kk] = res
        return True

    # --- simple tooltip implementation (CTk-styled when possible, fallback to Tk)
    def _schedule_show_material_tooltip(self, widget, var, delay=300):
        """Schedule showing a tooltip for the material referenced by the tkinter variable `var`.

        We debounce using `after` so the tooltip only appears when the cursor lingers.
        """
        try:
            # if this is Material B's widget and compare is disabled, don't show
            try:
                if widget is getattr(self, 'opt_matB', None) and not bool(getattr(self, 'compare_var', ctk.BooleanVar(value=False)).get()):
                    return
            except Exception:
                pass
            # cancel any existing schedule
            pid = getattr(self, '_tooltip_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
            # schedule new show
            self._tooltip_after_id = self.after(delay, lambda: self._show_material_tooltip(widget, var))
        except Exception:
            # immediate fallback: show now
            try:
                self._show_material_tooltip(widget, var)
            except Exception:
                pass

    def _show_material_tooltip(self, widget, var):
        """Create and show a small tooltip window near `widget` with material properties.

        `var` is a tkinter variable whose value is the material key name; we load
        the material and display a few fields (k, rho, cp, albedo, emissivity, evaporation).
        """
        try:
            # ensure any scheduled show id is cleared
            try:
                self._tooltip_after_id = None
            except Exception:
                pass
            # destroy existing tooltip if present (assign to local var so
            # static checkers can reason about None-ness)
            try:
                tw = getattr(self, '_tooltip_win', None)
                if tw is not None:
                    try:
                        tw.destroy()
                    except Exception:
                        pass
                self._tooltip_win = None
            except Exception:
                pass

            # defensive: if this is Material B and compare is off, don't show
            try:
                if widget is getattr(self, 'opt_matB', None) and not bool(getattr(self, 'compare_var', ctk.BooleanVar(value=False)).get()):
                    return
            except Exception:
                pass

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
                            return fmt.format(v) if v is not None else '—'
                        except Exception:
                            return '—'

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
            try:
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
                    pass
                self._tooltip_win = tw
            except Exception:
                self._tooltip_win = None
        except Exception:
            pass

    def _hide_material_tooltip(self):
        """Hide/destroy any visible tooltip and cancel scheduled shows."""
        try:
            pid = getattr(self, '_tooltip_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
                self._tooltip_after_id = None
        except Exception:
            pass
        try:
            tw = getattr(self, '_tooltip_win', None)
            if tw is not None:
                try:
                    tw.destroy()
                except Exception:
                    pass
            self._tooltip_win = None
        except Exception:
            pass

    # helper: apply app background color to a matplotlib figure and its axes
    def _apply_fig_bg(self, fig):
        try:
            bg = getattr(self, '_input_frame_bg', None)
            if not bg:
                try:
                    bg = self.cget('bg')
                except Exception:
                    bg = None
            if not bg:
                return
            mcol = self._mpl_color_from_ctk(bg)
            fig.patch.set_facecolor(mcol)
            for ax in getattr(fig, 'axes', []):
                try:
                    ax.set_facecolor(mcol)
                except Exception:
                    pass
        except Exception:
            pass

    def _mpl_color_from_ctk(self, c):
        """Convert a CTk color (hex string or tuple) to an MPL-compatible color.

        CTk may return colors as HEX strings like '#rrggbb' or as tuples.
        Handle common cases and fall back to the input unchanged.
        """
        try:
            if c is None:
                return None
            # already acceptable (named color or hex)
            if isinstance(c, str):
                return c
            # tuple of floats in 0..1
            if isinstance(c, tuple) and all(isinstance(x, float) and 0.0 <= x <= 1.0 for x in c):
                return c
            # tuple of ints 0..255
            if isinstance(c, tuple) and all(isinstance(x, (int, float)) for x in c):
                vals = tuple(float(x) / 255.0 if x > 1.0 else float(x) for x in c)
                return vals
        except Exception:
            pass
        return c

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

    # status text stored internally (no visible widget)
    # any code that previously called _set_status now writes to _status_text

    # --- Compare UI handling ---
    def toggle_compare(self):
        enabled = bool(self.compare_var.get())
        # enable/disable material B controls
        state = 'normal' if enabled else 'disabled'
        try:
            self.opt_matB.configure(state=state)
            self.entry_thickB.configure(state=state)
        except Exception:
            pass

        # visually indicate Material B entry disabled when compare is off (leave label color unchanged)
        try:
            try:
                if not enabled:
                    try:
                        self.entry_thickB.configure(text_color='gray')
                    except Exception:
                        pass
                else:
                    try:
                        self.entry_thickB.configure(text_color=None)
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception:
            pass

        # reconfigure animation axes to show 1 or 2 panels
        try:
            self.configure_animation_axes(two_panels=enabled)
        except Exception:
            pass

        # create/remove Compare results tab (CTkTabview)
        try:
            if enabled and getattr(self, 'tab_compare', None) is None:
                # add a 'Compare' tab to the CTkTabview and create a figure there
                self.tabview.add('Compare')
                self.tab_compare = self.tabview.tab('Compare')
                # create compare figure and canvas
                try:
                    self.fig_comp = Figure(figsize=(8, 6))
                    self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=self.tab_compare)
                    self.canvas_comp.get_tk_widget().pack(fill='both', expand=True)
                    self.axs_comp = self.fig_comp.subplots(6, 1)
                except Exception:
                    self.fig_comp = None
                    self.canvas_comp = None
                    self.axs_comp = None
            elif not enabled and getattr(self, 'tab_compare', None) is not None:
                try:
                    # CTkTabview provides a remove method in newer versions
                    remove_fn = getattr(self.tabview, 'remove', None)
                    if callable(remove_fn):
                        remove_fn('Compare')
                    else:
                        try:
                            delattr(self, 'tab_compare')
                        except Exception:
                            pass
                except Exception:
                    try:
                        delattr(self, 'tab_compare')
                    except Exception:
                        pass
                self.tab_compare = None
                self.fig_comp = None
                self.canvas_comp = None
                self.axs_comp = None
        except Exception:
            pass

    # --- Animation axes management ---
    def configure_animation_axes(self, two_panels: bool = False):
        # destroy old figure and recreate based on two_panels
        try:
            self.canvas_anim.get_tk_widget().destroy()
        except Exception:
            pass
        if two_panels:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharey=False)
            self.axes_anim = (axes[0], axes[1])
        else:
            fig, axes = plt.subplots(1, 1, figsize=(8, 3), sharey=False)
            self.axes_anim = (axes,)
        self.fig_anim = fig
        self.canvas_anim = FigureCanvasTkAgg(self.fig_anim, master=self.tab_animation)
        self.canvas_anim.get_tk_widget().pack(side='left', fill='both', expand=True)
        # redraw static background
        try:
            self.draw_anim_static()
        except Exception:
            pass

    def draw_anim_static(self):
        # draw a simple background showing material depths
        # guard: avoid drawing if app is closed
        try:
            if getattr(self, '_closed', False):
                return
            if not getattr(self, 'axes_anim', None):
                return
            if hasattr(self, 'winfo_exists') and not self.winfo_exists():
                return
        except Exception:
            pass
        ax = self.axes_anim[0]
        ax.clear()
        try:
            thickA = float(self.thickA.get())
        except Exception:
            thickA = 0.2
        ax.set_ylim(-thickA - 0.1, 0.1)
        ax.set_xlim(0, 1)
        ax.invert_yaxis()
        ax.set_title('Material A (left)')
        # right axis label if present
        if len(self.axes_anim) > 1:
            ax2 = self.axes_anim[1]
            try:
                thickB = float(self.thickB.get())
            except Exception:
                thickB = 0.2
            ax2.clear()
            ax2.set_ylim(-thickB - 0.1, 0.1)
            ax2.set_xlim(0, 1)
            ax2.invert_yaxis()
            ax2.set_title('Material B (right)')
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
        try:
            self.btn_start.configure(text='Stop' if self.animating else 'Start')
        except Exception:
            pass
        if self.animating:
            # start loop
            self.animate_step()

    def animate_step(self):
        # guard: stop if app closed or animation flag cleared
        try:
            if getattr(self, '_closed', False):
                self.animating = False
                return
            if not self.animating:
                return
            if hasattr(self, 'winfo_exists') and not self.winfo_exists():
                self.animating = False
                return
        except Exception:
            pass
        anim_data = getattr(self, 'anim_data', None)
        if anim_data is None:
            # nothing to animate
            self.animating = False
            return
        outA = anim_data.get('A')
        matA = anim_data.get('A_mat')
        outB = anim_data.get('B') if self.compare_var.get() else None
        matB = anim_data.get('B_mat') if self.compare_var.get() else None
        times = outA['times']
        i = int(self.anim_idx % len(times))
        # draw left
        axL = self.axes_anim[0]
        axL.clear()
        try:
            axL.plot(outA['T_profiles'][i] - 273.15, outA['z'], '-r')
        except Exception:
            pass
        axL.invert_yaxis()
        # right if present
        if len(self.axes_anim) > 1 and outB is not None:
            axR = self.axes_anim[1]
            axR.clear()
            try:
                axR.plot(outB['T_profiles'][i] - 273.15, outB['z'], '-b')
            except Exception:
                pass
            axR.invert_yaxis()
        # draw simple SEB arrows near surface on each axis
        try:
            self.draw_seb_arrows(axL, outA, i, mat=matA)
        except Exception:
            pass
        if len(self.axes_anim) > 1 and outB is not None:
            try:
                self.draw_seb_arrows(self.axes_anim[1], outB, i, mat=matB)
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
        # a consistent location regardless of data limits. Direction:
        #  - Q* (net shortwave) : downward when incoming
        #  - L (longwave net)    : downward when net incoming
        #  - H (sensible)       : upward when positive (surface->air)
        #  - E (latent)         : upward when positive
        #  - G (ground)         : downward when positive into the ground
        try:
            Q = out['Qstar'][idx]; L = out['L'][idx]; H = out['H'][idx];
            # only include latent flux if material explicitly allows evaporation
            if self._mat_allows_evap(mat):
                # safe-get E (may be list or array)
                E = out.get('E', np.zeros_like(out.get('times', [0])))[idx]
            else:
                E = 0.0
            G = out['G'][idx]
            vals = [Q, L, H, E, G]
            maxv = max(1.0, max(abs(v) for v in vals))
            # vertical positions (axes fraction) for arrows
            y_positions = [0.85, 0.72, 0.59, 0.46, 0.33]
            labels = ['Q*', 'L*', 'H', 'E', 'G']
            colors = ['orange', 'magenta', 'green', 'blue', 'saddlebrown']
            for k, v in enumerate(vals):
                y = y_positions[k]
                frac = 0.08 * (abs(v) / maxv)
                if k in (0, 1, 4):
                    # draw arrow pointing downwards (incoming into surface)
                    start = (0.92, y)
                    end = (0.92, y - frac)
                    va = 'top'
                else:
                    # draw arrow pointing upwards
                    start = (0.92, y)
                    end = (0.92, y + frac)
                    va = 'bottom'
                ax.annotate('', xy=end, xytext=start, xycoords='axes fraction', textcoords='axes fraction', arrowprops=dict(arrowstyle='-|>', color=colors[k], lw=2))
                # label and value
                try:
                    txt = f"{labels[k]} {v:+.0f} W/m2" if k != 0 else f"{labels[k]} {v:+.0f} W/m2"
                    ax.text(0.80, y, txt, transform=ax.transAxes, ha='left', va=va, fontsize=9, color=colors[k])
                except Exception:
                    pass
        except Exception:
            pass

    def update_tempseb(self, idx: int):
        # draw temperature profile and SEB arrows at given time index (index into out['times'])
        anim_data = getattr(self, 'anim_data', None)
        if anim_data is None:
            return
        outA = anim_data.get('A')
        outB = anim_data.get('B') if self.compare_var.get() else None
        try:
            self.ax_temp.clear()
            self.ax_temp.plot(outA['T_profiles'][idx] - 273.15, outA['z'], '-r', label='A')
            if outB is not None:
                self.ax_temp.plot(outB['T_profiles'][idx] - 273.15, outB['z'], '-b', label='B')
            self.ax_temp.invert_yaxis()
            self.ax_temp.set_xlabel('Temperature (°C)')
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
        except Exception:
            pass

    # --- Temp & SEB playback ---
    def toggle_temp_playback(self):
        if not getattr(self, 'anim_data', None):
            messagebox.showerror('Error', 'Run a simulation first')
            return
        self.temp_playing = not self.temp_playing
        try:
            self.temp_play_btn.configure(text='Stop' if self.temp_playing else 'Play')
        except Exception:
            pass
        if self.temp_playing:
            self.temp_play_step()

    def temp_play_step(self):
        # guard: stop if app closed or playback flag cleared
        try:
            if getattr(self, '_closed', False):
                self.temp_playing = False
                return
            if not self.temp_playing:
                return
            if hasattr(self, 'winfo_exists') and not self.winfo_exists():
                self.temp_playing = False
                return
        except Exception:
            pass
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
        try:
            self.temp_slider.set(nxt)
        except Exception:
            pass
        self.update_tempseb(nxt)
        delay = int(max(100, 1000 / max(0.001, float(self.speed_var.get() or 1.0))))
        try:
            self._temp_after_id = self.after(delay, self.temp_play_step)
        except Exception:
            self._temp_after_id = None
            self.temp_playing = False

    # --- Run simulation and populate results + animation data ---
    def _on_run(self):
        t = threading.Thread(target=self._run_thread, daemon=True)
        t.start()

    def _run_thread(self):
        # mark running and update status (store internally only)
        self._running = True
        try:
            self._status_text = 'Running simulation...'
        except Exception:
            pass
        try:
            # inputs are validated when the user leaves each input box and saved
            # into self._params by _validate_and_store. Here we simply use those
            # canonical values and proceed to run the model.
            p = getattr(self, '_params', {})
            Sb = p.get('Sb', self.DEFAULTS['Sb'])
            trise = p.get('trise', int(self.DEFAULTS['trise_hr'] * self.hour))
            tset = p.get('tset', int(self.DEFAULTS['tset_hr'] * self.hour))
            Ldown = p.get('Ldown', self.DEFAULTS['Ldown'])
            hcoef = p.get('h', self.DEFAULTS['hcoef'])
            Ta_mean = p.get('Ta_mean', float(self.DEFAULTS['Ta_mean_C']) + 273.15)
            Ta_amp = p.get('Ta_amp', self.DEFAULTS['Ta_amp_C'])

            params = {'Sb': Sb, 'trise': trise, 'tset': tset, 'Ldown': Ldown, 'h': hcoef, 'Ta_mean': Ta_mean, 'Ta_amp': Ta_amp}
            # solver time grid (coarse) and forcing resolution (high-res for interpolation)
            tmax = int(48 * self.hour)
            dt_val = int(1800)
            t_eval = np.arange(0, tmax + dt_val, dt_val)
            # prefer passing a high-resolution forcing interval to the model so it
            # generates Ta/S0 on a 1-min grid for interpolation (forcing_dt in seconds)
            params.update({'t_array': t_eval, 'beta': float(p.get('beta', self.DEFAULTS['beta'])), 'forcing_dt': 60.0, 'thickness': float(p.get('thickness_A', self.DEFAULTS['thickA']))})
            mA = load_material(self.matA.get())
            outA = run_simulation(mA, tmax=tmax, dt=dt_val, **params)
            outB = None
            mB = None
            if self.compare_var.get():
                mB = load_material(self.matB.get())
                params_b = params.copy()
                params_b['thickness'] = float(p.get('thickness_B', self.DEFAULTS['thickB']))
                outB = run_simulation(mB, tmax=tmax, dt=dt_val, **params_b)

            # store for animation (include material metadata so we know if
            # evaporation is enabled per material)
            self.anim_data = {'A': outA, 'A_mat': mA, 'B': outB, 'B_mat': mB}

            # update UI on main thread (pass material dicts so results plotting
            # can suppress latent flux where evaporation is disabled)
            self.after(0, lambda: self._show_results(outA, outB, mA, mB))
            self.after(0, lambda: self.draw_anim_static())
            # configure temp slider to match number of time steps
            try:
                nsteps = len(outA['times'])
                # configure CTkSlider range (from_ remains 0)
                self.temp_slider.configure(to=max(1, nsteps - 1), number_of_steps=max(1, nsteps - 1))
                self.temp_slider.set(0)
                try:
                    self.btn_start.configure(state='normal')
                except Exception:
                    pass
            except Exception:
                pass
            self.after(0, lambda: setattr(self, '_status_text', 'Simulation complete'))
        except Exception as exc:
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
        try:
            self._hide_material_tooltip()
        except Exception:
            pass
        # stop loops
        try:
            self.animating = False
        except Exception:
            pass
        try:
            self.temp_playing = False
        except Exception:
            pass

        # cancel scheduled after callbacks if present
        try:
            pid = getattr(self, '_preview_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
                self._preview_after_id = None
        except Exception:
            pass
        try:
            pid = getattr(self, '_poll_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
                self._poll_after_id = None
        except Exception:
            pass
        try:
            pid = getattr(self, '_anim_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
                self._anim_after_id = None
        except Exception:
            pass
        try:
            pid = getattr(self, '_temp_after_id', None)
            if pid is not None:
                try:
                    self.after_cancel(pid)
                except Exception:
                    pass
                self._temp_after_id = None
        except Exception:
            pass

        # attempt to cancel any remaining Tk 'after' callbacks (including CTk internals)
        try:
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
        except Exception:
            pass

        # attempt to destroy the window (ends mainloop)
        try:
            self.destroy()
        except Exception:
            try:
                self.quit()
            except Exception:
                pass

    def _show_results(self, outA, outB: Optional[dict], mA: Optional[object] = None, mB: Optional[object] = None):
        # populate the results axes (6 rows)
        for a in self.axs_res:
            a.clear()
        t = outA['times'] / self.hour
        self.axs_res[0].plot(t, outA['Ts'] - 273.15, color='r', label='A')
        if outB is not None:
            self.axs_res[0].plot(t, outB['Ts'] - 273.15, color='b', linestyle='--', label='B')
        self.axs_res[0].set_ylabel('T_surf (°C)')
        self.axs_res[0].legend()

        self.axs_res[1].plot(t, outA['Qstar'], color='orange')
        if outB is not None:
            self.axs_res[1].plot(t, outB['Qstar'], color='orange', linestyle='--')
        self.axs_res[1].set_ylabel('K* (W/m2)')

        self.axs_res[2].plot(t, outA['L'], color='magenta')
        if outB is not None:
            self.axs_res[2].plot(t, outB['L'], color='magenta', linestyle='--')
        self.axs_res[2].set_ylabel('L* (W/m2)')

        self.axs_res[3].plot(t, outA['H'], color='green')
        if outB is not None:
            self.axs_res[3].plot(t, outB['H'], color='green', linestyle='--')
        self.axs_res[3].set_ylabel('H (W/m2)')

        # plot latent flux E only for materials that support evaporation
        try:
            if self._mat_allows_evap(mA):
                eA = outA.get('E', np.zeros_like(outA['times']))
            else:
                eA = np.zeros_like(outA['times'])
            self.axs_res[4].plot(t, eA, color='blue')
            if outB is not None:
                if self._mat_allows_evap(mB):
                    eB = outB.get('E', np.zeros_like(outB['times']))
                else:
                    eB = np.zeros_like(outA['times'])
                self.axs_res[4].plot(t, eB, color='blue', linestyle='--')
            self.axs_res[4].set_ylabel('E (W/m2)')
        except Exception:
            try:
                self.axs_res[4].plot(t, outA.get('E', np.zeros_like(outA['times'])), color='blue')
                if outB is not None:
                    self.axs_res[4].plot(t, outB.get('E', np.zeros_like(outA['times'])), color='blue', linestyle='--')
                self.axs_res[4].set_ylabel('E (W/m2)')
            except Exception:
                pass

        self.axs_res[5].plot(t, outA['G'], color='saddlebrown')
        if outB is not None:
            self.axs_res[5].plot(t, outB['G'], color='saddlebrown', linestyle='--')
        self.axs_res[5].set_ylabel('G (W/m2)')
        self.axs_res[5].set_xlabel('Time (h)')

        try:
            self.canvas_res.draw()
        except Exception:
            pass

        # populate compare tab if present
        try:
            axs_comp = getattr(self, 'axs_comp', None)
            can_comp = getattr(self, 'canvas_comp', None)
            if outB is not None and axs_comp is not None:
                for a in axs_comp:
                    a.clear()
                # difference B - A
                axs_comp[0].plot(t, outB['Ts'] - outA['Ts'], color='purple')
                axs_comp[0].set_ylabel('ΔT_s (°C)')
                axs_comp[1].plot(t, outB['Qstar'] - outA['Qstar'], color='orange')
                axs_comp[1].set_ylabel('ΔK* (W/m2)')
                axs_comp[2].plot(t, outB['L'] - outA['L'], color='magenta')
                axs_comp[2].set_ylabel('ΔL* (W/m2)')
                axs_comp[3].plot(t, outB['H'] - outA['H'], color='green')
                axs_comp[3].set_ylabel('ΔH (W/m2)')
                axs_comp[4].plot(t, outB['E'] - outA['E'], color='blue')
                axs_comp[4].set_ylabel('ΔE (W/m2)')
                axs_comp[5].plot(t, outB['G'] - outA['G'], color='saddlebrown')
                axs_comp[5].set_ylabel('ΔG (W/m2)')
                axs_comp[5].set_xlabel('Time (h)')
                try:
                    if can_comp is not None:
                        can_comp.draw()
                except Exception:
                    pass
        except Exception:
            pass

        # update Temp & SEB panel (time index 0 default)
        try:
            self.update_tempseb(0)
        except Exception:
            pass


if __name__ == '__main__':
    app = App()
    app.mainloop()

