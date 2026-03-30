"""
=============================================================================
 Regenerative Suspension System (RSS) — Combined Simulation
=============================================================================
 Physics model
 -------------
   Quarter-car (2-DOF):
     m1·z̈1 = −c1·(ż1−ż2) − k1·(z1−z2)            [sprung mass / body]
     m2·z̈2 = +c1·(ż1−ż2) + k1·(z1−z2) − k2·(z2−h(t)) [unsprung / wheel]

   Road profile:  h(t) = A·sin(2π·f1·t) + 0.5A·sin(2π·f2·t)
     f = spatial_freq × v_ms   (speed shifts temporal frequency)

   Instantaneous available power:  P_avail = c1·v_rel²

   RSS power equations (impedance-matched):
     Linear:    P = Ke²·v_rel² / (4·R_coil)
     Rotary:    P = (Kt·M)²·v_rel² / (4·r_coil)   M = 2π·G/L
     Hydraulic: P = η_hyd·η_em·(Kt·Ap/Dm)²·v_rel² / (4·r_coil)

 Figures
 -------
   Figure 1 — Static overview: kinematics, instantaneous power, cumulative energy
   Figure 2 — Parametric analysis: velocity, Pareto front, mass, road roughness
   Figure 3 — Architecture comparison: energy by road class & all architectures vs speed
=============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
from scipy.integrate import solve_ivp, trapezoid
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

for _backend in ("TkAgg", "Qt5Agg", "MacOSX", "Agg"):
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        pass


# =============================================================================
#  WHITE REPORT PALETTE
# =============================================================================
BG        = "white"
PANEL     = "#f8f9fa"
BORDER    = "#cccccc"
COL_LIN   = "#c0392b"   # Linear EM       — deep red
COL_ROT   = "#16a085"   # Rotary Mech.    — teal
COL_HYD   = "#2980b9"   # Hydraulic EM    — blue
COL_AVAIL = "#e67e22"   # Available power — orange
COL_ROAD  = "#8e44ad"   # Road profile    — purple
COL_ZS    = "#c0392b"   # Sprung mass     — red
COL_ZU    = "#27ae60"   # Unsprung mass   — green
BLACK     = "#1a1a1a"
GREY      = "#555555"
LGREY     = "#888888"
EU_RED    = "#e74c3c"

matplotlib.rcParams.update({
    "figure.facecolor":   BG,
    "axes.facecolor":     PANEL,
    "axes.edgecolor":     BORDER,
    "axes.labelcolor":    BLACK,
    "axes.titlecolor":    BLACK,
    "xtick.color":        GREY,
    "ytick.color":        GREY,
    "text.color":         BLACK,
    "grid.color":         BORDER,
    "grid.alpha":         0.7,
    "legend.facecolor":   "white",
    "legend.edgecolor":   BORDER,
    "legend.labelcolor":  BLACK,
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "lines.linewidth":    2.0,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
    "legend.fontsize":    10,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
})


# =============================================================================
#  SYSTEM PARAMETERS  (Zuo & Zhang 2013)
# =============================================================================
M1_BASE  = 320.0      # Sprung mass, quarter-car body  [kg]
M2       = 59.0       # Unsprung mass, wheel + axle    [kg]
K1_BASE  = 18709.0    # Suspension spring stiffness    [N/m]
K2       = 233100.0   # Tyre stiffness                 [N/m]
C1_BASE  = 1500.0     # Baseline damping coefficient   [Ns/m]

# RSS first-principles constants (calibrated: Linear~20%, Rotary~42%, Hydraulic~45%)
KE_LIN   = 50.0       # Linear back-EMF constant   [Vs/m]
R_LIN    = 4.0        # Linear coil resistance      [Ω]

KT_ROT   = 0.0565     # Rotary generator Kt         [Vs/rad]
L_ROT    = 0.01       # Ball-screw lead             [m/rev]
G_ROT    = 2.0        # Gear ratio
R_ROT    = 2.0        # Rotary coil resistance      [Ω]

KT_HYD   = 0.0422     # Hydraulic generator Kt      [Vs/rad]
AP_HYD   = 0.002      # Piston area                 [m²]
DM_HYD   = 1e-6       # Motor displacement          [m³/rad]
R_HYD    = 2.0        # Hydraulic coil resistance   [Ω]
ETA_HYD  = 0.85       # Hydraulic circuit efficiency
ETA_EM   = 0.90       # EM conversion efficiency

ETA_ROT_FLAT = 0.426  # Flat rotary efficiency (Lui et al. 2017)
ETA_LEM = 0.205       # Linear EM efficiency (from meta-analysis of literature)
ETA_ROT = 0.409       # Rotary mech. efficiency (from meta-analysis of literature)
ETA_HYDR = 0.366      # Hydraulic EM efficiency (from meta-analysis of literature)

# ISO 8608 road-class amplitude multipliers (relative to Class A)
ROAD_CLASSES = {
    "A — Very Smooth": 1.0,
    "B — Smooth":      2.5,
    "C — Average":     3.6,
    "D — Rough":       5.5,
    "E — Very Rough":  9.0,
}


# =============================================================================
#  PHYSICS
# =============================================================================

def road_profile(t, v_kmph, road_mult):
    """Sinusoidal road profile. Spatial frequency shifts with speed."""
    v  = max(v_kmph / 3.6, 0.1)
    f1 = 0.1 * v
    f2 = 0.4 * v
    A  = road_mult * 0.01
    return A * np.sin(2 * np.pi * f1 * t) + (A * 0.5) * np.sin(2 * np.pi * f2 * t)


def qc_rhs(t, y, v_kmph, road_mult, m1, c1):
    """
    Quarter-car equations of motion.
    State: [z1, ż1, z2, ż2]  — z1=sprung (body), z2=unsprung (wheel)
    Suspension spring and damping scale with sprung mass ratio to maintain
    similar natural frequency across vehicle classes.
    """
    z1, v1, z2, v2 = y
    h     = road_profile(t, v_kmph, road_mult)
    ratio = m1 / M1_BASE
    k1_s  = K1_BASE * ratio
    c1_s  = c1 * ratio
    v_rel = v1 - v2
    F_sus = c1_s * v_rel + k1_s * (z1 - z2)
    F_tyr = K2 * (z2 - h)
    return [v1, -F_sus / m1, v2, (F_sus - F_tyr) / M2]


def run_sim(v_kmph=80.0, road_mult=1.0, m1=M1_BASE, c1=C1_BASE,
            t_end=6.0, n_pts=1500):
    """Integrate ODE and compute power for all three RSS architectures."""
    t_eval = np.linspace(0, t_end, n_pts)
    sol    = solve_ivp(
        qc_rhs, (0, t_end), [0, 0, 0, 0],
        t_eval=t_eval, args=(v_kmph, road_mult, m1, c1),
        method="RK45", rtol=1e-5, atol=1e-8,
    )
    z1, v1, z2, v2 = sol.y
    t = sol.t

    ratio  = m1 / M1_BASE
    k1_act = K1_BASE * ratio
    c1_act = c1 * ratio
    v_rel  = v1 - v2
    h_arr  = road_profile(t, v_kmph, road_mult)
    z1_dd  = (-c1_act * v_rel - k1_act * (z1 - z2)) / m1

    P_avail  = c1_act * v_rel ** 2

    M_amp  = (2 * np.pi * G_ROT) / L_ROT
    P_lin  = ((KE_LIN ** 2 * v_rel ** 2) / (4 * R_LIN))*ETA_LEM
    P_rot  = ((KT_ROT * M_amp) ** 2 * v_rel ** 2 / (4 * R_ROT))*ETA_ROT
    P_hyd  = (ETA_HYD * ETA_EM * (KT_HYD * AP_HYD / DM_HYD) ** 2
               * v_rel ** 2 / (4 * R_HYD))*ETA_HYDR
    P_rot_fl = P_avail * ETA_ROT_FLAT

    def cumul(P):
        return np.array([trapezoid(P[:i+1], t[:i+1]) for i in range(len(t))])

    return dict(
        t=t, z1=z1, v1=v1, z2=z2, v2=v2,
        h=h_arr, v_rel=v_rel, z1_dd=z1_dd,
        P_avail=P_avail,
        P_lin=P_lin, P_rot=P_rot, P_hyd=P_hyd,
        P_rot_fl=P_rot_fl,
        cumE_lin=cumul(P_lin),
        cumE_rot=cumul(P_rot),
        cumE_hyd=cumul(P_hyd),
        E_lin=float(trapezoid(P_lin, t)),
        E_rot=float(trapezoid(P_rot, t)),
        E_hyd=float(trapezoid(P_hyd, t)),
        E_rot_fl=float(trapezoid(P_rot_fl, t)),
        avg_avail=float(np.mean(P_avail)),
        avg_lin=float(np.mean(P_lin)),
        avg_rot=float(np.mean(P_rot)),
        avg_hyd=float(np.mean(P_hyd)),
        rms_accel=float(np.sqrt(np.mean(z1_dd ** 2))),
    )


# =============================================================================
#  AXIS STYLE HELPER
# =============================================================================

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=GREY, labelsize=10)
    ax.grid(True, color=BORDER, alpha=0.7, lw=0.6)
    if title:   ax.set_title(title,  color=BLACK, fontsize=13, pad=7, fontweight="bold")
    if xlabel:  ax.set_xlabel(xlabel, color=BLACK, fontsize=11)
    if ylabel:  ax.set_ylabel(ylabel, color=BLACK, fontsize=11)


# =============================================================================
#  FIGURE 1 — STATIC OVERVIEW  (no sliders — report-ready)
# =============================================================================

def build_overview():
    """
    Three-row static overview figure.
    Default conditions: 80 km/h, ISO 8608 Class C road.
    """
    r = run_sim(v_kmph=80, road_mult=ROAD_CLASSES["C — Average"])
    t = r["t"]

    fig = plt.figure(figsize=(16, 14), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("RSS Overview — 80 km/h, Class C")
    except Exception:
        pass

    fig.suptitle(
        "Regenerative Suspension System — Performance Overview\n",
        fontsize=14, color=BLACK, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(3, 2, top=0.925, bottom=0.07,
                           hspace=0.55, wspace=0.32, left=0.08, right=0.97)

    ax_kin  = fig.add_subplot(gs[0, :])
    ax_pow  = fig.add_subplot(gs[1, :])
    ax_ener = fig.add_subplot(gs[2, :])

    # ── Kinematics ────────────────────────────────────────────────────────────
    style_ax(ax_kin, "Suspension Kinematics", "Time (s)", "Displacement (mm)")
    ax_kin.plot(t, r["h"]  * 1000, color=COL_ROAD, lw=1.6, ls="--",
                label="Road profile  h(t)", alpha=0.85)
    ax_kin.plot(t, r["z1"] * 1000, color=COL_ZS,   lw=2.0,
                label="Sprung mass  z₁  (body)")
    ax_kin.plot(t, r["z2"] * 1000, color=COL_ZU,   lw=2.0,
                label="Unsprung mass  z₂  (wheel)")
    ax_kin.legend(loc="upper right", framealpha=0.95)
    ax_kin.text(0.01, 0.92,
                f"m₁ = {M1_BASE:.0f} kg  ·  c₁ = {C1_BASE:.0f} Ns/m  ·  "
                f"k₁ = {K1_BASE:.0f} N/m  ·  ISO Class C",
                transform=ax_kin.transAxes, color=LGREY, fontsize=9)

    # ── Instantaneous power ───────────────────────────────────────────────────
    style_ax(ax_pow, "Instantaneous Harvestable Power  P(t)",
             "Time (s)", "Power (W)")
    ax_pow.plot(t, r["P_avail"], color=COL_AVAIL, lw=1.4, alpha=0.55,
                label=f"Available  — mean {r['avg_avail']:.1f} W")
    ax_pow.plot(t, r["P_lin"],   color=COL_LIN,   lw=1.8,
                label=f"Linear EM  — mean {r['avg_lin']:.1f} W")
    ax_pow.plot(t, r["P_rot"],   color=COL_ROT,   lw=1.8,
                label=f"Rotary Mech.  — mean {r['avg_rot']:.1f} W")
    ax_pow.plot(t, r["P_hyd"],   color=COL_HYD,   lw=1.8,
                label=f"Hydraulic EM  — mean {r['avg_hyd']:.1f} W")
    ax_pow.legend(loc="upper right", framealpha=0.95, fontsize=9)

    # ── Cumulative energy ─────────────────────────────────────────────────────
    style_ax(ax_ener, "Cumulative Energy Harvested",
             "Time (s)", "Energy (J)")
    ax_ener.plot(t, r["cumE_lin"], color=COL_LIN, lw=2.2,
                 label=f"Linear EM  —  {r['E_lin']:.1f} J total  (η ≈ 20.6%)")
    ax_ener.plot(t, r["cumE_rot"], color=COL_ROT, lw=2.2,
                 label=f"Rotary Mech.  —  {r['E_rot']:.1f} J total  (η ≈ 40.9%)")
    ax_ener.plot(t, r["cumE_hyd"], color=COL_HYD, lw=2.2,
                 label=f"Hydraulic EM  —  {r['E_hyd']:.1f} J total  (η ≈ 36.6%)")
    for arr, val, col in [(r["cumE_lin"], r["E_lin"], COL_LIN),
                           (r["cumE_rot"], r["E_rot"], COL_ROT),
                           (r["cumE_hyd"], r["E_hyd"], COL_HYD)]:
        ax_ener.annotate(f"  {val:.0f} J",
                          xy=(t[-1], arr[-1]),
                          color=col, fontsize=10, va="center",
                          xytext=(t[-1] + 0.05, arr[-1]))
    ax_ener.legend(loc="upper left", framealpha=0.95)
    ax_ener.set_xlim(right=t[-1] * 1.12)

    return fig


# =============================================================================
#  FIGURE 2 — PARAMETRIC ANALYSIS
# =============================================================================

def build_parametric():
    """
    Two panels:
      (c) Pareto front: ride comfort vs energy  (damping sweep at 60 km/h)
      (d) Road roughness comparison  (Classes A, C, D)
    """
    print("  [Fig 2] Computing parametric sweeps …", flush=True)

    speeds = np.linspace(20, 120, 15)
    
    # (c) Pareto front
    #     Fixed springs (no mass scaling). Sweep c1 from 200 to 5000 Ns/m.
    #     At 60 km/h, road_mult=0.25 the EU comfort limit (0.5 m/s²) falls
    #     inside the sweep range, producing a meaningful trade-off curve.
    damping_vals = np.linspace(200, 5000, 30)
    par_energy, par_comfort = [], []
    for c in damping_vals:
        def _rhs(t, y, _c=c):
            z1, v1, z2, v2 = y
            h     = road_profile(t, 60, 0.25)
            v_rel = v1 - v2
            F_sus = _c * v_rel + K1_BASE * (z1 - z2)
            F_tyr = K2 * (z2 - h)
            return [v1, -F_sus / M1_BASE, v2, (F_sus - F_tyr) / M2]
        sol = solve_ivp(_rhs, (0, 8), [0, 0, 0, 0],
                         t_eval=np.linspace(0, 8, 2000),
                         method="RK45", rtol=1e-5, atol=1e-8)
        z1, v1, z2, v2 = sol.y
        v_rel  = v1 - v2
        z1_dd  = (-c * v_rel - K1_BASE * (z1 - z2)) / M1_BASE
        P_fl   = c * v_rel ** 2 * ETA_ROT_FLAT
        par_comfort.append(float(np.sqrt(np.mean(z1_dd ** 2))))
        par_energy.append(float(np.mean(P_fl)))

    # (d) Road roughness
    pw_a = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["A — Very Smooth"])["avg_rot"]
            for v in speeds]
    pw_c = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["C — Average"])["avg_rot"]
            for v in speeds]
    pw_d = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["D — Rough"])["avg_rot"]
            for v in speeds]

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("Parametric Analysis — Comfort, Energy, Road Roughness")
    except Exception:
        pass
    fig.subplots_adjust(wspace=0.30, top=0.910, bottom=0.11, left=0.08, right=0.97)

    # (c) Pareto front
    ax = axs[0]
    style_ax(ax, "(c) Pareto Front: Ride Comfort vs Energy Recovery",
              "RMS Sprung-Mass Acceleration  (m/s²)",
              "Mean Harvestable Power  (W)")
    sc = ax.scatter(par_comfort, par_energy, c=damping_vals,
                     cmap="plasma", s=70, zorder=4, label="Simulation points")
    ax.plot(par_comfort, par_energy, color=LGREY, ls="--", lw=1.0, alpha=0.6)
    ax.axvline(0.5, color=EU_RED, lw=2.2, ls="--",
                label="EU H&S comfort limit  (0.5 m/s²)")
    ax.axvspan(0.5, max(par_comfort) * 1.08, alpha=0.07, color=EU_RED)
    ax.text(0.70, 0.48, "Comfort infeasible\n(EU limit exceeded)",
             transform=ax.transAxes, color=EU_RED, fontsize=9, va="top")
    ax.text(0.02, 0.48, "Comfort feasible",
             transform=ax.transAxes, color="#16a085", fontsize=9, va="top")
    ax.text(0.02, 0.96, "(Initial Condition:\n60 km/h,  c₁ swept 200 – 5000 Ns/m)",
             transform=ax.transAxes, color="#000000", fontsize=9, va="top")
    cbar = plt.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label("Damping coefficient  $c_1$  (Ns/m)", fontsize=9)
    cbar.ax.yaxis.set_tick_params(labelsize=9)
    ax.legend(framealpha=0.95, loc="lower right", fontsize=9)
    ax.set_xlim(left=0.0)

    # (d) road roughness
    ax = axs[1]
    style_ax(ax, "(d) Road Roughness Scaling  ($G_d$ effect)",
              "Speed (km/h)", "Mean Harvestable Power per Corner (W)")
    ax.plot(speeds, pw_a, color="#7f8c8d", lw=2.0, marker="^", ms=5,
             label="Class A — Very Smooth (motorway)")
    ax.plot(speeds, pw_c, color=COL_ROT,   lw=2.0, marker="o", ms=5,
             label="Class C — Average (typical UK road)")
    ax.plot(speeds, pw_d, color=COL_LIN,   lw=2.0, marker="s", ms=5,
             label="Class D — Rough")
    ax.fill_between(speeds, pw_a, pw_d, alpha=0.09, color=COL_ROT)
    idx = np.argmin(np.abs(speeds - 80))
    fac = pw_d[idx] / pw_a[idx] if pw_a[idx] > 0 else 0
    ax.annotate(f"×{fac:.0f} gain  A→D\nat 80 km/h",
                 xy=(80, pw_d[idx]),
                 xytext=(85, pw_d[idx] * 0.80),
                 color=COL_LIN, fontsize=9,
                 arrowprops=dict(arrowstyle="->", color=COL_LIN, lw=1.2))
    ax.legend(framealpha=0.95, fontsize=9)

    print("  [Fig 2] Done.", flush=True)
    return fig


# =============================================================================
#  FIGURE 3 — ARCHITECTURE COMPARISON
# =============================================================================

def build_architecture():
    """
    Two panels:
      (a) Grouped bar: energy harvested per architecture for each ISO road class
      (b) Mean power vs speed for all three architectures on Class C road
    """
    print("  [Fig 3] Computing architecture comparison …", flush=True)

    class_names = list(ROAD_CLASSES.keys())
    class_mults = list(ROAD_CLASSES.values())
    short_names = [c.split("—")[0].strip() for c in class_names]

    E_lin_arr, E_rot_arr, E_hyd_arr = [], [], []
    for mult in class_mults:
        r = run_sim(v_kmph=80, road_mult=mult)
        E_lin_arr.append(r["E_lin"])
        E_rot_arr.append(r["E_rot"])
        E_hyd_arr.append(r["E_hyd"])

    speeds = np.linspace(20, 120, 15)
    pw_lin = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["C — Average"])["avg_lin"]
              for v in speeds]
    pw_rot = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["C — Average"])["avg_rot"]
              for v in speeds]
    pw_hyd = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["C — Average"])["avg_hyd"]
              for v in speeds]
    pw_av  = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["C — Average"])["avg_avail"]
              for v in speeds]

    fig, axs = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    try:
        fig.canvas.manager.set_window_title("RSS Architecture Comparison")
    except Exception:
        pass
    fig.suptitle(
        "Architecture Comparison — Linear EM  vs  Rotary Mech.  vs  Hydraulic EM",
        fontsize=14, color=BLACK, fontweight="bold", y=0.99,
    )
    fig.subplots_adjust(wspace=0.30, top=0.910, bottom=0.11, left=0.08, right=0.97)

    # (a) Grouped bar
    ax = axs[0]
    style_ax(ax, "(a) Energy Harvested per Corner vs Road Class\n(80 km/h, 6-second run)",
              "ISO 8608 Road Class", "Energy Harvested per Corner (J)")
    x = np.arange(len(class_names))
    w = 0.26
    b1 = ax.bar(x - w, E_lin_arr, w, color=COL_LIN, alpha=0.88,
                 label="Linear EM  (η ≈ 20.6%)", zorder=3)
    b2 = ax.bar(x,     E_rot_arr, w, color=COL_ROT, alpha=0.88,
                 label="Rotary Mech.  (η ≈ 40.9%)", zorder=3)
    b3 = ax.bar(x + w, E_hyd_arr, w, color=COL_HYD, alpha=0.88,
                 label="Hydraulic EM  (η ≈ 36.6%)", zorder=3)
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if h > 8:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                         f"{h:.0f}", ha="center", va="bottom", fontsize=8, color=BLACK)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, fontsize=10)
    ax.legend(framealpha=0.95, loc="upper left")

    pw_speed = [run_sim(v_kmph=v, road_mult=ROAD_CLASSES["C — Average"])["avg_rot"]
                for v in speeds]

    # (b) Power vs speed
    ax = axs[1]
    style_ax(ax, "(b) Mean Power vs Speed  (ISO 8608 Class C)",
              "Vehicle Speed (km/h)", "Mean Harvestable Power per Corner (W)")
    ax.plot(speeds, pw_av,  color=COL_AVAIL, lw=1.6, ls="--", alpha=0.65,
             label="Available power  (passive damper)")
    ax.plot(speeds, pw_lin, color=COL_LIN,   lw=2.2, marker="^", ms=5,
             label="Linear EM  (η ≈ 20.6%)")
    ax.plot(speeds, pw_rot, color=COL_ROT,   lw=2.2, marker="o", ms=5,
             label="Rotary Mech.  (η ≈ 40.9%)")
    ax.plot(speeds, pw_hyd, color=COL_HYD,   lw=2.2, marker="s", ms=5,
             label="Hydraulic EM  (η ≈ 36.6%)")
    v_ms  = speeds / 3.6
    coeff = np.polyfit(v_ms ** 2, pw_speed, 1)
    v_fit = np.linspace(speeds[0], speeds[-1], 100)
    ax.plot(v_fit, np.polyval(coeff, (v_fit / 3.6) ** 2),
             ls="--", color=LGREY, lw=1.4, label="$P \\propto V^2$ fit")
    ax.legend(framealpha=0.95, loc="upper left", fontsize=9)

    print("  [Fig 3] Done.", flush=True)
    return fig


# =============================================================================
#  ENTRY POINT
# =============================================================================

def main():
    print()
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║  RSS Simulation  —  Report-Ready Figures                         ║")
    print("║  White background  ·  Full legends  ·  Accurate physics          ║")
    print("╚══════════════════════════════════════════════════════════════════╝")
    print()

    print("  Building Figure 1: Static overview …")
    f1 = build_overview()

    print("  Building Figure 2: Parametric analysis …")
    f2 = build_parametric()

    print("  Building Figure 3: Architecture comparison …")
    f3 = build_architecture()

    print()
    print("  ✓  Three report-ready figures produced:")
    print("     Fig 1 — Overview     (kinematics · power · cumulative energy)")
    print("     Fig 2 — Parametric   (speed · Pareto front · mass · road class)")
    print("     Fig 3 — Architectures(energy by road class · power vs speed)")
    print()

    plt.show()


if __name__ == "__main__":
    main()
