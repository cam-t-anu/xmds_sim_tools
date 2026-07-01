#!/usr/bin/python3
"""
Gradient Echo Memory (GEM) XMDS2 Output Visualizer — EIT multi-mF edition
--------------------------------------------------------------------------
Supports both 1D (shape: t,x) and 3D (shape: t,x,y,z) XMDS2 outputs.
Extra mF=0 and mF=-1 levels are loaded when present; otherwise they are
treated as zero.

Panels (all modes):
  1. Spatial profiles over x at selected time  [time slider]
  2. Probe intensity at x=0 and x=x_max vs all time
  3. Spin-wave magnitude heatmap         (x vs t)  [mF selector]
  4. Spin-wave phase heatmap             (x vs t)  [same mF]
  5. Probe intensity |E|² heatmap        (x vs t)
  6. Excited coherence magnitude heatmap (x vs t)  [same mF]
  7. Excited coherence phase heatmap     (x vs t)  [same mF]

Additional 3D panel:
  8. Control-field transverse profile (y–z plane) at selected x and t

In 3D mode all 1D-style panels use fields averaged over y and z.

Usage:
    python gem-visualiser_new.py [path_to_hdf5_file]
"""

import sys
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, CheckButtons, RadioButtons
from matplotlib.ticker import AutoMinorLocator
from matplotlib.colors import LogNorm, Normalize

# ── file path ─────────────────────────────────────────────────────────────────
DEFAULT_FILE = (
    "/mnt/user-data/uploads/"
    "1Dgem-eit_De_8_7_Np_1_Ome_0_35_Omg_0_35_Pw_12_Tin_420_a0_0_a1_4_am1_0_"
    "bias_0_02_c0_1_dr_0_0023_ds_0_0023_gft_0_53_lds_0_ldw_1_mf_0_pdiff_0_"
    "pmdiff_0_tgap_40.h5"
)
h5_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FILE

# ── load data ─────────────────────────────────────────────────────────────────
with h5py.File(h5_path, "r") as f:
    g = f["1"]
    t_arr    = g["t"][:]
    x_arr    = g["x"][:]
    EI       = g["EI"][:]
    ER       = g["ER"][:]
    phi_geR  = g["phi_geR"][:]
    phi_geI  = g["phi_geI"][:]
    phi_gsR  = g["phi_gsR"][:]
    phi_gsI  = g["phi_gsI"][:]
    ctrl     = g["ctrl"][:]
    grad     = g["grad"][:]
    detuning = g["detuning"][:]
    Ni       = g["Ni_atoms"][:]
    y_arr    = g["y"][:] if "y" in g else None
    z_arr    = g["z"][:] if "z" in g else None

    def _load_or_zeros(key):
        return g[key][:] if key in g else np.zeros_like(ER)

    phi_ge0R  = _load_or_zeros("phi_ge0R")
    phi_ge0I  = _load_or_zeros("phi_ge0I")
    phi_gs0R  = _load_or_zeros("phi_gs0R")
    phi_gs0I  = _load_or_zeros("phi_gs0I")
    phi_gem1R = _load_or_zeros("phi_gem1R")
    phi_gem1I = _load_or_zeros("phi_gem1I")
    phi_gsm1R = _load_or_zeros("phi_gsm1R")
    phi_gsm1I = _load_or_zeros("phi_gsm1I")

# ── dimensionality ─────────────────────────────────────────────────────────────
is_3d = EI.ndim == 4   # shape (n_t, n_x, n_y, n_z)

if is_3d:
    ctrl_trans_3d = ctrl.copy()
    E_trans_3d    = ER**2 + EI**2
    x_ci          = 0
    y_mm          = y_arr * 1e3
    z_mm          = z_arr * 1e3
    for _arr_name in ("EI", "ER", "phi_geR", "phi_geI", "phi_gsR", "phi_gsI",
                      "ctrl", "grad", "detuning", "Ni",
                      "phi_ge0R", "phi_ge0I", "phi_gs0R", "phi_gs0I",
                      "phi_gem1R", "phi_gem1I", "phi_gsm1R", "phi_gsm1I"):
        globals()[_arr_name] = globals()[_arr_name].mean(axis=(2, 3))

# ── derived quantities ────────────────────────────────────────────────────────
E_int = ER**2 + EI**2

# mF = +1
phi_ge2      = phi_geR**2  + phi_geI**2
phi_gs2      = phi_gsR**2  + phi_gsI**2
phi_gs_mag   = np.sqrt(phi_gs2)
phi_gs_phase = np.angle(phi_gsR + 1j * phi_gsI)
phi_ge_mag   = np.sqrt(phi_ge2)
phi_ge_phase = np.angle(phi_geR + 1j * phi_geI)

# mF = 0
phi_ge0_2     = phi_ge0R**2  + phi_ge0I**2
phi_gs0_2     = phi_gs0R**2  + phi_gs0I**2
phi_gs0_mag   = np.sqrt(phi_gs0_2)
phi_gs0_phase = np.angle(phi_gs0R + 1j * phi_gs0I)
phi_ge0_mag   = np.sqrt(phi_ge0_2)
phi_ge0_phase = np.angle(phi_ge0R + 1j * phi_ge0I)

# mF = -1
phi_gem1_2     = phi_gem1R**2  + phi_gem1I**2
phi_gsm1_2     = phi_gsm1R**2  + phi_gsm1I**2
phi_gsm1_mag   = np.sqrt(phi_gsm1_2)
phi_gsm1_phase = np.angle(phi_gsm1R + 1j * phi_gsm1I)
phi_gem1_mag   = np.sqrt(phi_gem1_2)
phi_gem1_phase = np.angle(phi_gem1R + 1j * phi_gem1I)

t_us = t_arr * 1e6
x_mm = x_arr * 1e3

E_in  = E_int[:, 0]
E_out = E_int[:, -1]

def t_idx(t_val_us):
    return int(np.argmin(np.abs(t_us - t_val_us)))

# ── heatmap mF options ────────────────────────────────────────────────────────
# Each entry: (radio label, gs_mag, gs_phase, ge_mag, ge_phase)
HEATMAP_OPTIONS = [
    (r"$m_F=1$",  phi_gs_mag,   phi_gs_phase,   phi_ge_mag,   phi_ge_phase),
    (r"$m_F=0$",  phi_gs0_mag,  phi_gs0_phase,  phi_ge0_mag,  phi_ge0_phase),
    (r"$m_F=-1$", phi_gsm1_mag, phi_gsm1_phase, phi_gem1_mag, phi_gem1_phase),
]
HEATMAP_LABELS = [h[0] for h in HEATMAP_OPTIONS]

# ── theme ─────────────────────────────────────────────────────────────────────
BG      = "#1A1A2E"
PANEL   = "#12122A"
SPINE   = "#444466"
TC      = "#CCCCDD"
GRID_MJ = "#2A2A4A"
GRID_MN = "#1E1E38"

COLORS = {
    "E_int":      "#4C9BE8",
    "ER":         "#2176AE",
    "EI":         "#57C4E5",
    "phi_ge2":    "#E8734C",
    "phi_gs2":    "#55C464",
    "phi_ge0_2":  "#E8A84C",
    "phi_gs0_2":  "#55C4A8",
    "phi_gem1_2": "#E84CA8",
    "phi_gsm1_2": "#5588E8",
    "ctrl":       "#B07FD4",
    "grad":       "#E8C34C",
    "detuning":   "#E84C6A",
    "Ni":         "#888888",
}

LABELS = {
    "E_int":      r"$|E|^2$ (probe intensity)",
    "ER":         r"$E_R$ (probe, real)",
    "EI":         r"$E_I$ (probe, imag)",
    "phi_ge2":    r"$|\varphi_{ge}|^2$ ($m_F=1$, excited coherence)",
    "phi_gs2":    r"$|\varphi_{gs}|^2$ ($m_F=1$, spin-wave)",
    "phi_ge0_2":  r"$|\varphi_{ge,0}|^2$ ($m_F=0$, excited coherence)",
    "phi_gs0_2":  r"$|\varphi_{gs,0}|^2$ ($m_F=0$, spin-wave)",
    "phi_gem1_2": r"$|\varphi_{ge,-1}|^2$ ($m_F=-1$, excited coherence)",
    "phi_gsm1_2": r"$|\varphi_{gs,-1}|^2$ ($m_F=-1$, spin-wave)",
    "ctrl":       r"ctrl field",
    "grad":       r"gradient",
    "detuning":   r"detuning",
    "Ni":         r"$N_i$ (atom density)",
}

LEFT_KEYS = [
    "E_int", "ER", "EI",
    "phi_ge2",    "phi_gs2",
    "phi_ge0_2",  "phi_gs0_2",
    "phi_gem1_2", "phi_gsm1_2",
]
RIGHT_KEYS = ["ctrl", "grad", "detuning", "Ni"]
DATA_MAP = {
    "E_int":      E_int,      "ER": ER,           "EI": EI,
    "phi_ge2":    phi_ge2,    "phi_gs2":    phi_gs2,
    "phi_ge0_2":  phi_ge0_2,  "phi_gs0_2":  phi_gs0_2,
    "phi_gem1_2": phi_gem1_2, "phi_gsm1_2": phi_gsm1_2,
    "ctrl": ctrl, "grad": grad, "detuning": detuning, "Ni": Ni,
}

# ── helpers ───────────────────────────────────────────────────────────────────
def style_ax(ax, xlabel="", ylabel="", fs=8):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TC, labelsize=fs)
    for sp in ax.spines.values():
        sp.set_edgecolor(SPINE)
    if xlabel:
        ax.set_xlabel(xlabel, color=TC, fontsize=fs + 1)
    if ylabel:
        ax.set_ylabel(ylabel, color=TC, fontsize=fs + 1)
    ax.grid(True, which="major", color=GRID_MJ, lw=0.7)
    ax.grid(True, which="minor", color=GRID_MN, lw=0.3)
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

def _make_norm(mag):
    pos = mag[mag > 0]
    mx  = mag.max()
    if mx > 0 and len(pos) > 0 and (mx / pos.min()) > 1e3:
        return LogNorm(vmin=max(pos.min(), mx * 1e-6), vmax=mx), True
    return Normalize(vmin=0, vmax=mx if mx > 0 else 1), False

def _mask_phase(mag, phase):
    thresh = mag.max() * 1e-4
    pd = phase.copy()
    pd[mag < thresh] = np.nan
    return pd

# ── layout ────────────────────────────────────────────────────────────────────
fig_w    = 24 if is_3d else 22
slider_h = 1.3 if is_3d else 0.65
fig = plt.figure(figsize=(fig_w, 15), facecolor=BG)
fig.canvas.manager.set_window_title("GEM Visualizer — EIT multi-mF")

outer = gridspec.GridSpec(
    4, 1, figure=fig,
    height_ratios=[5, 3.5, 3.5, slider_h],
    hspace=0.48,
    left=0.06, right=0.97, top=0.94, bottom=0.06,
)

# Row 0: spatial profile + checkboxes
r0 = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=outer[0],
                                      width_ratios=[4, 1], wspace=0.05)
ax_main = fig.add_subplot(r0[0])
ax_ctrl = fig.add_subplot(r0[1], facecolor=PANEL)

style_ax(ax_main, xlabel="x  (mm)", ylabel="field amplitude / intensity  (arb.)")
ax_right = ax_main.twinx()
ax_right.set_facecolor(PANEL)
ax_right.tick_params(colors="#AAAAAA", labelsize=8)
ax_right.set_ylabel("ctrl / grad / detuning / Ni  (arb.)", color="#AAAAAA", fontsize=8)
for sp in ax_right.spines.values():
    sp.set_edgecolor(SPINE)

# Row 1: probe I/O | mF radio | spin-wave mag | spin-wave phase [| transverse 3D]
if is_3d:
    r1 = gridspec.GridSpecFromSubplotSpec(
        1, 5, subplot_spec=outer[1],
        width_ratios=[1, 0.22, 1.2, 1.2, 1.1], wspace=0.44)
else:
    r1 = gridspec.GridSpecFromSubplotSpec(
        1, 4, subplot_spec=outer[1],
        width_ratios=[1, 0.22, 1.2, 1.2], wspace=0.45)

ax_io       = fig.add_subplot(r1[0])
ax_radio_bg = fig.add_subplot(r1[1])
ax_heat     = fig.add_subplot(r1[2])
ax_phase    = fig.add_subplot(r1[3])
if is_3d:
    ax_trans = fig.add_subplot(r1[4])

style_ax(ax_io,    xlabel="time  (µs)", ylabel=r"$|E|^2$  (arb.)")
style_ax(ax_heat,  xlabel="time  (µs)", ylabel="x  (mm)")
style_ax(ax_phase, xlabel="time  (µs)", ylabel="x  (mm)")
if is_3d:
    style_ax(ax_trans, xlabel="y  (mm)", ylabel="z  (mm)")

ax_radio_bg.set_facecolor(PANEL)
ax_radio_bg.axis("off")
ax_radio_bg.set_title(r"Heatmap $m_F$", color=TC, fontsize=7.5, pad=3)

# Row 2: |E|² heatmap | excited coherence magnitude | excited coherence phase
r2_h = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[2],
                                        width_ratios=[1, 1.2, 1.2], wspace=0.42)
ax_E_heat   = fig.add_subplot(r2_h[0])
ax_ge_heat  = fig.add_subplot(r2_h[1])
ax_ge_phase = fig.add_subplot(r2_h[2])

style_ax(ax_E_heat,   xlabel="time  (µs)", ylabel="x  (mm)")
style_ax(ax_ge_heat,  xlabel="time  (µs)", ylabel="x  (mm)")
style_ax(ax_ge_phase, xlabel="time  (µs)", ylabel="x  (mm)")

# Row 3: time slider + optional x-position slider (3D)
if is_3d:
    r3 = gridspec.GridSpecFromSubplotSpec(
        2, 3, subplot_spec=outer[3],
        width_ratios=[0.04, 5, 0.04], wspace=0.01, hspace=1.4)
    ax_sl   = fig.add_subplot(r3[0, 1])
    ax_sl_x = fig.add_subplot(r3[1, 1])
else:
    r3 = gridspec.GridSpecFromSubplotSpec(
        1, 3, subplot_spec=outer[3],
        width_ratios=[0.04, 5, 0.04], wspace=0.01)
    ax_sl = fig.add_subplot(r3[1])

ax_sl.set_facecolor(PANEL)
slider = Slider(ax=ax_sl, label="time  (µs)",
                valmin=t_us.min(), valmax=t_us.max(), valinit=t_us[0],
                color="#4C9BE8", initcolor="#4C9BE8", track_color="#22223A")
for attr in ("label", "valtext"):
    getattr(slider, attr).set_color(TC)
    getattr(slider, attr).set_fontsize(9)

if is_3d:
    ax_sl_x.set_facecolor(PANEL)
    slider_x = Slider(ax=ax_sl_x, label="x  (mm)",
                      valmin=x_mm.min(), valmax=x_mm.max(), valinit=x_mm[x_ci],
                      color="#55C464", initcolor="#55C464", track_color="#22223A")
    for attr in ("label", "valtext"):
        getattr(slider_x, attr).set_color(TC)
        getattr(slider_x, attr).set_fontsize(9)

# ── global titles ─────────────────────────────────────────────────────────────
mode_str = "3D — y,z averaged for x-profiles" if is_3d else "1D"
fig.text(0.5, 0.975,
         f"Gradient Echo Memory — XMDS2 Simulation (EIT)  [{mode_str}]",
         ha="center", color="#EEEEFF", fontsize=13, fontweight="bold")
time_annot = fig.text(0.5, 0.948, f"t = {t_us[0]:.4f} µs",
                      ha="center", color="#88BBFF", fontsize=10)

ax_main.set_title(
    "Spatial profiles at selected time" + (" (y,z averaged)" if is_3d else ""),
    color="#AABBFF", fontsize=8.5, pad=4)
ax_io.set_title(r"Probe intensity: input ($x=0$) & output ($x=x_{max}$)",
                color="#AABBFF", fontsize=8.5, pad=4)
ax_heat.set_title(r"Spin-wave $|\varphi_{gs}|$ ($m_F=1$) — $x$–$t$ heatmap",
                  color="#AABBFF", fontsize=8.5, pad=4)
ax_phase.set_title(r"Spin-wave phase ($m_F=1$) — $x$–$t$ heatmap",
                   color="#AABBFF", fontsize=8.5, pad=4)
ax_E_heat.set_title(r"Probe intensity $|E|^2$ — $x$–$t$ heatmap",
                    color="#AABBFF", fontsize=8.5, pad=4)
ax_ge_heat.set_title(r"Excited coherence $|\varphi_{ge}|$ ($m_F=1$) — $x$–$t$ heatmap",
                     color="#AABBFF", fontsize=8.5, pad=4)
ax_ge_phase.set_title(r"Excited coherence phase ($m_F=1$) — $x$–$t$ heatmap",
                      color="#AABBFF", fontsize=8.5, pad=4)
if is_3d:
    ax_trans.set_title(rf"ctrl  $y$–$z$ at $x={x_mm[x_ci]:.2f}$ mm",
                       color="#AABBFF", fontsize=8.5, pad=4)

# ── checkboxes ────────────────────────────────────────────────────────────────
ax_ctrl.set_title("Show / hide", color=TC, fontsize=8, pad=4)
ax_ctrl.axis("off")

check_keys    = list(DATA_MAP.keys())
check_labels  = [LABELS[k] for k in check_keys]
default_active = [k in ("E_int", "phi_ge2", "phi_gs2") for k in check_keys]

p = ax_ctrl.get_position()
chk_ax = fig.add_axes([p.x0 + 0.003, p.y0 + 0.01,
                        p.width - 0.005, p.height - 0.02], facecolor=PANEL)
check = CheckButtons(chk_ax, check_labels, default_active)
for lbl in check.labels:
    lbl.set_color(TC); lbl.set_fontsize(7.0)
for rect in check.rectangles:
    rect.set_facecolor("#22223A"); rect.set_edgecolor("#555588")

# ── mF radio buttons ──────────────────────────────────────────────────────────
p_rb = ax_radio_bg.get_position()
rb_ax = fig.add_axes([p_rb.x0, p_rb.y0 + p_rb.height * 0.15,
                       p_rb.width, p_rb.height * 0.70], facecolor=PANEL)
radio_heat = RadioButtons(rb_ax, HEATMAP_LABELS, active=0, activecolor="#4C9BE8")
for lbl in radio_heat.labels:
    lbl.set_color(TC); lbl.set_fontsize(8)

# ── spatial profile lines ─────────────────────────────────────────────────────
line_objs = {}
for key in check_keys:
    ax = ax_right if key in RIGHT_KEYS else ax_main
    (ln,) = ax.plot(x_mm, DATA_MAP[key][0, :],
                    color=COLORS[key],
                    lw=1.6 if key in LEFT_KEYS else 1.0,
                    alpha=0.9, label=LABELS[key],
                    visible=default_active[check_keys.index(key)])
    line_objs[key] = ln

ax_main.set_xlim(x_mm.min(), x_mm.max())

if is_3d:
    xline_main = ax_main.axvline(x_mm[x_ci], color="#55FF55",
                                 lw=1.0, ls=":", alpha=0.7)

# ── probe I/O traces ──────────────────────────────────────────────────────────
ax_io.plot(t_us, E_in,  color="#4C9BE8", lw=1.4, label=r"$x=0$ (input)")
ax_io.plot(t_us, E_out, color="#E8734C", lw=1.4, label=r"$x=x_{max}$ (output)")
ax_io.set_xlim(t_us.min(), t_us.max())
ax_io.legend(fontsize=7.5, facecolor=BG, edgecolor=SPINE, labelcolor=TC)
vline_io = ax_io.axvline(t_us[0], color="#FFFF88", lw=1.1, ls="--", alpha=0.85)

# ── spin-wave heatmaps (row 1, initially mF=1) ────────────────────────────────
_gs_mag0, _gs_phase0 = HEATMAP_OPTIONS[0][1], HEATMAP_OPTIONS[0][2]
norm_h0, is_log0 = _make_norm(_gs_mag0)

im = ax_heat.imshow(
    _gs_mag0.T, origin="lower", aspect="auto",
    extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
    cmap="inferno", norm=norm_h0, interpolation="bilinear",
)
cbar = fig.colorbar(im, ax=ax_heat, pad=0.02, fraction=0.035)
cbar.set_label(r"$|\varphi_{gs}|$  (arb., log)" if is_log0 else r"$|\varphi_{gs}|$  (arb.)",
               color=TC, fontsize=8)
cbar.ax.tick_params(colors=TC, labelsize=7)
plt.setp(cbar.ax.get_yticklabels(), color=TC)
vline_heat = ax_heat.axvline(t_us[0], color="#FFFF88", lw=1.1, ls="--", alpha=0.85)

im_phase = ax_phase.imshow(
    _mask_phase(_gs_mag0, _gs_phase0).T, origin="lower", aspect="auto",
    extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
    cmap="hsv", vmin=-np.pi, vmax=np.pi, interpolation="bilinear",
)
cbar_phase = fig.colorbar(im_phase, ax=ax_phase, pad=0.02, fraction=0.035)
cbar_phase.set_label(r"phase  (rad)", color=TC, fontsize=8)
cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar_phase.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
cbar_phase.ax.tick_params(colors=TC, labelsize=7)
plt.setp(cbar_phase.ax.get_yticklabels(), color=TC)
vline_phase = ax_phase.axvline(t_us[0], color="#FFFF88", lw=1.1, ls="--", alpha=0.85)

# ── transverse ctrl panel (3D only) ───────────────────────────────────────────
if is_3d:
    trans_vmax = ctrl_trans_3d.max()
    im_trans = ax_trans.imshow(
        ctrl_trans_3d[0, x_ci, :, :],
        origin="lower", aspect="auto",
        extent=[y_mm.min(), y_mm.max(), z_mm.min(), z_mm.max()],
        cmap="magma", vmin=0, vmax=trans_vmax if trans_vmax > 0 else 1,
        interpolation="bilinear",
    )
    cbar_trans = fig.colorbar(im_trans, ax=ax_trans, pad=0.02, fraction=0.035)
    cbar_trans.set_label(r"ctrl  (arb.)", color=TC, fontsize=8)
    cbar_trans.ax.tick_params(colors=TC, labelsize=7)
    plt.setp(cbar_trans.ax.get_yticklabels(), color=TC)

# ── row 2: |E|² heatmap + excited coherence mag + phase (initially mF=1) ─────
E_heat_pos = E_int[E_int > 0]
E_heat_max = E_int.max()
if E_heat_max > 0 and len(E_heat_pos) > 0 and (E_heat_max / E_heat_pos.min()) > 1e3:
    norm_E = LogNorm(vmin=max(E_heat_pos.min(), E_heat_max * 1e-6), vmax=E_heat_max)
    cbar_E_lbl = r"$|E|^2$  (arb., log)"
else:
    norm_E = Normalize(vmin=0, vmax=E_heat_max if E_heat_max > 0 else 1)
    cbar_E_lbl = r"$|E|^2$  (arb.)"

im_E = ax_E_heat.imshow(
    E_int.T, origin="lower", aspect="auto",
    extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
    cmap="viridis", norm=norm_E, interpolation="bilinear",
)
cbar_E = fig.colorbar(im_E, ax=ax_E_heat, pad=0.02, fraction=0.035)
cbar_E.set_label(cbar_E_lbl, color=TC, fontsize=8)
cbar_E.ax.tick_params(colors=TC, labelsize=7)
plt.setp(cbar_E.ax.get_yticklabels(), color=TC)
vline_E_heat = ax_E_heat.axvline(t_us[0], color="#FFFF88", lw=1.1, ls="--", alpha=0.85)

_ge_mag0, _ge_phase0 = HEATMAP_OPTIONS[0][3], HEATMAP_OPTIONS[0][4]
norm_ge0, is_ge_log0 = _make_norm(_ge_mag0)

im_ge = ax_ge_heat.imshow(
    _ge_mag0.T, origin="lower", aspect="auto",
    extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
    cmap="inferno", norm=norm_ge0, interpolation="bilinear",
)
cbar_ge = fig.colorbar(im_ge, ax=ax_ge_heat, pad=0.02, fraction=0.035)
cbar_ge.set_label(r"$|\varphi_{ge}|$  (arb., log)" if is_ge_log0 else r"$|\varphi_{ge}|$  (arb.)",
                  color=TC, fontsize=8)
cbar_ge.ax.tick_params(colors=TC, labelsize=7)
plt.setp(cbar_ge.ax.get_yticklabels(), color=TC)
vline_ge_heat = ax_ge_heat.axvline(t_us[0], color="#FFFF88", lw=1.1, ls="--", alpha=0.85)

im_ge_phase = ax_ge_phase.imshow(
    _mask_phase(_ge_mag0, _ge_phase0).T, origin="lower", aspect="auto",
    extent=[t_us.min(), t_us.max(), x_mm.min(), x_mm.max()],
    cmap="hsv", vmin=-np.pi, vmax=np.pi, interpolation="bilinear",
)
cbar_ge_phase = fig.colorbar(im_ge_phase, ax=ax_ge_phase, pad=0.02, fraction=0.035)
cbar_ge_phase.set_label(r"phase  (rad)", color=TC, fontsize=8)
cbar_ge_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
cbar_ge_phase.set_ticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
cbar_ge_phase.ax.tick_params(colors=TC, labelsize=7)
plt.setp(cbar_ge_phase.ax.get_yticklabels(), color=TC)
vline_ge_phase = ax_ge_phase.axvline(t_us[0], color="#FFFF88", lw=1.1, ls="--", alpha=0.85)

# ── legend helper ─────────────────────────────────────────────────────────────
def refresh_legend():
    leg = ax_main.get_legend()
    if leg:
        leg.remove()
    vis = [k for k in check_keys if line_objs[k].get_visible()]
    if vis:
        ax_main.legend([line_objs[k] for k in vis], [LABELS[k] for k in vis],
                       loc="upper right", fontsize=7,
                       facecolor=BG, edgecolor=SPINE,
                       labelcolor=TC, framealpha=0.85)

refresh_legend()

# ── autoscale spatial ─────────────────────────────────────────────────────────
def autoscale_spatial():
    vis_l = [k for k in LEFT_KEYS  if line_objs[k].get_visible()]
    vis_r = [k for k in RIGHT_KEYS if line_objs[k].get_visible()]
    if vis_l:
        ylo = min(line_objs[k].get_ydata().min() for k in vis_l)
        yhi = max(line_objs[k].get_ydata().max() for k in vis_l)
        pad = (yhi - ylo) * 0.08 if yhi != ylo else 1e-30
        ax_main.set_ylim(ylo - pad, yhi + pad)
    if vis_r:
        ylo = min(line_objs[k].get_ydata().min() for k in vis_r)
        yhi = max(line_objs[k].get_ydata().max() for k in vis_r)
        pad = (yhi - ylo) * 0.08 if yhi != ylo else 1e-30
        ax_right.set_ylim(ylo - pad, yhi + pad)

autoscale_spatial()

# ── x-index helper (3D) ───────────────────────────────────────────────────────
if is_3d:
    def x_idx(x_val_mm):
        return int(np.argmin(np.abs(x_mm - x_val_mm)))

# ── mF heatmap switch callback ────────────────────────────────────────────────
def switch_heatmap(label):
    idx = HEATMAP_LABELS.index(label)
    gs_mag, gs_phase, ge_mag, ge_phase = HEATMAP_OPTIONS[idx][1:]

    norm_gs, is_log_gs = _make_norm(gs_mag)
    im.set_data(gs_mag.T)
    im.set_norm(norm_gs)
    cbar.update_normal(im)
    cbar.set_label(r"$|\varphi_{gs}|$  (arb., log)" if is_log_gs else r"$|\varphi_{gs}|$  (arb.)",
                   color=TC, fontsize=8)
    im_phase.set_data(_mask_phase(gs_mag, gs_phase).T)
    ax_heat.set_title(r"Spin-wave $|\varphi_{gs}|$ (" + label + r") — $x$–$t$ heatmap",
                      color="#AABBFF", fontsize=8.5, pad=4)
    ax_phase.set_title(r"Spin-wave phase (" + label + r") — $x$–$t$ heatmap",
                       color="#AABBFF", fontsize=8.5, pad=4)

    norm_ge, is_log_ge = _make_norm(ge_mag)
    im_ge.set_data(ge_mag.T)
    im_ge.set_norm(norm_ge)
    cbar_ge.update_normal(im_ge)
    cbar_ge.set_label(r"$|\varphi_{ge}|$  (arb., log)" if is_log_ge else r"$|\varphi_{ge}|$  (arb.)",
                      color=TC, fontsize=8)
    im_ge_phase.set_data(_mask_phase(ge_mag, ge_phase).T)
    ax_ge_heat.set_title(r"Excited coherence $|\varphi_{ge}|$ (" + label + r") — $x$–$t$ heatmap",
                         color="#AABBFF", fontsize=8.5, pad=4)
    ax_ge_phase.set_title(r"Excited coherence phase (" + label + r") — $x$–$t$ heatmap",
                          color="#AABBFF", fontsize=8.5, pad=4)
    fig.canvas.draw_idle()

radio_heat.on_clicked(switch_heatmap)

# ── slider callbacks ──────────────────────────────────────────────────────────
def update(val):
    ti    = t_idx(slider.val)
    t_now = t_us[ti]
    time_annot.set_text(f"t = {t_now:.4f} µs")
    for key, ln in line_objs.items():
        ln.set_ydata(DATA_MAP[key][ti, :])
    autoscale_spatial()
    vline_io      .set_xdata([t_now, t_now])
    vline_heat    .set_xdata([t_now, t_now])
    vline_phase   .set_xdata([t_now, t_now])
    vline_E_heat  .set_xdata([t_now, t_now])
    vline_ge_heat .set_xdata([t_now, t_now])
    vline_ge_phase.set_xdata([t_now, t_now])
    if is_3d:
        xi = x_idx(slider_x.val)
        im_trans.set_data(ctrl_trans_3d[ti, xi, :, :])
        ax_trans.set_title(
            rf"ctrl  $y$–$z$ at $x={x_mm[xi]:.2f}$ mm, $t={t_now:.2f}$ µs",
            color="#AABBFF", fontsize=8.5, pad=4)
    fig.canvas.draw_idle()

slider.on_changed(update)

if is_3d:
    def update_x(val):
        ti    = t_idx(slider.val)
        xi    = x_idx(slider_x.val)
        t_now = t_us[ti]
        im_trans.set_data(ctrl_trans_3d[ti, xi, :, :])
        xline_main.set_xdata([x_mm[xi], x_mm[xi]])
        ax_trans.set_title(
            rf"ctrl  $y$–$z$ at $x={x_mm[xi]:.2f}$ mm, $t={t_now:.2f}$ µs",
            color="#AABBFF", fontsize=8.5, pad=4)
        fig.canvas.draw_idle()

    slider_x.on_changed(update_x)

def toggle(label):
    key = check_keys[check_labels.index(label)]
    line_objs[key].set_visible(not line_objs[key].get_visible())
    autoscale_spatial()
    refresh_legend()
    fig.canvas.draw_idle()

check.on_clicked(toggle)

plt.show()
