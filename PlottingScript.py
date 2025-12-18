#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

# -------------------- CONFIG --------------------
DATA = "RYH1053C_La4Ni3O10_NaI_800C.txt"           
OUT_BASENAME = "RYH1053C_La4Ni3O10_NaI_800C_pub"     
XLIM = (5.0, 60.0)                                 

# Text in upper right
LABEL_TEXT    = r"NaI Flux 800$^\circ$C"
LABEL_FONTSIZE = 9
LABEL_TEXT5    = r"Nominal La$_4$Ni$_3$O$_{10}$"
LABEL5_FONTSIZE = 9
LABEL_TEXT2    = r"Avg RP Ni Valence = 2.71"
LABEL2_FONTSIZE = 7
LABEL_TEXT3    = r"R$_{wp} = 6.40\%$"
LABEL3_FONTSIZE = 7

# -------------------- FRAME / SPINE GEOMETRY --------------------

SPINE_TOP = 1.20          
RIGHT_SPINE_BOTTOM = 0.00   

# -------------------- Phase Ticks --------------------
PUB_COLORS = {
    "black":  "#000000",  # La2NiO4
    "orange": "#ff7f0e",  # La3Ni2O7
    "blue":   "#17becf",  # La4Ni3O10
    "purple": "#8a2be2",  # LaNiO3
    "green":  "#2ca02c",  # NiO
    "red":    "#d62728",  # LaOCl
    "pink":   "#e377c2",  # La2O3
    "brown":  "#8c564b",  # La(OH)3
}

# (inp file, legend string, color)
TICKS_SPECS = [
    ("RYH1053C_La2NiO4_NaI_800C.inp",      r"La$_2$NiO$_{4}$ $28.9\%$",     PUB_COLORS["black"]),
    #("RYH1074C_La3Ni2O6.92_NaCl_830C.inp",      r"La$_3$Ni$_2$O$_{6.92}$ $17.4\%$",     PUB_COLORS["orange"]),
    #("RYH1005C_La4Ni3O10_NaBr_800C.inp",      r"La$_4$Ni$_3$O$_{10}$  $31.3\%$",     PUB_COLORS["blue"]),
    ("RYH1053C_LaNiO3_NaI_800C.inp",         r"LaNiO$_3$ $71.1\%$",            PUB_COLORS["purple"]),
   # ("RYH1053C_NiO_NaI_800C.inp",            r"NiO $5.0\%$",                PUB_COLORS["green"]),
    #("RYH1009C_LaOCl_NaCl_830C.inp",      r"LaOCl $1.3\%$", PUB_COLORS["red"]),
    #("RYH1030F_La2O3_KBr_800C.inp",            r"La$_2$O$_3$ $5.43\%$",                PUB_COLORS["pink"]),
     #   ("RYH1030F_La(OH)3_KBr_800C.inp",            r"La(OH)$_3$ $2.82\%$",                PUB_COLORS["brown"]),

]


# -------------------- Layout  --------------------
GAP_MAIN_TO_TICKS = 0.08
TICK_LENGTH_FRAC  = 0.055
TICKS_ROW_SPACING_MULT = 1.8
GAP_TICKS_TO_DIFF = 0.10     
TOP_HEADROOM_FRAC = 0.22     
BOTTOM_PAD_FRAC   = 0.06     
TICKS_RAISE_FRAC  = 0.00
DIFF_BELOW_TICKS_GAP_FRAC = 0.06  

# -------------------- Rendering --------------------
FIGSIZE     = (3.9, 3.2)
POINT_SIZE  = 7.0
YOBS_COLOR  = "black"
YCALC_COLOR = "#e2008a"
DIFF_COLOR  = "#1f77b4"

# -------------------- Style --------------------
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.linewidth": 1.1,
    "axes.labelsize": 10.5,
    "font.size": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# -------------------- Readers --------------------
def load_lebail_csv(path):
    arr = np.genfromtxt(path, delimiter=",", comments="'", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("No numeric rows parsed from %s" % path)

    ncols = arr.shape[1]
    if ncols >= 4:
        x, yobs, ycalc, ydiff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        if not np.isfinite(ydiff).all():
            ydiff = yobs - ycalc
    elif ncols == 3:
        x, yobs, ycalc = arr[:, 0], arr[:, 1], arr[:, 2]
        ydiff = yobs - ycalc
    else:
        raise ValueError("%s: expected 3–4 numeric columns; found %d." % (path, ncols))

    used = np.column_stack([x, yobs, ycalc, ydiff])
    mask = np.isfinite(used).all(axis=1)
    x, yobs, ycalc, ydiff = x[mask], yobs[mask], ycalc[mask], ydiff[mask]
    print("[parse] %s: %d rows parsed (x, Yobs, Ycalc, Diff[true])" % (Path(path).name, x.size))
    return x, yobs, ycalc, ydiff

def parse_twotheta_from_inp(text):
    lines = text.splitlines()

    # A) lines starting with "hkl_m_d_th2"
    valsA = []
    for s in lines:
        ss = s.strip().replace("@", " ")
        if ss.lower().startswith("hkl_m_d_th2"):
            parts = ss.split()
            if len(parts) >= 7:
                try:
                    valsA.append(float(parts[6]))
                except ValueError:
                    pass
    if valsA:
        return np.sort(np.array(valsA))

    # B) load { ... } block style
    valsB, in_block = [], False
    for raw in lines:
        s = raw.strip().replace("@", " ")
        low = s.lower()
        if low.startswith("load") and "hkl_m_d_th2" in low:
            in_block = True
            continue
        if in_block and s == "}":
            break
        if in_block:
            parts = s.split()
            if len(parts) >= 6:
                try:
                    valsB.append(float(parts[5]))
                except ValueError:
                    nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s)
                    if len(nums) >= 2:
                        try:
                            valsB.append(float(nums[-2]))
                        except ValueError:
                            pass
    if valsB:
        return np.sort(np.array(valsB))

    if lines:
        header = re.split(r"[,\s]+", lines[0].strip().lower())
        for key in ("2theta", "th2", "two_theta", "two-theta"):
            if key in header:
                idx = header.index(key)
                valsC = []
                for ln in lines[1:]:
                    parts = re.split(r"[,\s]+", ln.strip())
                    if len(parts) > idx:
                        try:
                            valsC.append(float(parts[idx]))
                        except ValueError:
                            pass
                if valsC:
                    return np.sort(np.array(valsC))

    return np.array([])

def load_tick_twotheta(path, th2_min=None, th2_max=None):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    tt = parse_twotheta_from_inp(txt)
    if th2_min is not None:
        tt = tt[tt >= th2_min]
    if th2_max is not None:
        tt = tt[tt <= th2_max]
    tt = np.sort(tt)
    print("[ticks] %s: %d reflections within %.1f–%.1f°"
          % (Path(path).name, tt.size, XLIM[0], XLIM[1]))
    return tt

# -------------------- Plot --------------------
def plot_true_diff_with_ticks(
    data_file,
    out_basename,
    xlim,
    ticks_specs,
    gap_main_to_ticks=GAP_MAIN_TO_TICKS,
    tick_length_frac=TICK_LENGTH_FRAC,
    gap_ticks_to_diff=GAP_TICKS_TO_DIFF,
    top_headroom_frac=TOP_HEADROOM_FRAC,
    ticks_raise_frac=TICKS_RAISE_FRAC,
    bottom_pad_frac=BOTTOM_PAD_FRAC,
    figsize=FIGSIZE,
):
    # Load & crop to range
    th2, yobs, ycalc, ydiff_true = load_lebail_csv(data_file)
    m = (th2 >= xlim[0]) & (th2 <= xlim[1])
    th2, yobs, ycalc, ydiff_true = th2[m], yobs[m], ycalc[m], ydiff_true[m]

    # Geometry
    yr = float(yobs.max() - yobs.min()) or 1.0
    tick_len = tick_length_frac * yr
    tick_y0_base = yobs.min() - gap_main_to_ticks * yr - tick_len
    tick_y0 = tick_y0_base + ticks_raise_frac * yr

    # Figure / axes
    fig, ax = plt.subplots(figsize=figsize)

    # tick behavior
    ax.tick_params(axis="both", which="both", direction="out")

    ax.tick_params(axis="y",
                   which="both",
                   left=False, right=False,  # no tick marks
                   labelleft=False, labelright=False,
                   length=0)

    # Text block in upper right 
    ax.text(0.55, 1.17, LABEL_TEXT,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.12, LABEL_TEXT5,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL5_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.06, LABEL_TEXT2,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL2_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.02, LABEL_TEXT3,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL3_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))

    # Main data 
    ax.scatter(th2, yobs,
               s=POINT_SIZE, c=YOBS_COLOR, lw=0,
               label=r"$Y_{\mathrm{obs}}$", zorder=3)
    ax.plot(th2, ycalc,
            lw=0.7, color=YCALC_COLOR,
            label=r"$Y_{\mathrm{calc}}$", zorder=5)

    # Phase ticks
    phase_handles, phase_labels = [], []
    lowest_row_index = None
    for i, (file, label, color) in enumerate(ticks_specs):
        tt = load_tick_twotheta(file, th2_min=xlim[0], th2_max=xlim[1])
        if tt.size:
            y0 = tick_y0 - i * (tick_len * TICKS_ROW_SPACING_MULT)
            ax.vlines(tt, y0, y0 + tick_len,
                      color=color, lw=1, alpha=0.98, zorder=1)
            phase_handles.append(
                Line2D([0], [0],
                       marker="|", linestyle="None",
                       markersize=4, markeredgewidth=1,
                       color=color, label=label)
            )
            phase_labels.append(label)
            if (lowest_row_index is None) or (i > lowest_row_index):
                lowest_row_index = i

    # Diff curve below phase ticks
    if lowest_row_index is None:
        diff_top_target = tick_y0 - gap_ticks_to_diff * yr
    else:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        diff_top_target = lowest_tick_y0 - DIFF_BELOW_TICKS_GAP_FRAC * yr

    d_plot = ydiff_true + (diff_top_target - float(np.max(ydiff_true)))
    ax.plot(th2, d_plot,
            lw=0.9, color=DIFF_COLOR,
            label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$", zorder=2)

    # Axes labels/limits
    ax.set_xlim(xlim)
    ax.set_xlabel(r"$2\theta\;(\mathrm{deg.})$")
    ax.set_ylabel(r"Intensity (a.\,u.)")

    # y-lims: include diff region and ticks
    if lowest_row_index is not None:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        lowest_tick_bottom = lowest_tick_y0 - 0.02 * yr
    else:
        lowest_tick_bottom = tick_y0 - 0.02 * yr

    y_min = min(float(np.min(d_plot)),
                float(yobs.min()) - 0.05 * yr,
                float(lowest_tick_bottom)) - BOTTOM_PAD_FRAC * yr
    y_max = float(yobs.max()) + TOP_HEADROOM_FRAC * yr
    ax.set_ylim(y_min, y_max)

    # x / y tick locator setup
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())

    n_maj = max(1, int(round((xlim[1] - xlim[0]) / 10.0)))
    ax.yaxis.set_major_locator(MultipleLocator((y_max - y_min) / n_maj))
    ax.yaxis.set_minor_locator(NullLocator())

    #bottom + left native spines
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend handles
    base_handles = [
        Line2D([0], [0], marker="o", lw=0, ms=1.0, color=YOBS_COLOR,
               label=r"$Y_{\mathrm{obs}}$"),
        Line2D([0], [0], lw=0.5, color=YCALC_COLOR,
               label=r"$Y_{\mathrm{calc}}$"),
        Line2D([0], [0], lw=0.8, color=DIFF_COLOR,
               label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$"),
    ]
    lw_left = ax.spines["left"].get_linewidth()

    # Left vertical extension above the plot
    ax.plot(
        [0, 0],
        [1.0, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Top horizontal spine across the plot/legend
    ax.plot(
        [0.0, 1.0],
        [SPINE_TOP, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Right spine 
    ax.plot(
        [1.0, 1.0],
        [RIGHT_SPINE_BOTTOM, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Legend
    ax.legend(
        base_handles + phase_handles,
        [h.get_label() for h in base_handles] + phase_labels,
        frameon=False,
        fontsize=8,
        handletextpad=0.6,
        borderpad=0.1,
        labelspacing=0.4,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.2),
        ncol=1,
    )

    # Save
    out = Path(out_basename)
    plt.savefig(out.with_suffix(".png"))
    plt.savefig(out.with_suffix(".pdf"))
    print("Saved:", out.with_suffix(".png"), "and", out.with_suffix(".pdf"))
    plt.show()

# -------------------- Run --------------------
if __name__ == "__main__":
    plot_true_diff_with_ticks(
        data_file=DATA,
        out_basename=OUT_BASENAME,
        xlim=XLIM,
        ticks_specs=TICKS_SPECS,
    )


# In[2]:


from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

# -------------------- CONFIG --------------------
DATA = "RYH1055C_La4Ni3O10_NaI_900C.txt"            # CSV: x, Yobs, Ycalc, [Diff]
OUT_BASENAME = "RRYH1055C_La4Ni3O10_NaI_900C_pub"     # PNG + PDF
XLIM = (5.0, 60.0)                                 # fixed 2θ window

# Text in upper right
LABEL_TEXT    = r"NaI Flux 900$^\circ$C"
LABEL_FONTSIZE = 9
LABEL_TEXT5    = r"Nominal La$_4$Ni$_3$O$_{10}$"
LABEL5_FONTSIZE = 9
LABEL_TEXT2    = r"Avg RP Ni Valence = 2.61"
LABEL2_FONTSIZE = 7
LABEL_TEXT3    = r"R$_{wp} = 6.67\%$"
LABEL3_FONTSIZE = 7

# -------------------- FRAME / SPINE GEOMETRY --------------------
# Axes coordinates: y=1.0 is the top of the main plotting area.
# We draw custom spines using these numbers.
SPINE_TOP = 1.20            # how high the frame sits above the plot (also top horizontal line y)
RIGHT_SPINE_BOTTOM = 0.00   # how low the right spine goes (0.00 = bottom of axes)

# -------------------- Publication palette for PHASE TICKS --------------------
PUB_COLORS = {
    "black":  "#000000",  # La2NiO4
    "orange": "#ff7f0e",  # La3Ni2O7
    "blue":   "#17becf",  # La4Ni3O10
    "purple": "#8a2be2",  # LaNiO3
    "green":  "#2ca02c",  # NiO
    "red":    "#d62728",  # LaOCl
    "pink":   "#e377c2",  # La2O3
    "brown":  "#8c564b",  # La(OH)3
}

# (inp file, legend string, color)
TICKS_SPECS = [
    ("RYH1055C_La2NiO4_NaI_900C.inp",      r"La$_2$NiO$_{4}$ $18.4\%$",     PUB_COLORS["black"]),
    #("RYH1074C_La3Ni2O6.92_NaCl_830C.inp",      r"La$_3$Ni$_2$O$_{6.92}$ $17.4\%$",     PUB_COLORS["orange"]),
    ("RYH1055C_La4Ni3O10_NaI_900C.inp",      r"La$_4$Ni$_3$O$_{10}$  $56.5\%$",     PUB_COLORS["blue"]),
    ("RYH1055C_LaNiO3_NaI_900C.inp",         r"LaNiO$_3$ $21.0\%$",            PUB_COLORS["purple"]),
    ("RYH1055C_NiO_NaI_900C.inp",            r"NiO $4.1\%$",                PUB_COLORS["green"]),
    #("RYH1009C_LaOCl_NaCl_830C.inp",      r"LaOCl $1.3\%$", PUB_COLORS["red"]),
    #("RYH1030F_La2O3_KBr_800C.inp",            r"La$_2$O$_3$ $5.43\%$",                PUB_COLORS["pink"]),
     #   ("RYH1030F_La(OH)3_KBr_800C.inp",            r"La(OH)$_3$ $2.82\%$",                PUB_COLORS["brown"]),

]


# -------------------- Layout (fractions of Yobs dynamic range) --------------------
GAP_MAIN_TO_TICKS = 0.08
TICK_LENGTH_FRAC  = 0.055
TICKS_ROW_SPACING_MULT = 1.8
GAP_TICKS_TO_DIFF = 0.10     # fallback gap ticks→diff
TOP_HEADROOM_FRAC = 0.22     # space above Yobs for data/text
BOTTOM_PAD_FRAC   = 0.06     # extra pad below diff
TICKS_RAISE_FRAC  = 0.00
DIFF_BELOW_TICKS_GAP_FRAC = 0.06  # gap between lowest tick row and diff curve

# -------------------- Rendering --------------------
FIGSIZE     = (3.9, 3.2)
POINT_SIZE  = 7.0
YOBS_COLOR  = "black"
YCALC_COLOR = "#e2008a"
DIFF_COLOR  = "#1f77b4"

# -------------------- Style (LaTeX fonts) --------------------
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.linewidth": 1.1,
    "axes.labelsize": 10.5,
    "font.size": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# -------------------- Readers --------------------
def load_lebail_csv(path):
    arr = np.genfromtxt(path, delimiter=",", comments="'", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("No numeric rows parsed from %s" % path)

    ncols = arr.shape[1]
    if ncols >= 4:
        x, yobs, ycalc, ydiff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        if not np.isfinite(ydiff).all():
            ydiff = yobs - ycalc
    elif ncols == 3:
        x, yobs, ycalc = arr[:, 0], arr[:, 1], arr[:, 2]
        ydiff = yobs - ycalc
    else:
        raise ValueError("%s: expected 3–4 numeric columns; found %d." % (path, ncols))

    used = np.column_stack([x, yobs, ycalc, ydiff])
    mask = np.isfinite(used).all(axis=1)
    x, yobs, ycalc, ydiff = x[mask], yobs[mask], ycalc[mask], ydiff[mask]
    print("[parse] %s: %d rows parsed (x, Yobs, Ycalc, Diff[true])" % (Path(path).name, x.size))
    return x, yobs, ycalc, ydiff

def parse_twotheta_from_inp(text):
    lines = text.splitlines()

    # A) lines starting with "hkl_m_d_th2"
    valsA = []
    for s in lines:
        ss = s.strip().replace("@", " ")
        if ss.lower().startswith("hkl_m_d_th2"):
            parts = ss.split()
            if len(parts) >= 7:
                try:
                    valsA.append(float(parts[6]))
                except ValueError:
                    pass
    if valsA:
        return np.sort(np.array(valsA))

    # B) load { ... } block style
    valsB, in_block = [], False
    for raw in lines:
        s = raw.strip().replace("@", " ")
        low = s.lower()
        if low.startswith("load") and "hkl_m_d_th2" in low:
            in_block = True
            continue
        if in_block and s == "}":
            break
        if in_block:
            parts = s.split()
            if len(parts) >= 6:
                try:
                    valsB.append(float(parts[5]))
                except ValueError:
                    nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s)
                    if len(nums) >= 2:
                        try:
                            valsB.append(float(nums[-2]))
                        except ValueError:
                            pass
    if valsB:
        return np.sort(np.array(valsB))

    # C) headered table with 2theta/th2
    if lines:
        header = re.split(r"[,\s]+", lines[0].strip().lower())
        for key in ("2theta", "th2", "two_theta", "two-theta"):
            if key in header:
                idx = header.index(key)
                valsC = []
                for ln in lines[1:]:
                    parts = re.split(r"[,\s]+", ln.strip())
                    if len(parts) > idx:
                        try:
                            valsC.append(float(parts[idx]))
                        except ValueError:
                            pass
                if valsC:
                    return np.sort(np.array(valsC))

    return np.array([])

def load_tick_twotheta(path, th2_min=None, th2_max=None):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    tt = parse_twotheta_from_inp(txt)
    if th2_min is not None:
        tt = tt[tt >= th2_min]
    if th2_max is not None:
        tt = tt[tt <= th2_max]
    tt = np.sort(tt)
    print("[ticks] %s: %d reflections within %.1f–%.1f°"
          % (Path(path).name, tt.size, XLIM[0], XLIM[1]))
    return tt

# -------------------- Plot --------------------
def plot_true_diff_with_ticks(
    data_file,
    out_basename,
    xlim,
    ticks_specs,
    gap_main_to_ticks=GAP_MAIN_TO_TICKS,
    tick_length_frac=TICK_LENGTH_FRAC,
    gap_ticks_to_diff=GAP_TICKS_TO_DIFF,
    top_headroom_frac=TOP_HEADROOM_FRAC,
    ticks_raise_frac=TICKS_RAISE_FRAC,
    bottom_pad_frac=BOTTOM_PAD_FRAC,
    figsize=FIGSIZE,
):
    # Load & crop to range
    th2, yobs, ycalc, ydiff_true = load_lebail_csv(data_file)
    m = (th2 >= xlim[0]) & (th2 <= xlim[1])
    th2, yobs, ycalc, ydiff_true = th2[m], yobs[m], ycalc[m], ydiff_true[m]

    # Geometry
    yr = float(yobs.max() - yobs.min()) or 1.0
    tick_len = tick_length_frac * yr
    tick_y0_base = yobs.min() - gap_main_to_ticks * yr - tick_len
    tick_y0 = tick_y0_base + ticks_raise_frac * yr

    # Figure / axes
    fig, ax = plt.subplots(figsize=figsize)

    # tick behavior
    ax.tick_params(axis="both", which="both", direction="out")

    # remove *all* y ticks and labels
    ax.tick_params(axis="y",
                   which="both",
                   left=False, right=False,  # no tick marks
                   labelleft=False, labelright=False,
                   length=0)

    # Text block in upper right (above plot)
    ax.text(0.55, 1.17, LABEL_TEXT,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.12, LABEL_TEXT5,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL5_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.06, LABEL_TEXT2,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL2_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.02, LABEL_TEXT3,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL3_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))

    # Main data (obs vs calc)
    ax.scatter(th2, yobs,
               s=POINT_SIZE, c=YOBS_COLOR, lw=0,
               label=r"$Y_{\mathrm{obs}}$", zorder=3)
    ax.plot(th2, ycalc,
            lw=0.7, color=YCALC_COLOR,
            label=r"$Y_{\mathrm{calc}}$", zorder=5)

    # Phase ticks
    phase_handles, phase_labels = [], []
    lowest_row_index = None
    for i, (file, label, color) in enumerate(ticks_specs):
        tt = load_tick_twotheta(file, th2_min=xlim[0], th2_max=xlim[1])
        if tt.size:
            y0 = tick_y0 - i * (tick_len * TICKS_ROW_SPACING_MULT)
            ax.vlines(tt, y0, y0 + tick_len,
                      color=color, lw=1, alpha=0.98, zorder=1)
            phase_handles.append(
                Line2D([0], [0],
                       marker="|", linestyle="None",
                       markersize=4, markeredgewidth=1,
                       color=color, label=label)
            )
            phase_labels.append(label)
            if (lowest_row_index is None) or (i > lowest_row_index):
                lowest_row_index = i

    # Diff curve below phase ticks
    if lowest_row_index is None:
        diff_top_target = tick_y0 - gap_ticks_to_diff * yr
    else:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        diff_top_target = lowest_tick_y0 - DIFF_BELOW_TICKS_GAP_FRAC * yr

    d_plot = ydiff_true + (diff_top_target - float(np.max(ydiff_true)))
    ax.plot(th2, d_plot,
            lw=0.9, color=DIFF_COLOR,
            label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$", zorder=2)

    # Axes labels/limits
    ax.set_xlim(xlim)
    ax.set_xlabel(r"$2\theta\;(\mathrm{deg.})$")
    ax.set_ylabel(r"Intensity (a.\,u.)")

    # y-lims: include diff region and ticks
    if lowest_row_index is not None:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        lowest_tick_bottom = lowest_tick_y0 - 0.02 * yr
    else:
        lowest_tick_bottom = tick_y0 - 0.02 * yr

    y_min = min(float(np.min(d_plot)),
                float(yobs.min()) - 0.05 * yr,
                float(lowest_tick_bottom)) - BOTTOM_PAD_FRAC * yr
    y_max = float(yobs.max()) + TOP_HEADROOM_FRAC * yr
    ax.set_ylim(y_min, y_max)

    # x / y tick locator setup
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())

    n_maj = max(1, int(round((xlim[1] - xlim[0]) / 10.0)))
    ax.yaxis.set_major_locator(MultipleLocator((y_max - y_min) / n_maj))
    ax.yaxis.set_minor_locator(NullLocator())

    # Show only bottom + left native spines
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend handles
    base_handles = [
        Line2D([0], [0], marker="o", lw=0, ms=1.0, color=YOBS_COLOR,
               label=r"$Y_{\mathrm{obs}}$"),
        Line2D([0], [0], lw=0.5, color=YCALC_COLOR,
               label=r"$Y_{\mathrm{calc}}$"),
        Line2D([0], [0], lw=0.8, color=DIFF_COLOR,
               label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$"),
    ]
    lw_left = ax.spines["left"].get_linewidth()

    # ---- CUSTOM FRAME SPINES (uses SPINE_TOP / RIGHT_SPINE_BOTTOM) ----
    # Left vertical extension above the plot
    ax.plot(
        [0, 0],
        [1.0, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Top horizontal spine across the plot/legend
    ax.plot(
        [0.0, 1.0],
        [SPINE_TOP, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Right spine (full height from RIGHT_SPINE_BOTTOM up to SPINE_TOP)
    ax.plot(
        [1.0, 1.0],
        [RIGHT_SPINE_BOTTOM, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Legend
    ax.legend(
        base_handles + phase_handles,
        [h.get_label() for h in base_handles] + phase_labels,
        frameon=False,
        fontsize=8,
        handletextpad=0.6,
        borderpad=0.1,
        labelspacing=0.4,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.2),
        ncol=1,
    )

    # Save
    out = Path(out_basename)
    plt.savefig(out.with_suffix(".png"))
    plt.savefig(out.with_suffix(".pdf"))
    print("Saved:", out.with_suffix(".png"), "and", out.with_suffix(".pdf"))
    plt.show()

# -------------------- Run --------------------
if __name__ == "__main__":
    plot_true_diff_with_ticks(
        data_file=DATA,
        out_basename=OUT_BASENAME,
        xlim=XLIM,
        ticks_specs=TICKS_SPECS,
    )


# In[3]:


from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

# -------------------- CONFIG --------------------
DATA = "RYH1070C_La4Ni3O10_NaI_1000C.txt"            # CSV: x, Yobs, Ycalc, [Diff]
OUT_BASENAME = "RYH1070C_La4Ni3O10_NaI_1000C_pub"     # PNG + PDF
XLIM = (5.0, 60.0)                                 # fixed 2θ window

# Text in upper right
LABEL_TEXT    = r"NaI Flux 1000$^\circ$C"
LABEL_FONTSIZE = 9
LABEL_TEXT5    = r"Nominal La$_4$Ni$_3$O$_{10}$"
LABEL5_FONTSIZE = 9
LABEL_TEXT2    = r"Avg RP Ni Valence = 2.60"
LABEL2_FONTSIZE = 7
LABEL_TEXT3    = r"R$_{wp} = 6.13\%$"
LABEL3_FONTSIZE = 7

# -------------------- FRAME / SPINE GEOMETRY --------------------
# Axes coordinates: y=1.0 is the top of the main plotting area.
# We draw custom spines using these numbers.
SPINE_TOP = 1.20            # how high the frame sits above the plot (also top horizontal line y)
RIGHT_SPINE_BOTTOM = 0.00   # how low the right spine goes (0.00 = bottom of axes)

# -------------------- Publication palette for PHASE TICKS --------------------
PUB_COLORS = {
    "black":  "#000000",  # La2NiO4
    "orange": "#ff7f0e",  # La3Ni2O7
    "blue":   "#17becf",  # La4Ni3O10
    "purple": "#8a2be2",  # LaNiO3
    "green":  "#2ca02c",  # NiO
    "red":    "#d62728",  # LaOCl
    "pink":   "#e377c2",  # La2O3
    "brown":  "#8c564b",  # La(OH)3
}

# (inp file, legend string, color)
TICKS_SPECS = [
    ("RYH1070C_La2NiO4_NaI_1000C.inp",      r"La$_2$NiO$_{4}$ $16.5\%$",     PUB_COLORS["black"]),
    #("RYH1074C_La3Ni2O6.92_NaCl_830C.inp",      r"La$_3$Ni$_2$O$_{6.92}$ $17.4\%$",     PUB_COLORS["orange"]),
    ("RYH1070C_La4Ni3O10_NaI_1000C.inp",      r"La$_4$Ni$_3$O$_{10}$  $69.8\%$",     PUB_COLORS["blue"]),
    ("RYH1070C_LaNiO3_NaI_1000C.inp",         r"LaNiO$_3$ $13.7\%$",            PUB_COLORS["purple"]),
    #("RYH1055C_NiO_NaI_900C.inp",            r"NiO $4.1\%$",                PUB_COLORS["green"]),
    #("RYH1009C_LaOCl_NaCl_830C.inp",      r"LaOCl $1.3\%$", PUB_COLORS["red"]),
    #("RYH1030F_La2O3_KBr_800C.inp",            r"La$_2$O$_3$ $5.43\%$",                PUB_COLORS["pink"]),
     #   ("RYH1030F_La(OH)3_KBr_800C.inp",            r"La(OH)$_3$ $2.82\%$",                PUB_COLORS["brown"]),

]


# -------------------- Layout (fractions of Yobs dynamic range) --------------------
GAP_MAIN_TO_TICKS = 0.08
TICK_LENGTH_FRAC  = 0.055
TICKS_ROW_SPACING_MULT = 1.8
GAP_TICKS_TO_DIFF = 0.10     # fallback gap ticks→diff
TOP_HEADROOM_FRAC = 0.22     # space above Yobs for data/text
BOTTOM_PAD_FRAC   = 0.06     # extra pad below diff
TICKS_RAISE_FRAC  = 0.00
DIFF_BELOW_TICKS_GAP_FRAC = 0.06  # gap between lowest tick row and diff curve

# -------------------- Rendering --------------------
FIGSIZE     = (3.9, 3.2)
POINT_SIZE  = 7.0
YOBS_COLOR  = "black"
YCALC_COLOR = "#e2008a"
DIFF_COLOR  = "#1f77b4"

# -------------------- Style (LaTeX fonts) --------------------
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.linewidth": 1.1,
    "axes.labelsize": 10.5,
    "font.size": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# -------------------- Readers --------------------
def load_lebail_csv(path):
    arr = np.genfromtxt(path, delimiter=",", comments="'", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("No numeric rows parsed from %s" % path)

    ncols = arr.shape[1]
    if ncols >= 4:
        x, yobs, ycalc, ydiff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        if not np.isfinite(ydiff).all():
            ydiff = yobs - ycalc
    elif ncols == 3:
        x, yobs, ycalc = arr[:, 0], arr[:, 1], arr[:, 2]
        ydiff = yobs - ycalc
    else:
        raise ValueError("%s: expected 3–4 numeric columns; found %d." % (path, ncols))

    used = np.column_stack([x, yobs, ycalc, ydiff])
    mask = np.isfinite(used).all(axis=1)
    x, yobs, ycalc, ydiff = x[mask], yobs[mask], ycalc[mask], ydiff[mask]
    print("[parse] %s: %d rows parsed (x, Yobs, Ycalc, Diff[true])" % (Path(path).name, x.size))
    return x, yobs, ycalc, ydiff

def parse_twotheta_from_inp(text):
    lines = text.splitlines()

    # A) lines starting with "hkl_m_d_th2"
    valsA = []
    for s in lines:
        ss = s.strip().replace("@", " ")
        if ss.lower().startswith("hkl_m_d_th2"):
            parts = ss.split()
            if len(parts) >= 7:
                try:
                    valsA.append(float(parts[6]))
                except ValueError:
                    pass
    if valsA:
        return np.sort(np.array(valsA))

    # B) load { ... } block style
    valsB, in_block = [], False
    for raw in lines:
        s = raw.strip().replace("@", " ")
        low = s.lower()
        if low.startswith("load") and "hkl_m_d_th2" in low:
            in_block = True
            continue
        if in_block and s == "}":
            break
        if in_block:
            parts = s.split()
            if len(parts) >= 6:
                try:
                    valsB.append(float(parts[5]))
                except ValueError:
                    nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s)
                    if len(nums) >= 2:
                        try:
                            valsB.append(float(nums[-2]))
                        except ValueError:
                            pass
    if valsB:
        return np.sort(np.array(valsB))

    # C) headered table with 2theta/th2
    if lines:
        header = re.split(r"[,\s]+", lines[0].strip().lower())
        for key in ("2theta", "th2", "two_theta", "two-theta"):
            if key in header:
                idx = header.index(key)
                valsC = []
                for ln in lines[1:]:
                    parts = re.split(r"[,\s]+", ln.strip())
                    if len(parts) > idx:
                        try:
                            valsC.append(float(parts[idx]))
                        except ValueError:
                            pass
                if valsC:
                    return np.sort(np.array(valsC))

    return np.array([])

def load_tick_twotheta(path, th2_min=None, th2_max=None):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    tt = parse_twotheta_from_inp(txt)
    if th2_min is not None:
        tt = tt[tt >= th2_min]
    if th2_max is not None:
        tt = tt[tt <= th2_max]
    tt = np.sort(tt)
    print("[ticks] %s: %d reflections within %.1f–%.1f°"
          % (Path(path).name, tt.size, XLIM[0], XLIM[1]))
    return tt

# -------------------- Plot --------------------
def plot_true_diff_with_ticks(
    data_file,
    out_basename,
    xlim,
    ticks_specs,
    gap_main_to_ticks=GAP_MAIN_TO_TICKS,
    tick_length_frac=TICK_LENGTH_FRAC,
    gap_ticks_to_diff=GAP_TICKS_TO_DIFF,
    top_headroom_frac=TOP_HEADROOM_FRAC,
    ticks_raise_frac=TICKS_RAISE_FRAC,
    bottom_pad_frac=BOTTOM_PAD_FRAC,
    figsize=FIGSIZE,
):
    # Load & crop to range
    th2, yobs, ycalc, ydiff_true = load_lebail_csv(data_file)
    m = (th2 >= xlim[0]) & (th2 <= xlim[1])
    th2, yobs, ycalc, ydiff_true = th2[m], yobs[m], ycalc[m], ydiff_true[m]

    # Geometry
    yr = float(yobs.max() - yobs.min()) or 1.0
    tick_len = tick_length_frac * yr
    tick_y0_base = yobs.min() - gap_main_to_ticks * yr - tick_len
    tick_y0 = tick_y0_base + ticks_raise_frac * yr

    # Figure / axes
    fig, ax = plt.subplots(figsize=figsize)

    # tick behavior
    ax.tick_params(axis="both", which="both", direction="out")

    # remove *all* y ticks and labels
    ax.tick_params(axis="y",
                   which="both",
                   left=False, right=False,  # no tick marks
                   labelleft=False, labelright=False,
                   length=0)

    # Text block in upper right (above plot)
    ax.text(0.55, 1.17, LABEL_TEXT,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.12, LABEL_TEXT5,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL5_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.06, LABEL_TEXT2,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL2_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.02, LABEL_TEXT3,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL3_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))

    # Main data (obs vs calc)
    ax.scatter(th2, yobs,
               s=POINT_SIZE, c=YOBS_COLOR, lw=0,
               label=r"$Y_{\mathrm{obs}}$", zorder=3)
    ax.plot(th2, ycalc,
            lw=0.7, color=YCALC_COLOR,
            label=r"$Y_{\mathrm{calc}}$", zorder=5)

    # Phase ticks
    phase_handles, phase_labels = [], []
    lowest_row_index = None
    for i, (file, label, color) in enumerate(ticks_specs):
        tt = load_tick_twotheta(file, th2_min=xlim[0], th2_max=xlim[1])
        if tt.size:
            y0 = tick_y0 - i * (tick_len * TICKS_ROW_SPACING_MULT)
            ax.vlines(tt, y0, y0 + tick_len,
                      color=color, lw=1, alpha=0.98, zorder=1)
            phase_handles.append(
                Line2D([0], [0],
                       marker="|", linestyle="None",
                       markersize=4, markeredgewidth=1,
                       color=color, label=label)
            )
            phase_labels.append(label)
            if (lowest_row_index is None) or (i > lowest_row_index):
                lowest_row_index = i

    # Diff curve below phase ticks
    if lowest_row_index is None:
        diff_top_target = tick_y0 - gap_ticks_to_diff * yr
    else:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        diff_top_target = lowest_tick_y0 - DIFF_BELOW_TICKS_GAP_FRAC * yr

    d_plot = ydiff_true + (diff_top_target - float(np.max(ydiff_true)))
    ax.plot(th2, d_plot,
            lw=0.9, color=DIFF_COLOR,
            label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$", zorder=2)

    # Axes labels/limits
    ax.set_xlim(xlim)
    ax.set_xlabel(r"$2\theta\;(\mathrm{deg.})$")
    ax.set_ylabel(r"Intensity (a.\,u.)")

    # y-lims: include diff region and ticks
    if lowest_row_index is not None:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        lowest_tick_bottom = lowest_tick_y0 - 0.02 * yr
    else:
        lowest_tick_bottom = tick_y0 - 0.02 * yr

    y_min = min(float(np.min(d_plot)),
                float(yobs.min()) - 0.05 * yr,
                float(lowest_tick_bottom)) - BOTTOM_PAD_FRAC * yr
    y_max = float(yobs.max()) + TOP_HEADROOM_FRAC * yr
    ax.set_ylim(y_min, y_max)

    # x / y tick locator setup
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())

    n_maj = max(1, int(round((xlim[1] - xlim[0]) / 10.0)))
    ax.yaxis.set_major_locator(MultipleLocator((y_max - y_min) / n_maj))
    ax.yaxis.set_minor_locator(NullLocator())

    # Show only bottom + left native spines
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend handles
    base_handles = [
        Line2D([0], [0], marker="o", lw=0, ms=1.0, color=YOBS_COLOR,
               label=r"$Y_{\mathrm{obs}}$"),
        Line2D([0], [0], lw=0.5, color=YCALC_COLOR,
               label=r"$Y_{\mathrm{calc}}$"),
        Line2D([0], [0], lw=0.8, color=DIFF_COLOR,
               label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$"),
    ]
    lw_left = ax.spines["left"].get_linewidth()

    # ---- CUSTOM FRAME SPINES (uses SPINE_TOP / RIGHT_SPINE_BOTTOM) ----
    # Left vertical extension above the plot
    ax.plot(
        [0, 0],
        [1.0, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Top horizontal spine across the plot/legend
    ax.plot(
        [0.0, 1.0],
        [SPINE_TOP, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Right spine (full height from RIGHT_SPINE_BOTTOM up to SPINE_TOP)
    ax.plot(
        [1.0, 1.0],
        [RIGHT_SPINE_BOTTOM, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Legend
    ax.legend(
        base_handles + phase_handles,
        [h.get_label() for h in base_handles] + phase_labels,
        frameon=False,
        fontsize=8,
        handletextpad=0.6,
        borderpad=0.1,
        labelspacing=0.4,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.2),
        ncol=1,
    )

    # Save
    out = Path(out_basename)
    plt.savefig(out.with_suffix(".png"))
    plt.savefig(out.with_suffix(".pdf"))
    print("Saved:", out.with_suffix(".png"), "and", out.with_suffix(".pdf"))
    plt.show()

# -------------------- Run --------------------
if __name__ == "__main__":
    plot_true_diff_with_ticks(
        data_file=DATA,
        out_basename=OUT_BASENAME,
        xlim=XLIM,
        ticks_specs=TICKS_SPECS,
    )


# In[4]:


from pathlib import Path
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, NullLocator

# -------------------- CONFIG --------------------
DATA = "RYH1075C_La4Ni3O10_NaI_1100C.txt"            # CSV: x, Yobs, Ycalc, [Diff]
OUT_BASENAME = "RYH1075C_La4Ni3O10_NaI_1100C_pub"     # PNG + PDF
XLIM = (5.0, 60.0)                                 # fixed 2θ window

# Text in upper right
LABEL_TEXT    = r"NaI Flux 1100$^\circ$C"
LABEL_FONTSIZE = 9
LABEL_TEXT5    = r"Nominal La$_4$Ni$_3$O$_{10}$"
LABEL5_FONTSIZE = 9
LABEL_TEXT2    = r"Avg RP Ni Valence = 2.0"
LABEL2_FONTSIZE = 7
LABEL_TEXT3    = r"R$_{wp} = 8.13\%$"
LABEL3_FONTSIZE = 7

# -------------------- FRAME / SPINE GEOMETRY --------------------
# Axes coordinates: y=1.0 is the top of the main plotting area.
# We draw custom spines using these numbers.
SPINE_TOP = 1.20            # how high the frame sits above the plot (also top horizontal line y)
RIGHT_SPINE_BOTTOM = 0.00   # how low the right spine goes (0.00 = bottom of axes)

# -------------------- Publication palette for PHASE TICKS --------------------
PUB_COLORS = {
    "black":  "#000000",  # La2NiO4
    "orange": "#ff7f0e",  # La3Ni2O7
    "blue":   "#17becf",  # La4Ni3O10
    "purple": "#8a2be2",  # LaNiO3
    "green":  "#2ca02c",  # NiO
    "red":    "#d62728",  # LaOCl
    "pink":   "#e377c2",  # La2O3
    "brown":  "#8c564b",  # La(OH)3
}

# (inp file, legend string, color)
TICKS_SPECS = [
    ("RYH1075C_La2NiO4_NaI_1100C.inp",      r"La$_2$NiO$_{4}$ $90.2\%$",     PUB_COLORS["black"]),
    #("RYH1074C_La3Ni2O6.92_NaCl_830C.inp",      r"La$_3$Ni$_2$O$_{6.92}$ $17.4\%$",     PUB_COLORS["orange"]),
   # ("RYH1075C_La4Ni3O10_NaI_1000C.inp",      r"La$_4$Ni$_3$O$_{10}$  $69.8\%$",     PUB_COLORS["blue"]),
    #("RYH1075C_LaNiO3_NaI_1000C.inp",         r"LaNiO$_3$ $13.7\%$",            PUB_COLORS["purple"]),
    ("RYH1075C_NiO_NaI_1100C.inp",            r"NiO $9.8\%$",                PUB_COLORS["green"]),
    #("RYH1009C_LaOCl_NaCl_830C.inp",      r"LaOCl $1.3\%$", PUB_COLORS["red"]),
    #("RYH1030F_La2O3_KBr_800C.inp",            r"La$_2$O$_3$ $5.43\%$",                PUB_COLORS["pink"]),
     #   ("RYH1030F_La(OH)3_KBr_800C.inp",            r"La(OH)$_3$ $2.82\%$",                PUB_COLORS["brown"]),

]


# -------------------- Layout (fractions of Yobs dynamic range) --------------------
GAP_MAIN_TO_TICKS = 0.08
TICK_LENGTH_FRAC  = 0.055
TICKS_ROW_SPACING_MULT = 1.8
GAP_TICKS_TO_DIFF = 0.10     # fallback gap ticks→diff
TOP_HEADROOM_FRAC = 0.22     # space above Yobs for data/text
BOTTOM_PAD_FRAC   = 0.06     # extra pad below diff
TICKS_RAISE_FRAC  = 0.00
DIFF_BELOW_TICKS_GAP_FRAC = 0.06  # gap between lowest tick row and diff curve

# -------------------- Rendering --------------------
FIGSIZE     = (3.9, 3.2)
POINT_SIZE  = 7.0
YOBS_COLOR  = "black"
YCALC_COLOR = "#e2008a"
DIFF_COLOR  = "#1f77b4"

# -------------------- Style (LaTeX fonts) --------------------
rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.latex.preamble": r"\usepackage{amsmath}",
    "axes.linewidth": 1.1,
    "axes.labelsize": 10.5,
    "font.size": 10,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.size": 4.0, "ytick.major.size": 4.0,
    "savefig.dpi": 600, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

# -------------------- Readers --------------------
def load_lebail_csv(path):
    arr = np.genfromtxt(path, delimiter=",", comments="'", dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.size == 0:
        raise ValueError("No numeric rows parsed from %s" % path)

    ncols = arr.shape[1]
    if ncols >= 4:
        x, yobs, ycalc, ydiff = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        if not np.isfinite(ydiff).all():
            ydiff = yobs - ycalc
    elif ncols == 3:
        x, yobs, ycalc = arr[:, 0], arr[:, 1], arr[:, 2]
        ydiff = yobs - ycalc
    else:
        raise ValueError("%s: expected 3–4 numeric columns; found %d." % (path, ncols))

    used = np.column_stack([x, yobs, ycalc, ydiff])
    mask = np.isfinite(used).all(axis=1)
    x, yobs, ycalc, ydiff = x[mask], yobs[mask], ycalc[mask], ydiff[mask]
    print("[parse] %s: %d rows parsed (x, Yobs, Ycalc, Diff[true])" % (Path(path).name, x.size))
    return x, yobs, ycalc, ydiff

def parse_twotheta_from_inp(text):
    lines = text.splitlines()

    # A) lines starting with "hkl_m_d_th2"
    valsA = []
    for s in lines:
        ss = s.strip().replace("@", " ")
        if ss.lower().startswith("hkl_m_d_th2"):
            parts = ss.split()
            if len(parts) >= 7:
                try:
                    valsA.append(float(parts[6]))
                except ValueError:
                    pass
    if valsA:
        return np.sort(np.array(valsA))

    # B) load { ... } block style
    valsB, in_block = [], False
    for raw in lines:
        s = raw.strip().replace("@", " ")
        low = s.lower()
        if low.startswith("load") and "hkl_m_d_th2" in low:
            in_block = True
            continue
        if in_block and s == "}":
            break
        if in_block:
            parts = s.split()
            if len(parts) >= 6:
                try:
                    valsB.append(float(parts[5]))
                except ValueError:
                    nums = re.findall(r'[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?', s)
                    if len(nums) >= 2:
                        try:
                            valsB.append(float(nums[-2]))
                        except ValueError:
                            pass
    if valsB:
        return np.sort(np.array(valsB))

    # C) headered table with 2theta/th2
    if lines:
        header = re.split(r"[,\s]+", lines[0].strip().lower())
        for key in ("2theta", "th2", "two_theta", "two-theta"):
            if key in header:
                idx = header.index(key)
                valsC = []
                for ln in lines[1:]:
                    parts = re.split(r"[,\s]+", ln.strip())
                    if len(parts) > idx:
                        try:
                            valsC.append(float(parts[idx]))
                        except ValueError:
                            pass
                if valsC:
                    return np.sort(np.array(valsC))

    return np.array([])

def load_tick_twotheta(path, th2_min=None, th2_max=None):
    txt = Path(path).read_text(encoding="utf-8", errors="ignore")
    tt = parse_twotheta_from_inp(txt)
    if th2_min is not None:
        tt = tt[tt >= th2_min]
    if th2_max is not None:
        tt = tt[tt <= th2_max]
    tt = np.sort(tt)
    print("[ticks] %s: %d reflections within %.1f–%.1f°"
          % (Path(path).name, tt.size, XLIM[0], XLIM[1]))
    return tt

# -------------------- Plot --------------------
def plot_true_diff_with_ticks(
    data_file,
    out_basename,
    xlim,
    ticks_specs,
    gap_main_to_ticks=GAP_MAIN_TO_TICKS,
    tick_length_frac=TICK_LENGTH_FRAC,
    gap_ticks_to_diff=GAP_TICKS_TO_DIFF,
    top_headroom_frac=TOP_HEADROOM_FRAC,
    ticks_raise_frac=TICKS_RAISE_FRAC,
    bottom_pad_frac=BOTTOM_PAD_FRAC,
    figsize=FIGSIZE,
):
    # Load & crop to range
    th2, yobs, ycalc, ydiff_true = load_lebail_csv(data_file)
    m = (th2 >= xlim[0]) & (th2 <= xlim[1])
    th2, yobs, ycalc, ydiff_true = th2[m], yobs[m], ycalc[m], ydiff_true[m]

    # Geometry
    yr = float(yobs.max() - yobs.min()) or 1.0
    tick_len = tick_length_frac * yr
    tick_y0_base = yobs.min() - gap_main_to_ticks * yr - tick_len
    tick_y0 = tick_y0_base + ticks_raise_frac * yr

    # Figure / axes
    fig, ax = plt.subplots(figsize=figsize)

    # tick behavior
    ax.tick_params(axis="both", which="both", direction="out")

    # remove *all* y ticks and labels
    ax.tick_params(axis="y",
                   which="both",
                   left=False, right=False,  # no tick marks
                   labelleft=False, labelright=False,
                   length=0)

    # Text block in upper right (above plot)
    ax.text(0.55, 1.17, LABEL_TEXT,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.12, LABEL_TEXT5,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL5_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.06, LABEL_TEXT2,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL2_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))
    ax.text(0.55, 1.02, LABEL_TEXT3,
            transform=ax.transAxes, ha="left", va="top",
            fontsize=LABEL3_FONTSIZE,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2))

    # Main data (obs vs calc)
    ax.scatter(th2, yobs,
               s=POINT_SIZE, c=YOBS_COLOR, lw=0,
               label=r"$Y_{\mathrm{obs}}$", zorder=3)
    ax.plot(th2, ycalc,
            lw=0.7, color=YCALC_COLOR,
            label=r"$Y_{\mathrm{calc}}$", zorder=5)

    # Phase ticks
    phase_handles, phase_labels = [], []
    lowest_row_index = None
    for i, (file, label, color) in enumerate(ticks_specs):
        tt = load_tick_twotheta(file, th2_min=xlim[0], th2_max=xlim[1])
        if tt.size:
            y0 = tick_y0 - i * (tick_len * TICKS_ROW_SPACING_MULT)
            ax.vlines(tt, y0, y0 + tick_len,
                      color=color, lw=1, alpha=0.98, zorder=1)
            phase_handles.append(
                Line2D([0], [0],
                       marker="|", linestyle="None",
                       markersize=4, markeredgewidth=1,
                       color=color, label=label)
            )
            phase_labels.append(label)
            if (lowest_row_index is None) or (i > lowest_row_index):
                lowest_row_index = i

    # Diff curve below phase ticks
    if lowest_row_index is None:
        diff_top_target = tick_y0 - gap_ticks_to_diff * yr
    else:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        diff_top_target = lowest_tick_y0 - DIFF_BELOW_TICKS_GAP_FRAC * yr

    d_plot = ydiff_true + (diff_top_target - float(np.max(ydiff_true)))
    ax.plot(th2, d_plot,
            lw=0.9, color=DIFF_COLOR,
            label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$", zorder=2)

    # Axes labels/limits
    ax.set_xlim(xlim)
    ax.set_xlabel(r"$2\theta\;(\mathrm{deg.})$")
    ax.set_ylabel(r"Intensity (a.\,u.)")

    # y-lims: include diff region and ticks
    if lowest_row_index is not None:
        lowest_tick_y0 = tick_y0 - lowest_row_index * (tick_len * TICKS_ROW_SPACING_MULT)
        lowest_tick_bottom = lowest_tick_y0 - 0.02 * yr
    else:
        lowest_tick_bottom = tick_y0 - 0.02 * yr

    y_min = min(float(np.min(d_plot)),
                float(yobs.min()) - 0.05 * yr,
                float(lowest_tick_bottom)) - BOTTOM_PAD_FRAC * yr
    y_max = float(yobs.max()) + TOP_HEADROOM_FRAC * yr
    ax.set_ylim(y_min, y_max)

    # x / y tick locator setup
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.xaxis.set_minor_locator(NullLocator())

    n_maj = max(1, int(round((xlim[1] - xlim[0]) / 10.0)))
    ax.yaxis.set_major_locator(MultipleLocator((y_max - y_min) / n_maj))
    ax.yaxis.set_minor_locator(NullLocator())

    # Show only bottom + left native spines
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_linewidth(rcParams["axes.linewidth"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend handles
    base_handles = [
        Line2D([0], [0], marker="o", lw=0, ms=1.0, color=YOBS_COLOR,
               label=r"$Y_{\mathrm{obs}}$"),
        Line2D([0], [0], lw=0.5, color=YCALC_COLOR,
               label=r"$Y_{\mathrm{calc}}$"),
        Line2D([0], [0], lw=0.8, color=DIFF_COLOR,
               label=r"$Y_{\mathrm{obs}}-Y_{\mathrm{calc}}$"),
    ]
    lw_left = ax.spines["left"].get_linewidth()

    # ---- CUSTOM FRAME SPINES (uses SPINE_TOP / RIGHT_SPINE_BOTTOM) ----
    # Left vertical extension above the plot
    ax.plot(
        [0, 0],
        [1.0, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Top horizontal spine across the plot/legend
    ax.plot(
        [0.0, 1.0],
        [SPINE_TOP, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Right spine (full height from RIGHT_SPINE_BOTTOM up to SPINE_TOP)
    ax.plot(
        [1.0, 1.0],
        [RIGHT_SPINE_BOTTOM, SPINE_TOP],
        transform=ax.transAxes,
        color="black", lw=lw_left,
        clip_on=False,
        solid_capstyle="butt",
    )

    # Legend
    ax.legend(
        base_handles + phase_handles,
        [h.get_label() for h in base_handles] + phase_labels,
        frameon=False,
        fontsize=8,
        handletextpad=0.6,
        borderpad=0.1,
        labelspacing=0.4,
        loc="upper left",
        bbox_to_anchor=(0.0, 1.2),
        ncol=1,
    )

    # Save
    out = Path(out_basename)
    plt.savefig(out.with_suffix(".png"))
    plt.savefig(out.with_suffix(".pdf"))
    print("Saved:", out.with_suffix(".png"), "and", out.with_suffix(".pdf"))
    plt.show()

# -------------------- Run --------------------
if __name__ == "__main__":
    plot_true_diff_with_ticks(
        data_file=DATA,
        out_basename=OUT_BASENAME,
        xlim=XLIM,
        ticks_specs=TICKS_SPECS,
    )


# In[ ]:




