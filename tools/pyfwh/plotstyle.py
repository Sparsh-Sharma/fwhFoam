"""Paper figure style for matplotlib (Times-matched, for the newtx body font).

Call ``use_paper()`` before plotting. The manuscript body is set in newtx
(Times); the figures therefore use a Times-compatible serif face (STIX, with
Times New Roman and DejaVu Serif as fallbacks) and STIX mathtext, at 8--9 pt,
so that the type in the figures matches the surrounding text at final size.

This sets type and axis appearance only; it does not change the colours chosen
in each figure. Modelled on the JFM house-style helper used in the
square-cylinder study (STIX ~ Times, maths in the same family, ticks inward on
all four sides, thin axes, no legend frame, fonts embedded as TrueType).
"""
import matplotlib as mpl


def use_paper():
    mpl.rcParams.update({
        # --- type: Times-compatible serif, maths in the same family ------
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.unicode_minus": False,   # use ASCII hyphen for a consistent minus
        # --- axes --------------------------------------------------------
        "axes.linewidth": 0.6,
        "lines.linewidth": 1.1,
        # --- ticks inward, all four sides --------------------------------
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        # --- legend ------------------------------------------------------
        "legend.frameon": False,
        # --- output: vector PDF, fonts embedded as TrueType --------------
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
