"""Shared figure style and drawing primitives.

Palette and canvas conventions are carried over from the group's previous submission so the two
papers read as a series. Everything is vector PDF with embedded TrueType fonts; nothing depends
on a matplotlib style file that might drift.

Measured and projected values must be visually separable. Use MEASURED_KW and PROJECTED_KW
rather than inventing a distinction per figure.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# --- Canvas -----------------------------------------------------------------

# IEEEtran's printed widths are about 3.5 in for one column and 7.16 in across
# both columns.  Figures are drawn at those physical sizes instead of on a large
# presentation canvas that LaTeX would shrink (along with all of its labels).
COLUMN_WIDTH = 3.50
PAGE_WIDTH = 7.16
FIGSIZE = (PAGE_WIDTH, 3.9)
DPI = 160

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "dejavusans",
        "pdf.fonttype": 42,  # embed TrueType, not Type 3
        "ps.fonttype": 42,
        "axes.linewidth": 0.9,
        "axes.edgecolor": "#666666",
        "axes.labelcolor": "#111111",
        "xtick.color": "#666666",
        "ytick.color": "#666666",
        "text.color": "#111111",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": 7.4,
        "axes.titlesize": 8.4,
        "axes.labelsize": 7.4,
        "xtick.labelsize": 6.8,
        "ytick.labelsize": 6.8,
        "legend.fontsize": 6.8,
        "hatch.linewidth": 0.55,
    }
)

# --- Palette ----------------------------------------------------------------

BLACK = "#111111"
GRAY = "#666666"
LIGHT_GRAY = "#F2F2F2"
BLUE = "#0B6393"
CHART_BLUE = "#3A8FD0"
PALE_BLUE = "#CEE2F0"
GREEN = "#3B7B2C"
CHART_GREEN = "#3D8748"
PALE_GREEN = "#D7E9D1"
TEAL = "#0B8D65"
PALE_TEAL = "#D8F0E7"
PURPLE = "#625184"
PALE_PURPLE = "#DAD0E8"
ORANGE = "#C86817"
AMBER = "#D5960F"
PALE_ORANGE = "#FDE3CA"
RED = "#C70602"
NAVY = "#142656"

# Role colours, used consistently across every figure in the paper.
SERVER = BLUE
SERVER_PALE = PALE_BLUE
CLIENT = TEAL
CLIENT_PALE = PALE_TEAL
FAILURE = RED

# Measured versus projected. Applied everywhere; never redefined locally.
MEASURED_KW = dict(edgecolor=BLACK, linewidth=1.4, alpha=1.0, hatch=None)
PROJECTED_KW = dict(edgecolor=GRAY, linewidth=1.0, alpha=0.55, hatch="///")


# --- Primitives -------------------------------------------------------------


def canvas(figsize=FIGSIZE):
    """Blank axes in 0..100 x 0..100 coordinates, for schematic diagrams."""
    fig, ax = plt.subplots(figsize=figsize, dpi=DPI)
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.02, top=0.98)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def save(fig, outdir: pathlib.Path, name: str) -> pathlib.Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def heading(ax, x, y, text, size=9, color=BLACK, ha="left", weight="bold"):
    ax.text(
        x, y, text, fontsize=size, color=color, ha=ha, va="center", fontweight=weight
    )


def box(ax, x, y, w, h, facecolor=LIGHT_GRAY, edgecolor=GRAY, lw=1.2, radius=0.9, **kw):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        **kw,
    )
    ax.add_patch(patch)
    return patch


def labelled_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle=None,
    facecolor=LIGHT_GRAY,
    edgecolor=GRAY,
    title_size=7.5,
    sub_size=6.5,
    color=BLACK,
):
    box(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor)
    cx = x + w / 2
    if subtitle:
        ax.text(
            cx,
            y + h * 0.62,
            title,
            fontsize=title_size,
            ha="center",
            va="center",
            fontweight="bold",
            color=color,
        )
        ax.text(
            cx,
            y + h * 0.30,
            subtitle,
            fontsize=sub_size,
            ha="center",
            va="center",
            color=GRAY,
        )
    else:
        ax.text(
            cx,
            y + h / 2,
            title,
            fontsize=title_size,
            ha="center",
            va="center",
            fontweight="bold",
            color=color,
        )


def arrow(
    ax, x1, y1, x2, y2, color=GRAY, lw=1.6, style="-|>", mutation=16, linestyle="-"
):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle=style,
            mutation_scale=mutation,
            color=color,
            linewidth=lw,
            linestyle=linestyle,
            shrinkA=0,
            shrinkB=0,
        )
    )


def tag(ax, x, y, text, kind="V"):
    """Small evidence tag. Use on any panel mixing measured and projected values."""
    colors = {"V": GREEN, "A": AMBER, "U": RED}
    ax.text(
        x,
        y,
        f"[{kind}] {text}",
        fontsize=6.5,
        color=colors.get(kind, GRAY),
        ha="left",
        va="center",
        style="italic",
    )


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


def save_png(fig, outdir: pathlib.Path, name: str, dpi: int = 220) -> pathlib.Path:
    """PNG alongside the PDF, for quick visual inspection only. Not used by the manuscript."""
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{name}.png"
    fig.savefig(path, dpi=dpi)
    return path
