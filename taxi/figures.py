"""Shared look and file handling for the report figures (reports/figures/*.png).

Every figure: PNG at 200 dpi, axis labels with units, recessive grid and axes, and
categorical colors in a fixed order (validated for color vision deficiency in that order).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from taxi import config

DPI = 200

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots, always used in this order: blue, orange, aqua, yellow, ...
SERIES = [
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
]
# One-hue blue ramp for magnitude (light = little, dark = much).
SEQUENTIAL_BLUE = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SECONDARY,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 9,
            "axes.grid": True,
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "grid.linestyle": "-",
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "xtick.labelcolor": INK_SECONDARY,
            "ytick.labelcolor": INK_SECONDARY,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "legend.labelcolor": INK_SECONDARY,
            "font.family": "sans-serif",
            "lines.linewidth": 2,
        }
    )


def subplots(width: float, height: float, **kwargs) -> tuple[plt.Figure, plt.Axes]:
    """A figure of width x height inches with constrained layout, in the shared style."""
    apply_style()
    return plt.subplots(figsize=(width, height), layout="constrained", **kwargs)


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save to reports/figures/<name>.png (reports/dev/figures in a dev run) and close it."""
    path = config.figures_dir() / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path
