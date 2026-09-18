"""A small shared visual language, so every figure in the project matches."""
from __future__ import annotations

BG = "#FBFBFA"
FG = "#1A1A18"
MUTED = "#6B6B66"
GRID = "#DDDDD8"

ACCENT = "#C4553B"        # the vehicle / primary signal
REFERENCE = "#6B8FA3"     # commanded reference
SLIDER = "#7A6BA8"        # slider / actuator channels
LIMIT = "#B03A2E"         # limits and saturation
OK = "#4E8A6B"            # margins and healthy values

SERIES = [ACCENT, REFERENCE, OK, SLIDER, "#A88A3F", MUTED]


def apply(plt) -> None:
    plt.rcParams.update({
        "figure.facecolor": BG,
        "axes.facecolor": BG,
        "savefig.facecolor": BG,
        "axes.edgecolor": GRID,
        "axes.labelcolor": FG,
        "axes.titlecolor": FG,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "text.color": FG,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
        "figure.dpi": 130,
        "font.family": "sans-serif",
    })
