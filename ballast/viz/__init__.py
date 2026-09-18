"""Plotting and 3-D animation.

Requires matplotlib, which is an optional dependency::

    pip install "ballast[viz]"
"""
from .plots import dashboard, plot_modes, plot_trajectory_3d, plot_wind_envelope
from .animate import animate, save_animation

__all__ = ["dashboard", "plot_trajectory_3d", "plot_modes", "plot_wind_envelope",
           "animate", "save_animation"]
