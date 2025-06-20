"""
ggpubpy: matplotlib Based Publication-Ready Plots

A Python library that provides easy-to-use functions for creating and customizing 
matplotlib-based, publication-ready plots with built-in statistical tests and 
automatic p-value or significance star annotations.

This project is directly inspired by R's ggpubr package.
"""

__version__ = "0.1.0"
__author__ = "Izzet Turkalp Akbasli"
__email__ = "izzetakbasli@gmail.com"

from .plots import (
    plot_violin_with_stats,
    plot_boxplot_with_stats,
    significance_stars,
    DEFAULT_PALETTE,
)
from . import datasets

# Convenient aliases
violinggplot = plot_violin_with_stats
boxggplot = plot_boxplot_with_stats

# Import dataset functions
from .datasets import load_iris, get_iris_palette, list_datasets

__all__ = [
    "plot_violin_with_stats",
    "plot_boxplot_with_stats",
    "violinggplot",
    "boxggplot", 
    "significance_stars",
    "DEFAULT_PALETTE",
    "datasets",
    "load_iris",
    "get_iris_palette", 
    "list_datasets",
]
