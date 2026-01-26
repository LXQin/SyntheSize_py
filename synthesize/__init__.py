"""
SyntheSize: A Python package for optimizing sample size in supervised machine learning.

This package implements the SyntheSize algorithm for determining optimal sample sizes
for classification tasks using bulk transcriptomic sequencing data.
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("synthesize")
except PackageNotFoundError:
    __version__ = "unknown"

from .core import (
    # Classification methods
    LOGIS,
    SVM,
    KNN,
    RF,
    XGB,
    # Evaluation functions
    eval_classifier,
    heatmap_eval,
    UMAP_eval,
    vis_classifier,
    # Utility functions
    fit_curve,
    get_data_metrics,
    visualize,
)

__all__ = [
    "LOGIS",
    "SVM",
    "KNN",
    "RF",
    "XGB",
    "eval_classifier",
    "heatmap_eval",
    "UMAP_eval",
    "vis_classifier",
    "fit_curve",
    "get_data_metrics",
    "visualize",
]
