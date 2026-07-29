"""Mojo-accelerated fuzzy clustering and Mamdani inference."""

from . import cluster, control
from .cluster import cmeans, cmeans_predict
from .defuzzify import (
    EmptyMembershipError,
    InconsistentMFDataError,
    arglcut,
    bisector,
    centroid,
    dcentroid,
    defuzz,
    interp_membership,
    interp_universe,
    lambda_cut,
    lambda_cut_series,
)
from .membership import (
    dsigmf,
    gauss2mf,
    gaussmf,
    gbellmf,
    pimf,
    psigmf,
    sigmf,
    smf,
    trapmf,
    trimf,
    zmf,
)

__version__ = "0.1.0"

__all__ = [
    "cluster", "control", "cmeans", "cmeans_predict", "defuzz", "centroid",
    "dcentroid", "bisector", "interp_membership", "interp_universe",
    "lambda_cut", "lambda_cut_series", "arglcut", "trimf", "trapmf",
    "gaussmf", "gauss2mf", "gbellmf", "sigmf", "dsigmf", "psigmf",
    "smf", "zmf", "pimf",
]
