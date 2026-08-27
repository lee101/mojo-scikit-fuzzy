"""Fuzzy c-means clustering with the scikit-fuzzy calling convention."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


def _validate(
    data, clusters: int, m: float, metric: str, *, limit_clusters: bool = True
) -> np.ndarray:
    values = f64(data)
    if values.ndim != 2:
        raise ValueError("data must be a 2-D array shaped (features, samples)")
    if 0 in values.shape:
        raise ValueError("data must contain at least one feature and one sample")
    if clusters < 1 or (limit_clusters and clusters > values.shape[1]):
        raise ValueError("c must be between 1 and the number of samples")
    if not np.isfinite(m) or m <= 1:
        raise ValueError("m must be greater than 1")
    if metric != "euclidean":
        raise NotImplementedError("the covered metric subset is metric='euclidean'")
    return values


def _initial_membership(c: int, n: int, init, seed):
    if init is None:
        if seed is not None:
            np.random.seed(seed=seed)
        u0 = np.random.rand(c, n)
        u0 /= np.sum(u0, axis=0, keepdims=True)
    else:
        u0 = f64(init, copy=True)
        if u0.shape != (c, n):
            raise ValueError(f"init must have shape {(c, n)}")
    if not np.all(np.isfinite(u0)) or np.any(u0 < 0):
        raise ValueError("init must contain finite, non-negative memberships")
    if np.any(u0.sum(axis=0) == 0):
        raise ValueError("each init column must contain a positive membership")
    return u0, np.fmax(u0, np.finfo(np.float64).eps)


def _integer(name, value, *, minimum):
    converted = int(value)
    if converted != value or converted < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return converted


def cmeans(data, c, m, error, maxiter, metric="euclidean", init=None, seed=None):
    """Fuzzy c-means clustering, API-compatible with ``skfuzzy.cluster.cmeans``."""
    c = _integer("c", c, minimum=1)
    maxiter = _integer("maxiter", maxiter, minimum=1)
    m = float(m)
    error = float(error)
    if not np.isfinite(error) or error < 0:
        raise ValueError("error must be finite and non-negative")
    values = _validate(data, c, m, metric)
    features, samples = values.shape
    u0, u = _initial_membership(int(c), samples, init, seed)
    centers = np.empty((c, features), dtype=np.float64)
    distances = np.empty((c, samples), dtype=np.float64)
    um = np.empty_like(u)
    objective = []

    function = lib().msf_cmeans_step
    convergence = lib().msf_normdiff_and_copy
    previous = u.copy() if error > 0 else None
    p = 0
    while p < maxiter:
        objective.append(
            function(
                addr(values), addr(u), addr(centers), addr(distances), addr(um),
                c, features, samples, m,
            )
        )
        p += 1
        if previous is not None and convergence(
            addr(u), addr(previous), u.size
        ) < error:
            break

    fpc = lib().msf_fpc(addr(u), c, samples)
    return centers, u, u0, distances, np.asarray(objective), p, fpc


def cmeans_predict(
    test_data, cntr_trained, m, error, maxiter, metric="euclidean", init=None, seed=None
):
    """Predict memberships from fixed centers, matching scikit-fuzzy."""
    maxiter = _integer("maxiter", maxiter, minimum=1)
    m = float(m)
    error = float(error)
    if not np.isfinite(error) or error < 0:
        raise ValueError("error must be finite and non-negative")
    centers = f64(cntr_trained)
    if centers.ndim != 2:
        raise ValueError("cntr_trained must be a 2-D array")
    if 0 in centers.shape:
        raise ValueError("cntr_trained must not be empty")
    c, features = centers.shape
    values = _validate(test_data, c, m, metric, limit_clusters=False)
    if values.shape[0] != features:
        raise ValueError("test_data and cntr_trained have different feature counts")
    samples = values.shape[1]
    u0, u = _initial_membership(c, samples, init, seed)
    distances = np.empty((c, samples), dtype=np.float64)
    um = np.empty_like(u)
    objective = []

    function = lib().msf_cmeans_predict_step
    cached_function = lib().msf_cmeans_predict_cached_step
    convergence = lib().msf_normdiff_and_copy
    previous = u.copy() if error > 0 else None
    p = 0
    while p < maxiter:
        if p == 0:
            objective.append(
                function(
                    addr(values), addr(centers), addr(u), addr(distances), addr(um),
                    c, features, samples, m,
                )
            )
        else:
            objective.append(
                cached_function(
                    addr(u), addr(distances), addr(um), c, samples, m,
                )
            )
        p += 1
        if previous is not None and convergence(
            addr(u), addr(previous), u.size
        ) < error:
            break

    fpc = lib().msf_fpc(addr(u), c, samples)
    return u, u0, distances, np.asarray(objective), p, fpc
