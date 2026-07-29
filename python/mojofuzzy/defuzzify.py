"""Defuzzification and interpolation functions matching scikit-fuzzy."""

from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib


class EmptyMembershipError(AssertionError):
    pass


class InconsistentMFDataError(AssertionError):
    pass


def _pair(x, mfx):
    universe = f64(x).ravel()
    membership = f64(mfx).ravel()
    if universe.size != membership.size:
        raise InconsistentMFDataError()
    if universe.size == 0:
        raise ValueError("x and mfx must not be empty")
    return universe, membership


def centroid(x, mfx):
    universe, membership = _pair(x, mfx)
    return lib().msf_centroid(addr(universe), addr(membership), universe.size)


def dcentroid(x, mfx, x0):
    universe = f64(x) - x0
    return x0 + centroid(universe, mfx)


def bisector(x, mfx):
    universe, membership = _pair(x, mfx)
    return lib().msf_bisector(addr(universe), addr(membership), universe.size)


def defuzz(x, mfx, mode):
    universe, membership = _pair(x, mfx)
    mode = mode.lower()
    if "centroid" in mode or "bisector" in mode:
        if membership.sum() == 0:
            raise EmptyMembershipError()
        return centroid(universe, membership) if "centroid" in mode else bisector(universe, membership)
    maximum = membership.max()
    points = universe[membership == maximum]
    if "mom" in mode:
        return np.mean(points)
    if "som" in mode:
        return np.min(points)
    if "lom" in mode:
        return np.max(points)
    raise ValueError(f"The input for `mode`, {mode}, was incorrect.")


def interp_membership(x, xmf, xx, zero_outside_x=True):
    universe, membership = _pair(x, xmf)
    if universe.size < 2:
        raise ValueError("x must contain at least two points")
    if np.any(np.diff(universe) <= 0):
        raise ValueError("x must be strictly increasing")
    query = f64(np.atleast_1d(xx))
    result = np.empty_like(query)
    if query.size:
        lib().msf_interp_membership(
            addr(universe), addr(membership), addr(query), addr(result),
            universe.size, query.size, int(zero_outside_x),
        )
    reshaped = result.reshape(np.shape(xx))
    return float(reshaped) if np.ndim(xx) == 0 else reshaped


def interp_universe(x, xmf, y):
    universe, membership = _pair(x, xmf)
    mask = membership > y if y == 0 else membership >= y
    indices = np.where(np.diff(mask))[0]
    points = universe[indices] + (
        (y - membership[indices])
        * (universe[indices + 1] - universe[indices])
        / (membership[indices + 1] - membership[indices])
    )
    return list(set(points.tolist()))


def lambda_cut(ms, lcut):
    membership = np.asarray(ms)
    return ((membership >= lcut) if lcut == 1 else (membership > lcut)).astype(int)


def arglcut(ms, lambdacut):
    return np.nonzero(lambdacut <= np.asarray(ms))


def lambda_cut_series(x, mfx, n):
    universe, membership = _pair(x, mfx)
    levels = np.linspace(membership.min(), membership.max(), n)
    result = np.zeros((n, 3))
    result[:, 0] = levels
    apex = np.nonzero(membership == membership.max())[0][0]
    left = universe[:apex + 1][membership[:apex + 1] == membership[:apex + 1].min()].max()
    right = universe[apex:][membership[apex:] == membership[apex:].min()].min()
    result[0, 1:] = [left, right]
    for i, level in enumerate(levels[1:], start=1):
        indices = np.nonzero(membership >= level - 1e-6)[0]
        result[i, 1:] = universe[indices[[0, -1]]]
    return result
